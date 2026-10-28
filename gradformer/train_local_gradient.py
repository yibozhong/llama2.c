"""Single-GPU local input-gradient constraint pilot with matched CE control."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from gradformer.local_gradient import backward, gradient_validation
from gradformer.local_ce import clip_local_gradients, validation_metrics
from gradformer.runtime import construct, optimizer_for
from gradformer.train import atomic_save, batch
from gradformer.train_local import configure_parameter_lr


def validate_resume(previous, current):
    for key in ("alpha", "beta", "parameter_lr", "hidden_step_size", "seed", "model", "dataset",
                "source", "schedule_steps", "clipping", "batch_tokens", "target", "ce_implementation"):
        if previous[key] != current[key]:
            raise ValueError("Cannot change " + key + " on resume")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--alpha", type=float, default=0.); p.add_argument("--beta", type=float, default=1.)
    p.add_argument("--parameter-lr", type=float, default=.0005)
    p.add_argument("--hidden-step-size", type=float, default=1.)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--stop-after-step", type=int, default=1000)
    p.add_argument("--resume", action="store_true"); p.add_argument("--label", default="")
    args = p.parse_args()
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        p.error("Single GPU only")
    if not all(math.isfinite(v) and v >= 0 for v in (args.alpha, args.beta)):
        p.error("Nonnegative finite auxiliary coefficients required")
    if not all(math.isfinite(v) and v > 0 for v in (args.parameter_lr, args.hidden_step_size)):
        p.error("Positive finite LR and eta required")
    if not 0 < args.stop_after_step <= 6104:
        p.error("Invalid endpoint")
    torch.cuda.set_device(0); device = torch.device("cuda")
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    train = np.memmap(data / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(data / "val.bin", dtype=np.uint16, mode="r")
    model = construct("llama2", "sgd", depth=6, width=288, heads=6,
                      vocab=manifest["vocab_size"], sequence=256, step_size=args.hidden_step_size, device=device)
    optimizer = optimizer_for(model, "llama2", 32768)
    configure_parameter_lr(optimizer, args.parameter_lr)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "source": json.loads(Path("gradformer/source.json").read_text()),
              "dataset": manifest, "model": asdict(model.params), "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "parameters": sum(p.numel() for p in model.parameters()),
              "batch_tokens": 32768, "batch": 16, "sequence": 256, "accumulation": 8,
              "schedule_steps": 6104, "warmup_steps": 100,
              "target": "detached gradient of post-update local mean-token CE w.r.t. local input; includes current block",
              "objective": "mean across layers of CE + alpha*raw_MSE + beta*(1-cosine)",
              "target_scale": "unweighted microbatch mean-token CE; not scaled by depth/accumulation",
              "gradient_boundary": "hidden detached between blocks; no downstream CE or auxiliary gradient",
              "readout": "shared norm/head trained by final CE; tied embedding also receives first-layer CE and auxiliary input paths",
              "clipping": "norm 1 independently per block+eta; separate tied head+norm group",
              "ce_implementation": "CCE full vocabulary, no filtering, fp32 accumulation",
              "cce_commit": "3de376c106a1916bc5e1b619f9c77c87a461ee1c",
              "diagnostics": "weighted CE/aux parameter gradients on block matrices, first microbatch of logged steps; excludes eta/readout",
              "optimizer_groups": [{k:v for k,v in g.items() if k != "params"} for g in optimizer.param_groups]}
    latest_path = out / "latest.json"
    start, elapsed, best = 0, 0., float("inf")
    if args.resume:
        validate_resume(json.loads((out / "config.json").read_text()), config)
        latest = json.loads(latest_path.read_text()); folder = out / latest["directory"]
        model.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=True))
        state = torch.load(folder / "rank-0.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["data_rng"]
        torch.set_rng_state(state["torch_rng"].cpu()); torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        start, elapsed, best = state["step"], state["training_seconds"], state["best_val"]
    elif latest_path.exists():
        raise ValueError("Existing output requires resume")
    import wandb
    run_id = (out / "wandb-id.txt").read_text().strip() if args.resume else wandb.util.generate_id()
    (out / "wandb-id.txt").write_text(run_id); (out / "config.json").write_text(json.dumps(config, indent=2))
    run = wandb.init(project="nanochat", entity="yibozhong657-none", id=run_id, resume="allow",
                     name=f"local-gradient-mse{args.alpha:g}-cos{args.beta:g}-eta{args.hidden_step_size:g}-lr{args.parameter_lr:g}-{args.label}",
                     group="gradformer-local-gradient-20261028", job_type="pretrain", config=config)
    run.define_metric("step"); run.define_metric("*", step_metric="step")
    handle = (out / "metrics.jsonl").open("a")

    def emit(row):
        if any(not math.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
            raise RuntimeError("Nonfinite metric")
        handle.write(json.dumps(row) + "\n"); handle.flush()
        run.log(row); print(json.dumps(row), flush=True)

    def evaluate(step):
        vrng = np.random.default_rng(987654)
        values = validation_metrics(model, (batch(val, vrng, 16, 256, device) for _ in range(32)))
        vrng = np.random.default_rng(987654)
        stats = gradient_validation(model, (batch(val, vrng, 16, 256, device) for _ in range(8)))
        emit(dict(step=step, tokens=step*32768, training_seconds=elapsed, **values, **stats))
        return values["val_ce"]

    def save(step):
        previous = json.loads(latest_path.read_text())["directory"] if latest_path.exists() else "checkpoint-1"
        folder = out / ("checkpoint-1" if previous == "checkpoint-0" else "checkpoint-0")
        folder.mkdir(exist_ok=True)
        atomic_save(model.state_dict(), folder / "model.pt")
        atomic_save(dict(optimizer=optimizer.state_dict(), data_rng=rng.bit_generator.state,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                         step=step, training_seconds=elapsed, best_val=best), folder / "rank-0.pt")
        tmp = out / "latest.json.tmp"; tmp.write_text(json.dumps(dict(directory=folder.name, step=step))); tmp.replace(latest_path)

    code = 1
    try:
        value = evaluate(start); best = min(best, value)
        torch.cuda.reset_peak_memory_stats()
        for step in range(start, args.stop_after_step):
            factor = (step+1)/100 if step < 100 else .5*(1+math.cos(math.pi*(step-100)/(6104-100)))
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); begin = time.perf_counter()
            losses, statistics, signals = [], [], None
            log_step = step == start or (step+1) % 10 == 0
            for micro in range(8):
                x, y = batch(train, rng, 16, 256, device)
                loss, stats, signal = backward(model, x, y, alpha=args.alpha, beta=args.beta,
                                               accumulation=8, diagnostics=log_step and micro == 0)
                losses.append(loss); statistics.append(stats)
                if signal is not None:
                    signals = signal
            layer_norms, shared_norm = clip_local_gradients(model)
            optimizer.step()
            torch.cuda.synchronize(); duration = time.perf_counter()-begin; elapsed += duration
            losses, statistics = torch.stack(losses).mean(0), torch.stack(statistics).mean(0)
            if not torch.isfinite(losses).all() or not torch.isfinite(statistics).all():
                raise RuntimeError("Nonfinite loss or gradient alignment")
            if log_step:
                row = dict(step=step+1, tokens=(step+1)*32768, train_ce=losses[-1, 0].item(),
                           train_local_ce=losses[:, 0].mean().item(), train_aux=losses[:, 1].mean().item(),
                           parameter_lr=optimizer.param_groups[0]["lr"], step_seconds=duration,
                           training_seconds=elapsed, tokens_per_second=32768/duration,
                           peak_memory_gb=torch.cuda.max_memory_allocated()/2**30,
                           shared_gradient_norm=shared_norm.item())
                for i in range(6):
                    row.update({f"local/layer_{i}/ce":losses[i, 0].item(),
                                f"local/layer_{i}/gradient_norm":layer_norms[i].item(),
                                f"local/layer_{i}/eta":torch.nn.functional.softplus(model.hidden_step_raw[i]).item()})
                    for j, name in enumerate(("mse", "cosine", "prediction_norm", "target_norm", "directional_fraction")):
                        row[f"train_gradient/layer_{i}/{name}"] = statistics[i, j].item()
                    for j, name in enumerate(("ce_norm", "aux_norm", "aux_ce_ratio", "cosine")):
                        row[f"micro0_parameter_gradient/layer_{i}/{name}"] = signals[i, j].item()
                emit(row)
            if (step+1) % 250 == 0 or step+1 == args.stop_after_step:
                value = evaluate(step+1); best = min(best, value)
            if step+1 == args.stop_after_step:
                save(step+1)
        run.summary.update(last_step=args.stop_after_step, pilot_tokens=args.stop_after_step*32768,
                           pilot_val_ce=value, best_val_ce=best, training_seconds=elapsed)
        code = 0
    finally:
        handle.close(); run.finish(exit_code=code)


if __name__ == "__main__":
    main()
