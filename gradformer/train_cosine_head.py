"""Single-GPU cosine-block / NTP-readout experiment with matched controls."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from gradformer.cosine_head import backward, clip_gradients
from gradformer.cosine_probe import validation_metrics
from gradformer.runtime import construct, optimizer_for, loss_forward
from gradformer.train import atomic_save, batch
from gradformer.train_local import configure_parameter_lr


def validate_resume(previous, current):
    for key in ("mode", "last_eta", "parameter_lr", "hidden_step_size", "seed", "model",
                "dataset", "source", "schedule_steps", "clipping", "batch_tokens", "ce_implementation"):
        if previous[key] != current[key]:
            raise ValueError("Cannot change " + key + " on resume")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["cosine-head", "cosine-only", "ntp-only"], default="cosine-head")
    p.add_argument("--last-eta", choices=["ntp", "fixed"], default="ntp")
    p.add_argument("--parameter-lr", type=float, default=.0005)
    p.add_argument("--hidden-step-size", type=float, default=.01)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--stop-after-step", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--label", default="")
    args = p.parse_args()
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        p.error("Single GPU only")
    if not all(math.isfinite(v) and v > 0 for v in (args.parameter_lr, args.hidden_step_size)):
        p.error("Positive finite LR and eta required")
    if args.mode == "cosine-only" and args.last_eta != "fixed":
        p.error("Cosine-only cannot train the last eta")
    if args.mode == "ntp-only" and args.last_eta != "ntp":
        p.error("The NTP-only control trains all step sizes")
    if not 0 < args.stop_after_step <= 6104:
        p.error("Invalid endpoint")
    torch.cuda.set_device(0); device = torch.device("cuda")
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    train = np.memmap(data / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(data / "val.bin", dtype=np.uint16, mode="r")
    model = construct("llama2", "adam", depth=6, width=288, heads=6,
                      vocab=manifest["vocab_size"], sequence=256, step_size=args.hidden_step_size, device=device)
    optimizer = optimizer_for(model, "llama2", 32768)
    configure_parameter_lr(optimizer, args.parameter_lr)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "source": json.loads(Path("gradformer/source.json").read_text()),
              "dataset": manifest, "model": asdict(model.params), "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "parameters": sum(p.numel() for p in model.parameters()),
              "batch_tokens": 32768, "batch": 16, "sequence": 256, "accumulation": 8,
              "schedule_steps": 6104, "warmup_steps": 100, "target_source": "online; no EMA",
              "target": "unscaled per-microbatch mean NTP input gradient, detached",
              "block_signal": "NTP" if args.mode == "ntp-only" else "cosine only, connected across blocks",
              "direct_head_signal": "none" if args.mode == "cosine-only" else "NTP coefficient 1",
              "cosine_coefficient": 0 if args.mode == "ntp-only" else 1, "mse_coefficient": 0,
              "tied_embedding": "cosine input path plus direct CE head path; no CE input path" if args.mode == "cosine-head" else args.mode,
              "last_eta_signal": args.last_eta,
              "clipping": "separate norm-1 groups: blocks+early eta; tied readout+final norm; final eta",
              "ce_implementation": "native full-vocabulary autocast CE, matching historical cosine-only",
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
                     name=f"gradformer-{args.mode}-eta{args.hidden_step_size:g}-lr{args.parameter_lr:g}-{args.label}",
                     group="gradformer-cosine-head-lr-20261028", job_type="pretrain", config=config)
    run.define_metric("step"); run.define_metric("*", step_metric="step")
    handle = (out / "metrics.jsonl").open("a")

    def emit(row):
        if any(not math.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
            raise RuntimeError("Nonfinite metric")
        handle.write(json.dumps(row) + "\n"); handle.flush()
        run.log(row); print(json.dumps(row), flush=True)

    def evaluate(step):
        model.eval(); vrng = np.random.default_rng(987654)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            losses = [loss_forward(model, "llama2", *batch(val, vrng, 16, 256, device), False)[0].detach() for _ in range(32)]
        vrng = np.random.default_rng(987654)
        stats = validation_metrics(model, (batch(val, vrng, 16, 256, device) for _ in range(8)))
        value = torch.stack(losses).mean().item()
        emit(dict(step=step, tokens=step*32768, val_ce=value, training_seconds=elapsed, **stats))
        model.train(); return value

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
            mult = (step+1)/100 if step < 100 else .5*(1+math.cos(math.pi*(step-100)/6004))
            for g in optimizer.param_groups:
                g["lr"] = g["initial_lr"] * mult
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); begin = time.perf_counter()
            ces, auxes, diagnostics = [], [], []
            for _ in range(8):
                x, y = batch(train, rng, 16, 256, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ce, aux, metric = backward(model, x, y, mode=args.mode,
                                               train_last_eta=args.last_eta == "ntp", accumulation=8)
                ces.append(ce); auxes.append(aux)
                if metric is not None:
                    diagnostics.append(metric)
            norms = clip_gradients(model); optimizer.step()
            torch.cuda.synchronize(); duration = time.perf_counter()-begin; elapsed += duration
            if step == start or (step+1) % 10 == 0:
                row = dict(step=step+1, tokens=(step+1)*32768, train_ce=torch.stack(ces).mean().item(),
                           cosine_loss=torch.stack(auxes).mean().item(), parameter_lr=optimizer.param_groups[0]["lr"],
                           training_seconds=elapsed, step_seconds=duration,
                           peak_memory_gb=torch.cuda.max_memory_allocated()/2**30, **norms)
                if diagnostics:
                    avg = torch.stack(diagnostics).mean(0)
                    for i, values in enumerate(avg.tolist()):
                        row.update({f"prediction/layer_{i}/{k}":v for k,v in zip(("mse", "cosine", "prediction_norm", "target_norm", "valid_fraction"), values)})
                row.update({f"hidden_eta/layer_{i}":v for i,v in enumerate(torch.nn.functional.softplus(model.hidden_step_raw).detach().tolist())})
                emit(row)
            if (step+1) % 250 == 0 or step+1 == args.stop_after_step:
                value = evaluate(step+1); best = min(best, value)
            if (step+1) % 1000 == 0 or step+1 == args.stop_after_step:
                save(step+1)
        run.summary.update(dict(last_step=args.stop_after_step, pilot_tokens=args.stop_after_step*32768,
                                best_val_ce=best, training_seconds=elapsed, completed=args.stop_after_step == 6104))
        code = 0
    finally:
        handle.close(); run.finish(exit_code=code)


if __name__ == "__main__":
    main()
