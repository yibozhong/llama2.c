"""Single-GPU strict-local CE pilot, all tokens, shared head, hidden Adam/SGD."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from gradformer.local_ce import local_backward, end_to_end_backward, clip_local_gradients, validation_metrics
from gradformer.runtime import construct, optimizer_for
from gradformer.train import atomic_save, batch


def configure_parameter_lr(optimizer, learning_rate):
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Parameter learning rate must be finite and positive")
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] = learning_rate


def validate_resume(previous, config):
    for key in ("seed", "dataset", "model", "batch_tokens", "objective", "clipping", "ce_implementation", "cce_commit", "schedule_steps", "source"):
        if previous[key] != config[key]:
            raise ValueError(f"Cannot change {key} on local-CE resume")
    for key, default in (("readout_training", "all"), ("parameter_lr", .0005), ("training_objective", "local")):
        if previous.get(key, default) != config[key]:
            raise ValueError(f"Cannot change {key} on resume")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=200000000)
    parser.add_argument("--stop-after-step", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--update", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--readout-training", choices=["all", "final"], default="all")
    parser.add_argument("--hidden-step-size", type=float, default=.01)
    parser.add_argument("--parameter-lr", type=float, default=.0005)
    parser.add_argument("--training-objective", choices=["local", "e2e"], default="local")
    parser.add_argument("--run-group", default="gradformer-llama2-200000000tokens")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in (args.hidden_step_size, args.parameter_lr)):
        parser.error("Hidden step size and parameter LR must be finite and positive")
    if args.training_objective == "e2e" and args.readout_training != "final":
        parser.error("The matched end-to-end control uses final readout training")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        parser.error("This strict-local pilot supports one GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    train = np.memmap(Path(args.data) / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(Path(args.data) / "val.bin", dtype=np.uint16, mode="r")
    model = construct("llama2", args.update, depth=6, width=288, heads=6,
                      vocab=manifest["vocab_size"], sequence=256, step_size=args.hidden_step_size, device=device)
    optimizer = optimizer_for(model, "llama2", 32768)
    configure_parameter_lr(optimizer, args.parameter_lr)
    steps = math.ceil(args.tokens / 32768)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path("gradformer/source.json").read_text())
    config = {**vars(args), "source": source, "dataset": manifest, "model": asdict(model.params),
              "parameters": sum(p.numel() for p in model.parameters()), "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "batch": 16, "sequence": 256, "batch_tokens": 32768,
              "accumulation": 8, "schedule_steps": steps,
              "objective": ("mean of all-layer all-token local CE" if args.training_objective == "local" else "final all-token CE divided by depth; connected hidden path"),
              "hidden_update": args.update,
              "gradient_boundary": ("hidden AND any moments detached between blocks" if args.training_objective == "local" else "connected through all hidden updates"),
              "readout_training": args.readout_training,
              "readout": "shared tied embedding/head and final RMSNorm; " + ("final-layer readout gradients only" if args.readout_training == "final" else "all-layer readout gradients"),
              "tied_embedding_input_gradient": ("first-layer local CE, independently of readout gradient policy" if args.training_objective == "local" else "final CE through all blocks"),
              "local_ce_coefficient": 1 / len(model.layers) if args.training_objective == "local" else None,
              "final_ce_coefficient": 1 / len(model.layers),
              "clipping": "max_norm=1 independently per block+its eta, and separately shared head/norm",
              "ce_implementation": "CCE full vocabulary; no filtering; fp32 gradient accumulation",
              "cce_commit": "3de376c106a1916bc5e1b619f9c77c87a461ee1c",
              "optimizer_groups": [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]}
    start, training_seconds, best = 0, 0., float("inf")
    latest_path = out / "latest.json"
    if args.resume:
        previous = json.loads((out / "config.json").read_text())
        validate_resume(previous, config)
        latest = json.loads(latest_path.read_text()); folder = out / latest["directory"]
        model.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=False))
        state = torch.load(folder / "rank-0.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["data_rng"]
        torch.set_rng_state(state["torch_rng"].cpu()); torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        start, training_seconds, best = state["step"], state["training_seconds"], state["best_val"]
    elif latest_path.exists():
        raise ValueError("Existing output checkpoint requires --resume")
    import wandb
    run_id = ((out / "wandb-id.txt").read_text().strip() if args.resume else wandb.util.generate_id())
    (out / "wandb-id.txt").write_text(run_id)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    run = wandb.init(project="nanochat", entity="yibozhong657-none", id=run_id, resume="allow",
                     name=f"gradformer-{args.training_objective}-{args.update}-{args.readout_training}-eta{args.hidden_step_size:g}-lr{args.parameter_lr:g}-seed{args.seed}" + ("-" + args.run_label if args.run_label else ""),
                     group=args.run_group, tags=["strict-local" if args.training_objective == "local" else "matched-e2e", "all-tokens", "shared-head", "hidden-" + args.update, args.readout_training + "-readout", "cce"],
                     job_type="pretrain", config=config)
    run.define_metric("step"); run.define_metric("*", step_metric="step")
    handle = (out / "metrics.jsonl").open("a")

    def emit(row):
        if any(not math.isfinite(v) for v in row.values() if isinstance(v, (int, float))):
            raise RuntimeError("Nonfinite local-CE metric")
        handle.write(json.dumps(row) + "\n"); handle.flush()
        run.log(row); print(json.dumps(row), flush=True)

    def evaluate(step):
        vrng = np.random.default_rng(987654)
        batches = (batch(val, vrng, 16, 256, device) for _ in range(args.eval_batches))
        metrics = validation_metrics(model, batches)
        emit(dict(step=step, tokens=step * 32768, training_seconds=training_seconds, **metrics))
        return metrics["val_ce"]

    def save(step):
        previous = json.loads(latest_path.read_text())["directory"] if latest_path.exists() else "checkpoint-1"
        folder = out / ("checkpoint-1" if previous == "checkpoint-0" else "checkpoint-0")
        folder.mkdir(exist_ok=True)
        atomic_save(model.state_dict(), folder / "model.pt")
        atomic_save(dict(optimizer=optimizer.state_dict(), data_rng=rng.bit_generator.state,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                         step=step, training_seconds=training_seconds, best_val=best), folder / "rank-0.pt")
        temporary = out / "latest.json.tmp"
        temporary.write_text(json.dumps(dict(directory=folder.name, step=step))); temporary.replace(latest_path)

    exit_code = 1
    try:
        value = evaluate(start); best = min(best, value)
        torch.cuda.reset_peak_memory_stats()
        endpoint = min(args.stop_after_step or steps, steps)
        for step in range(start, endpoint):
            warmup = min(100, max(1, int(.05 * steps)))
            multiplier = ((step + 1) / warmup if step < warmup else
                          .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * multiplier
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); begin = time.perf_counter()
            losses = []
            for _ in range(8):
                x, y = batch(train, rng, 16, 256, device)
                if args.training_objective == "local":
                    losses.append(local_backward(model, x, y, accumulation=8, readout_training=args.readout_training))
                else:
                    losses.append(end_to_end_backward(model, x, y, accumulation=8))
            layer_norms, shared_norm = clip_local_gradients(model)
            optimizer.step()
            torch.cuda.synchronize(); duration = time.perf_counter() - begin
            training_seconds += duration
            losses = torch.stack(losses).mean(0)
            if not torch.isfinite(losses).all():
                raise RuntimeError("Nonfinite local CE")
            if step == start or (step + 1) % 10 == 0:
                row = dict(step=step + 1, tokens=(step + 1) * 32768, train_ce=losses[-1].item(),
                           parameter_lr=optimizer.param_groups[0]["lr"], step_seconds=duration,
                           training_seconds=training_seconds, tokens_per_second=32768 / duration,
                           peak_memory_gb=torch.cuda.max_memory_allocated() / 2**30,
                           shared_gradient_norm=shared_norm.item())
                prefix = "local" if args.training_objective == "local" else "block"
                if args.training_objective == "local":
                    row["train_local_ce"] = losses.mean().item()
                    row.update({f"local/layer_{i}/ce": losses[i].item() for i in range(6)})
                for i in range(6):
                    row.update({f"{prefix}/layer_{i}/gradient_norm": layer_norms[i].item(),
                                f"{prefix}/layer_{i}/step_size": torch.nn.functional.softplus(model.hidden_step_raw[i]).item()})
                emit(row)
            if (step + 1) % args.eval_every == 0 or step + 1 == endpoint:
                value = evaluate(step + 1)
                if value < best:
                    best = value; atomic_save(model.state_dict(), out / "best-model.pt")
            if (step + 1) % args.save_every == 0 or step + 1 == endpoint:
                save(step + 1)
        run.summary.update(dict(last_step=endpoint, pilot_tokens=endpoint * 32768, pilot_val_ce=value,
                                best_val_ce=best, training_seconds=training_seconds, completed=endpoint == steps,
                                stopped_for_resume=endpoint < steps))
        exit_code = 0
    finally:
        handle.close(); run.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
