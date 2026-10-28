"""Same-GPU full-macrostep benchmark; all cases retain hidden Adam."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from gradformer.local_ce import amp_context, local_backward, clip_local_gradients
from gradformer.runtime import construct, optimizer_for


def measure(case, warmup, steps):
    torch.manual_seed(1337)
    model = construct("llama2", "adam", depth=6, width=288, heads=6,
                      vocab=32000, sequence=256, step_size=.01, device="cuda")
    optimizer = optimizer_for(model, "llama2", 32768)
    rng = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randint(32000, (16, 256), device="cuda", generator=rng)
    y = torch.randint(32000, (16, 256), device="cuda", generator=rng)
    times = []
    for step in range(warmup + steps):
        optimizer.zero_grad(set_to_none=True)
        if step == warmup:
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); begin = time.perf_counter()
        for _ in range(8):
            if case == "e2e-adam-ntp":
                with amp_context(model):
                    loss = model(x, y, return_loss=True)
                (loss / 8).backward()
            else:
                loss = local_backward(model, x, y, accumulation=8,
                                      implementation="cce" if case == "local-adam-cce" else "dense").mean()
            assert torch.isfinite(loss).item()
        if case == "e2e-adam-ntp":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        else:
            clip_local_gradients(model)
        optimizer.step(); torch.cuda.synchronize()
        if step >= warmup:
            times.append(time.perf_counter() - begin)
    return dict(case=case, timings=times, median_step_seconds=statistics.median(times),
                peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    p.add_argument("--warmup", type=int, default=3); p.add_argument("--steps", type=int, default=10)
    args = p.parse_args()
    import wandb
    run = wandb.init(project="nanochat", entity="yibozhong657-none", job_type="benchmark",
                     name="gradformer-local-ce-all-token-benchmark",
                     config={**vars(args), "gpu": torch.cuda.get_device_name(), "accumulation": 8,
                             "source": json.loads(Path('gradformer/source.json').read_text())})
    exit_code = 1
    try:
        rows = []
        for case in ("e2e-adam-ntp", "local-adam-dense", "local-adam-cce"):
            row = measure(case, args.warmup, args.steps)
            row["relative_to_e2e_adam"] = row["median_step_seconds"] / (rows[0]["median_step_seconds"] if rows else row["median_step_seconds"])
            rows.append(row); print(json.dumps(row), flush=True)
            run.log({k: v for k, v in row.items() if k != "timings"})
            gc.collect(); torch.cuda.empty_cache()
        result = dict(gpu=torch.cuda.get_device_name(), results=rows,
                      passed=rows[-1]["relative_to_e2e_adam"] < 10)
        Path(args.output).write_text(json.dumps(result, indent=2)); run.summary.update(result)
        if not result["passed"]:
            raise RuntimeError("Fused local CE exceeds 10x end-to-end hidden-Adam cost gate")
        exit_code = 0
    finally:
        run.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
