"""Full-macrostep cost gate for the all-layer EMA suffix target pilot."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from gradformer.balance import for_model
from gradformer.ema import EMATeacher
from gradformer.runtime import construct, optimizer_for, loss_forward


def measure(name, warmup, steps):
    torch.manual_seed(1337)
    baseline, pure, ema = name == "baseline", name.endswith("pure"), name.startswith("ema")
    model = construct("llama2", "baseline" if baseline else "adam", depth=6, width=288,
                      heads=6, vocab=32000, sequence=256, step_size=.01, device="cuda")
    optimizer = optimizer_for(model, "llama2", 32768)
    engine = None if baseline else for_model(model, "llama2", cosine_only=pure)
    teacher = EMATeacher(model, .99) if ema else None
    rng = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randint(32000, (16, 256), device="cuda", generator=rng)
    y = torch.randint(32000, (16, 256), device="cuda", generator=rng)
    times = []
    for step in range(warmup + steps):
        optimizer.zero_grad(set_to_none=True)
        if engine:
            engine.reset()
        if step == warmup:
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); begin = time.perf_counter()
        for _ in range(8):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ntp, hs, ps = loss_forward(model, "llama2", x, y, not baseline)
                targets = teacher.targets(hs, ps, y) if teacher else None
            if engine:
                aux, _ = engine.backward(ntp, hs, ps, alpha=0, beta=1, accumulation=8, target_gradients=targets)
                assert torch.isfinite(aux).item()
            else:
                (ntp / 8).backward()
            assert torch.isfinite(ntp).item()
        if engine:
            engine.finish(accumulation=8, fixed_weight=1, diagnostics=False)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if teacher:
            teacher.update(model)
        torch.cuda.synchronize()
        if step >= warmup:
            times.append(time.perf_counter() - begin)
    result = dict(case=name, median_step_seconds=statistics.median(times), timings=times,
                  peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
                  final_ce=ntp.item(), teacher_updates=teacher.updates if teacher else 0)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=3); parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    import wandb
    run = wandb.init(project="nanochat", entity="yibozhong657-none", job_type="benchmark",
                     name="gradformer-llama2-ema-suffix-benchmark", config={**vars(args), "gpu": torch.cuda.get_device_name(),
                     "source": json.loads(Path('gradformer/source.json').read_text()), "batch_tokens": 32768, "accumulation": 8})
    try:
        rows = []
        for name in ("baseline", "online-joint", "online-pure", "ema-joint", "ema-pure"):
            row = measure(name, args.warmup, args.steps)
            row['slowdown_vs_baseline'] = row['median_step_seconds'] / (rows[0]['median_step_seconds'] if rows else row['median_step_seconds'])
            rows.append(row); run.log({k: v for k, v in row.items() if k != 'timings'})
            print(json.dumps(row), flush=True)
            gc.collect(); torch.cuda.empty_cache()
        result = dict(gpu=torch.cuda.get_device_name(), results=rows,
                      passed=all(r['slowdown_vs_baseline'] < 10 for r in rows))
        Path(args.output).write_text(json.dumps(result, indent=2)); run.summary.update(result)
        if not result['passed']:
            raise RuntimeError('EMA pilot exceeds the 10x baseline cost gate')
    finally:
        run.finish()


if __name__ == "__main__":
    main()
