# Ordinary Transformer parameter learning-rate sweep

Authorized 2026-10-28 after the per-layer energy pilot. Sweep the actual
standard residual Transformer, not hidden-SGD/Adam or local learning.

Reuse the exact immutable training source8803aef / llama2-energy-v1 bundle,
uv-v4 container and existing launcher with kind=baseline. That path constructs
the ordinary Transformer and uses its ordinary residual blocks. There is no
learned hidden step or hidden optimizer. The launcher's required eta=.01 is
ignored by this baseline; inherited energy-description metadata fields do not
select its architecture. The authoritative fields are kind=baseline and
model.hidden_update=baseline. Output directory energy is a legacy launcher
name, not an energy model.

Keep6 layers, D288,6 heads,15,191,712 parameters, tied embedding/head, same
TinyStories/tokenizer/data, seed1337, sequence256, microbatch16, accumulation8,
32768 tokens/step. Final full-vocabulary CCE coefficient1, global norm1 clip,
parameter AdamW, warmup100 and original6104-step cosine schedule. Each trial
trains1000 steps /32,768,000 tokens. Fixed validation sample131072 tokens.

First coarse points: parameter peak LR .00005, .0005, .005. Reuse the completed
.0005 baseline iyh6yugm from7336756.0 (CE1.793137), preserving its provenance;
submit the other two independently in parallel. Each new run includes the
existing50-step finite/checkpoint gate and restoration before continuing.
Refine around the best coarse point if needed, with the same budget per run.
Report all trials; no automatic longer training or energy-model expansion.

All runs log to W&B project nanochat, group gradformer-layer-energy-20261028,
with explicit baseline-lr labels. The existing group keeps this comparison
together. Different GPU types do not support direct raw wall-time comparisons.
Estimated first-phase additional storage about730MB; home usage before launch
35094MiB against40960MiB soft quota. Training implementation is unchanged, so
no repeated mathematical tests are needed for this parameter-only sweep.

Completed results and evidence follow.


## Completed result record

Five parameter learning rates, each1000 steps /32,768,000 tokens:four new independent GPU jobs plus the existing .0005 reference. All completed normally, with W&B finished and source/config/data/seed/model/parameter count/LR schedule/token counts/finite metrics/checkpoint status cross-checked.

The model is the ordinary residual Transformer:6 layers, D288,6 heads,15,191,712 parameters, tied embedding/head. Exact training source8803aef, immutable llama2-energy-v1 bundle, uv-v4 container, same TinyStories/tokenizer/data, seed1337 and fixed131072-token validation sample. Final full-vocabulary CCE coefficient1, global norm1 clipping, parameter AdamW, sequence256, microbatch16, accumulation8,32768 tokens/step, warmup100 and original6104-step cosine schedule. No hidden optimizer or local losses.

## Final validation CE

| Peak parameter LR | Final CE | Best observed CE | Training seconds | GPU | W&B run |
| --- | ---: | ---: | ---: | --- | --- |
| 5e-05 | 2.988640 | 2.988640 | 76.108 | RTX PRO 6000 Blackwell Server Edition | [ogc1kue0](https://wandb.ai/yibozhong657-none/nanochat/runs/ogc1kue0) |
| 0.0005 | 1.793137 | 1.793137 | 75.251 | RTX PRO 6000 Blackwell Server Edition | [iyh6yugm](https://wandb.ai/yibozhong657-none/nanochat/runs/iyh6yugm) |
| 0.0015 | 1.680402 | 1.680402 | 75.328 | RTX PRO 6000 Blackwell Server Edition | [v0jjrscm](https://wandb.ai/yibozhong657-none/nanochat/runs/v0jjrscm) |
| 0.005 | 1.696200 | 1.696200 | 75.181 | RTX PRO 6000 Blackwell Server Edition | [w9nb4mqw](https://wandb.ai/yibozhong657-none/nanochat/runs/w9nb4mqw) |
| 0.015 | 3.059543 | 3.059543 | 97.163 | L40 | [o1h1g4kc](https://wandb.ai/yibozhong657-none/nanochat/runs/o1h1g4kc) |

Best tested LR is 0.0015:CE1.680402, improving the original .0005 reference by0.112736. LR .005 is close (CE1.696200), while .015 gives much poorer convergence and nonmonotone validation. The useful tested region is approximately .0015–.005 for this particular1000-step budget; this single-seed screen does not establish a universal optimum for longer pretraining.

The current best energy run has CE1.959035, so its gap to the tuned ordinary baseline is0.278634. Both have the same token/step budget; the energy result is from the previously completed energy grid, not a newly expanded sweep.

Training seconds exclude process startup/compilation/evaluation/checkpointing. The original .0005 baseline, .0015 and .005 runs all used RTX PRO6000 Blackwell Server Edition and each required about75 seconds of training. The .015 trial used L40; its raw seconds are not a matched-device performance comparison.

## Provenance and verification

Coarse sweep:7336861.0/.1 at .00005/.005, reusing7336756.0/iyh6yugm at .0005. Because .005 was the best coarse point, refinement7336862.0/.1 tested .0015/.015. All four new jobs exited0. The reused reference is explicitly marked in baseline-lr-submissions.json. New runs passed the50-step finite/checkpoint gate and validation-after-restore check (tolerance1e-6); the final saved model tensors were checked against in-memory parameters on the GPU.

Auditor:repo gradformer/chtc/experiments/audit_baseline_lr.py. Evidence:baseline-lr-audit.json, baseline-lr-wandb.json, baseline-lr-submissions.json; all are metadata-only checks on the access point. The reused launcher stores new outputs under a directory named energy and carries legacy energy-description fields, but both kind and model.hidden_update are baseline, and no hidden_eta metric exists. The required eta argument is ignored by the baseline path.

All baseline LR jobs have finished. No energy-model jobs, longer runs, or extra seeds were submitted. The unrelated old eight-GPU job7310803 remains idle.
