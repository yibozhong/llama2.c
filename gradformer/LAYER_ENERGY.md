# Per-layer conditional energy pilot

Authorized 2026-10-28. Separate from the cancelled local-CE gradient-supervision experiment.

Each layer has its own scalar energy per token. No gradient prediction targets,
EMA, local CE, or auxiliary loss. Only final full-vocabulary NTP CE, coefficient
1, trains all parameters end to end. Six layers give six hidden optimization
steps; no repeated whole-model refinement loop.

## Math and causality

For each layer, c = RMS(causal_attention(RMS(h))). Define independent candidate
z and conditional energy E_t(z_t; c_t) = sum_j a_j * SiLU(Wz RMS(z_t) + Wc c_t)_j.
Evaluate g_t = partial E_t / partial z_t at z_t = h_t, then h_new = h - eta*g
for SGD, or use bias-corrected Adam moments accumulated across depth. Moments
reset on each forward; beta1=.9, beta2=.999, epsilon=.01. Eta is positive and
learnable per layer via softplus. Independent layer energies are NOT a single
fixed objective and do not guarantee monotone energy or CE descent.

Context is fixed only in the inner partial derivative. Final CE differentiates
through both the candidate and context, including attention. Context is NOT
detached for training. Differentiating the sum of ordinary causal-transformer
token energies directly with respect to the shared h would leak future inputs
through later positions' energies; the independent candidate construction
prevents that. This is a conditional energy, not the total derivative of
E(h; c(h)). Holding c fixed is part of the defined optimization problem.

For s = rsqrt(mean(z*z)+eps), n = z*s*w, u = Wz*n + Wc*c:

- SiLU'(u) = sigmoid(u) * (1 + u * (1-sigmoid(u))).
- q = Wz_transpose * (a * SiLU'(u)), elementwise multiplied by w.
- g = s * (q - z*s*s*mean(q*z)).

This exact analytic gradient is evaluated with normal differentiable PyTorch
operations and compiled per block. Outer backward includes the mixed second
derivatives of the scalar potential. Causal attention, outside the inner
derivative, only needs a normal first backward and retains SDPA/Flash support.
This removes autograd-in-forward overhead, not the mathematical higher-order
dependence. Inference works under inference_mode without labels or autograd.
The scalar energy need not be explicitly materialized to evaluate its exact
gradient; check_energy compares against independent autograd of the scalar.

Signed amplitudes a allow nonconvex potentials. No convexity/convergence claim.
RMS normalization controls scale but its derivative is included exactly, even
the epsilon term. No batch/token averaging enters the inner energy gradient;
duplicating a batch does not change an example's update. The initial eta and
energy scale are hyperparameters, so both hidden eta and parameter LR are swept.

## Comparisons and budget

Same TinyStories/tokenizer/data/seed1337 as prior runs; 6 layers, D=288, 6 heads,
S=256, microbatch16, accumulation8, 32768 tokens per optimizer step. The energy
MLP has width1152 (two D-to-1152 maps plus scalar amplitudes); standard SwiGLU
has width768 (three D-to-768 maps). Parameter counts are approximately equal.
Both have tied embedding/head, global norm1 clipping, full-vocabulary CCE with
filtering disabled, parameter AdamW, and per-block compilation. Baseline uses
the actual standard residual Transformer with no hidden optimizer/learned eta.

Each pilot runs 1000 steps / 32,768,000 tokens, warmup100 and the original6104
step cosine schedule. Fixed validation samples contain131072 tokens. Grid:

| Hidden update | Initial eta | Parameter LR |
| --- | --- | --- |
| SGD | .001 / .01 / .1 | .0005 |
| SGD | .01 | .00005 / .005 |
| Adam | .001 / .01 / .1 | .0005 |
| Standard Transformer | none | .0005 |

Center job runs math/causal/BF16/compiler/CCE/checkpoint tests, a same-device
three-way benchmark, then SGD center50-step recovery gate and1000 steps, then
standard Transformer1000 on the same GPU. After the center gate passes, submit
the seven independent remaining configurations together. Full grid9 training
runs in8 single-GPU jobs. Each output keeps at most two checkpoint slots;
estimated total3.1GB. Check available home quota before submission.

All computation on allocated GPU jobs; pinned uv-v4 container and CCE commit
3de376c. Every check/benchmark/train run logs to W&B project nanochat, group
gradformer-layer-energy-20261028. Compare CE by tokens and SAME-GPU time, report
compilation/startup separately from steady training time, and do not infer
equal wall time from different GPU models. Single-seed short pilots are not a
definitive architecture verdict.

## Verified gate and submission

Training source8803aef, immutable energy-v1 bundle. Center7336756.0 on RTX PRO
6000 Blackwell passed the full GPU gate, W&B check run
[ls6jorn4](https://wandb.ai/yibozhong657-none/nanochat/runs/ls6jorn4):

- FP64 analytic/autograd gradient relative error9.23e-17; outer gradient error
  2.47e-16; directional finite-difference discrepancy2.85e-12.
- Full-model FP32 parameter gradients relative error below7.32e-7; BF16 below
  .00177 against nested autograd. Compiler BF16 max per-parameter relative
  difference.0257 (Adam); CCE versus dense below.0073.
- Both SGD/Adam passed causal-prefix, batch duplication, inference_mode and
  exact next-update checkpoint restoration tests. All parameters received
  finite nonzero final-CE gradients.

Short fixed-input benchmark on the same GPU (4 measured steps after2 warmups,
excluding compilation/startup,32768 tokens/step):

| Model | Parameters | Seconds/step | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| Standard residual | 15,191,712 | .063550 | .5484 |
| Energy SGD | 15,200,358 | .068193 | .6139 |
| Energy Adam | 15,200,358 | .082771 | .8067 |

The SGD and Adam overheads are approximately7% and30% in this short benchmark;
this is not time to equal quality, nor does it include real-data loading.

Center passed the50-step gate (CE7.349141) and exact same validation after resume.
Seven remaining configurations submitted together as7336758.0–6 in the order
listed in energy-cohort.txt. The center subsequently completed1000 steps with
CE2.335076 and81.642 training seconds, W&B
[qgo28ibv](https://wandb.ai/yibozhong657-none/nanochat/runs/qgo28ibv).
The same-GPU ordinary baseline completed1000 steps with CE1.793137 and75.251
training seconds, W&B
[iyh6yugm](https://wandb.ai/yibozhong657-none/nanochat/runs/iyh6yugm).
Center job exited0; output archive returned; source/config/LR schedule/token
counts/50-step resume validation/checkpoint status and finished W&B metrics
were cross-checked. Training seconds exclude startup, evaluation and saving;
recorded run wall seconds are102.917 (energy) and86.376 (baseline), with the
energy50-step process restart included and job-level setup/checks excluded.

Thus this SGD center has an8.5% training-time overhead and worse CE at equal
tokens; no quality/compute benefit is demonstrated. SGD eta.001 also finished
with CE2.778312 on L40 (pl1ojunt); its GPU seconds are not comparable to the
Blackwell baseline. All remaining configurations subsequently completed; see final results below.


## Completed pilot sweep

All8 jobs exited0 and all9 training runs completed1000 steps /32,768,000 tokens. Source/config/data/seed/LR schedules/token counts/finite metrics/saved-checkpoint status and finished W&B final metrics were cross-checked. Full metadata audit in the workspace: /home/yzhong239/gradformer/reports/energy-final-audit.json. The50-step resume CE check used1e-6 tolerance:one Adam run differed by4.768e-7 (one FP32 ULP), all others were identical. The independent allocated-GPU next-update restoration test was exact.

| Hidden update | Initial eta | Parameter LR | Validation CE | GPU | Run |
| --- | ---: | ---: | ---: | --- | --- |
| baseline | none | 0.0005 | 1.793137 | RTX PRO 6000 Blackwell Server Edition | [iyh6yugm](https://wandb.ai/yibozhong657-none/nanochat/runs/iyh6yugm) |
| sgd | 0.001 | 0.0005 | 2.778312 | L40 | [pl1ojunt](https://wandb.ai/yibozhong657-none/nanochat/runs/pl1ojunt) |
| sgd | 0.01 | 0.0005 | 2.335076 | RTX PRO 6000 Blackwell Server Edition | [qgo28ibv](https://wandb.ai/yibozhong657-none/nanochat/runs/qgo28ibv) |
| sgd | 0.1 | 0.0005 | 2.071961 | L40S | [broi1s0i](https://wandb.ai/yibozhong657-none/nanochat/runs/broi1s0i) |
| sgd | 0.01 | 5e-05 | 3.771230 | RTX PRO 6000 Blackwell Server Edition | [07wgt8qa](https://wandb.ai/yibozhong657-none/nanochat/runs/07wgt8qa) |
| sgd | 0.01 | 0.005 | 1.959035 | L40 | [vhnzpfdz](https://wandb.ai/yibozhong657-none/nanochat/runs/vhnzpfdz) |
| adam | 0.001 | 0.0005 | 4.031477 | L40 | [z2re9vqq](https://wandb.ai/yibozhong657-none/nanochat/runs/z2re9vqq) |
| adam | 0.01 | 0.0005 | 2.171038 | L40S | [jlt1qn3r](https://wandb.ai/yibozhong657-none/nanochat/runs/jlt1qn3r) |
| adam | 0.1 | 0.0005 | 2.082594 | RTX PRO 6000 Blackwell Server Edition | [yhh39a5z](https://wandb.ai/yibozhong657-none/nanochat/runs/yhh39a5z) |

Best tested energy is SGD, initial eta.01, parameter LR.005, CE1.959035. Ordinary Transformer CE1.793137 remains better by.165898 at equal tokens. Increasing SGD parameter LR from.0005 to.005 improved CE2.335076 to1.959035; increasing eta from.01 to.1 at LR.0005 improved CE to2.071961. Best Adam of the three eta points is2.082594. All runs' validation CE continued decreasing through1000 steps.

Settings clearly matter. Both SGD learning-rate/eta directions favor the tested upper boundary; the combination eta.1/LR.005 was not tested. Adam parameter LR was fixed at.0005. Thus this grid does not exhaust LR effects or establish a general optimizer ranking. It also has no demonstrated quality/compute advantage over the ordinary Transformer. The best energy ran on L40, so its109.386 seconds must not be directly compared with the Blackwell baseline's75.251 seconds. The matched center/standard pair remains the valid direct wall-time comparison (81.642 versus75.251 training seconds).

No energy jobs remain running or queued. No extra sweep or longer training has been submitted. The unrelated old8-GPU job7310803 remains idle.
