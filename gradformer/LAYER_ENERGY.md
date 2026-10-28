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

Implementation and test results: pending GPU gate at initial commit.
