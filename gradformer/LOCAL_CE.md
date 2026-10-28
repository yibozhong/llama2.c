# Strict local CE with optimizer-based hidden updates

The pilot uses six blocks, width 288, six attention heads, a shared 32000-token
readout and RMSNorm, and the existing hidden Adam update. Each block predicts an
update signal. Its local loss evaluates the updated hidden state through the
shared readout, with all token positions and the full vocabulary supervised.
There is no gradient prediction MSE/cosine, no EMA teacher, and no added ordinary
residual variant. The update signal is learned through its task effect and is
not asserted to equal an explicit task gradient.

The training objective is the mean across layers of each layer's mean-token CE.
Both hidden states and hidden-Adam moments are detached at block boundaries.
The moments still retain their forward values across depth. Each block and its
own learnable step size receive only that block's CE gradient. The shared
head/norm receive gradients from all local losses. Since the head is tied to the
embedding, its matrix additionally receives the first block's input-path gradient.
Head/norm therefore couple layers across training steps, even though downstream
CE never backpropagates into upstream blocks.

All gradients accumulate for eight microbatches before one AdamW step. Gradient
clipping at norm 1 is independent for each block together with its step-size
component; shared head/norm are clipped as a separate group. This prevents other
layers' gradient magnitudes from changing a block's clipping factor. Weight
decay and the block's own optimizer history still affect its parameter update.

CCE is pinned at upstream commit
`3de376c106a1916bc5e1b619f9c77c87a461ee1c` (version 25.9.3).
The `local-ce` uv dependency group records the source and transitive dependencies.
Use `uv sync --frozen --extra gpu --group local-ce` to reproduce it.
The cluster job reuses the immutable llama2-uv-v4 SIF, which has the matching
PyTorch 2.9.1+cu128 and Triton 3.5.1 plus GCC, and adds an archived checkout of
that exact CCE commit through PYTHONPATH. The upstream package explicitly
supports source-path use. Runtime versions and input archive hashes are saved.
No changes are installed into the shared image or the access-point environment.

CCE's vocabulary-gradient filtering is disabled. fp32/compensated gradient
accumulation is enabled. The kernel tiles the full vocabulary without storing
the full logits tensor; it does not subsample tokens or vocabulary. GPU checks
compare all-token loss and hidden/head gradients to dense CE, including the
actual 4096-token, width-288, vocabulary-32000 shape. Small-model tests verify
local routing and demonstrate the unwanted moment-path gradient when only
hidden states are detached. They also check accumulated gradients, independent
clipping, inference values, checkpoint recovery, and evaluation side effects.

The benchmark compares end-to-end hidden Adam with one final NTP loss, strict
local hidden Adam with dense CE, and strict local hidden Adam with CCE. All use
the same GPU and eight-microbatch complete steps, with three warmup steps and
ten measured steps. Compilation/warmup is excluded. The pilot proceeds only
after correctness checks and a less-than-10x end-to-end hidden-Adam cost gate.

Training is seed 1337, 16x256 microbatches, 32768 tokens per step, parameter
AdamW lr=0.0005/betas=.9/.95, hidden Adam eta=.01/betas=.9/.999/eps=.01,
with the original 200M-token learning-rate schedule. A 50-step checkpoint is
resumed to 1000 steps (32768000 tokens); there is no automatic full-budget
extension. W&B project is `yibozhong657-none/nanochat`.

Validation uses 32 fixed microbatches for all layers, recording per-layer CE,
the CE change induced by the forward hidden update under the same head, and
the fraction of tokens whose CE improved. These are task-effect metrics, not
gradient-prediction cosine. Inference runs the original hidden-Adam forward
and evaluates only the final head; no labels or backward passes are needed.

## SGD with final-layer readout training

The next independent pilot runs `train_local --update sgd --readout-training final`.
Hidden updates use `h_next = h - softplus(raw_eta) * prediction`, without moments.
Initial eta remains 0.01 and is learnable; parameter AdamW remains lr=0.0005,
with the same warmup and 200M-token schedule. All layer losses retain coefficient
1/6. This fixed configuration is a reference before learning-rate tuning.

Intermediate losses detach the readout matrix and RMSNorm weight, while retaining
the readout derivative with respect to hidden. Each block and its eta therefore
still train on their own full-token CE. Only the final-layer CE trains the direct
readout branches. Embedding/head tying is retained, so the shared matrix ALSO
receives the first-layer CE gradient through the input embedding; it is not a
matrix trained exclusively by the final loss. Final RMSNorm has no such input
path and receives only the final-layer loss gradient.

`check_local_sgd` checks these routes, an independent SGD-gradient reference at
accumulation 1/3/8, dense/CCE gradients, unchanged forward values and block gradients
when readout parameters are detached, recovery, and validation side effects.
`benchmark_local --sgd-final` compares end-to-end hidden SGD with CCE, the previous
local hidden Adam with CCE, local SGD with all-layer readout gradients, and local
SGD with final-layer readout gradients, on one GPU with identical macrostep sizes.
These are cost measurements; only the last case receives a new training pilot.
The matching CHTC launcher is `chtc/experiments/pretrain-llama-local-sgd.sub`.

Validation additionally records prediction RMS, actual hidden-update RMS, and
update RMS divided by incoming-hidden RMS at each layer. Metrics average the
per-microbatch RMS values. These help distinguish small update magnitudes from
poor update directions without changing the optimization objective.

Future learning-rate exploration should start with a small order-of-magnitude
sweep, then narrow the range. Distinguish the hidden initial eta from parameter
AdamW learning rate, use plausible ranges for each, and preserve the seed, data,
token budget, and validation protocol across trials. Each trial has its own W&B
run; the user's examples 0.001/0.01/0.1 describe the spacing, not mandatory values
for every optimizer.

## Learning-rate sweep and matched training control

`train_local` exposes `--hidden-step-size` (learnable initial eta) and
`--parameter-lr` (peak LR for every AdamW parameter group, including eta).
Defaults remain 0.01 and 0.0005. Resume rejects changes to either setting or
the training objective. Actual scheduled parameter LR is logged every ten steps.

The first grid crosses eta 0.001/0.01/0.1 with parameter LR
0.00005/0.0005/0.005, including a repeat of the center point. All nine independent
single-GPU jobs are submitted together, as requested by the user. Each checks
the controls and connected-gradient implementation on GPU, then trains 50 steps
and resumes its own checkpoint to 1000 steps. Every run has the same seed, data,
200M schedule horizon, batch and validation protocol. Only finite-value failures
stop a candidate; a poor early CE alone does not prune it. W&B group is
`gradformer-local-sgd-lr-sweep-20261028`.

Submit from the cluster directory using `condor_submit pretrain-llama-local-lr.sub
trials=local-lr-grid.tsv`. The immutable code bundle is `llama2-local-lr-v1.tar.gz`.
Each output archive includes the cluster and process IDs. Budget approximately
0.4 GiB output per candidate; preserve checkpoints for follow-up comparisons.

`--training-objective e2e --readout-training final` enables a diagnostic control
using exactly the same hidden SGD forward, full-token CCE and independent
per-block clipping, with no hidden detaches and only final CE. Final CE retains
coefficient 1/depth in both objectives, matching the direct head gradient scale.
End-to-end runs do not label their training losses as per-layer local CE. The
same all-layer validation diagnostic evaluates both objectives. The matched end-to-end grid in `local-lr-e2e.tsv` uses all nine of the same
LR pairs, submitted concurrently with the local grid. Both objectives therefore
have the same initial tuning budget, allowing the effect of local training to
be compared without changing the hidden update or CE implementation. It is distinct from historical end-to-end runs with
native CE and global clipping. A finite single-seed, 1000-step sweep cannot prove
that all learning-rate or convergence effects have been excluded.
