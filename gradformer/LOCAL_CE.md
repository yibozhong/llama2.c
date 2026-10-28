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
