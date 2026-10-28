# Cosine blocks with an NTP-trained readout

The historical cosine-only model reached validation CE 9.081894 at 1000 steps,
with cosine 0.583240. It had no direct NTP parameter updates: final norm and last
hidden step stayed at initialization. This experiment restores direct CE to
the readout while keeping block parameters free of NTP gradients.

The forward remains hidden Adam (depth moments beta1=.9, beta2=.999, epsilon=.01).
Each layer predicts a direction; targets remain the online final mean-token CE
gradient at each layer's input, including the current block, and are detached.
There is no EMA or MSE. Cosine coefficient is 1 and its graph remains connected
across depth: this is not strictly isolated layer-local learning.

Parameter routing:

- Block parameters and early hidden step sizes receive cosine gradients only.
- Final norm receives direct final CE, coefficient 1.
- The tied head/embedding receives both direct readout CE and cosine through
  the embedding input path. It receives no NTP through that input path, but
  head updates naturally change the tied input embedding on subsequent steps.
- The main variant also gives only the final hidden step scalar its CE
  derivative. It has no cosine signal. A center-point ablation keeps it fixed,
  isolating readout training from this scalar calibration.

The implementation requests CE VJPs at readout alias nodes and the last scalar,
alongside the real hidden targets. This is equivalent to detaching hidden for
the head loss and separately differentiating the last scalar with all its
incoming hidden/prediction/moments held fixed. It reuses the original CE graph,
so no second vocabulary projection or second-order derivatives are required.
An independent GPU reference instead uses a literal detached-hidden second
head loss and native full backpropagation to check these VJPs.

Clip norm 1 separately for blocks plus early step scalars, head plus final norm,
and the final step scalar. Thus a head or final-step gradient cannot change the
clipping factor on the block cosine gradients. This differs from the historical
global clipping, so fresh pure-cosine and NTP-only controls use the same new
grouping. Parameters remain AdamW beta=.9/.95, matrix decay=.1, scalar decay=0.

The five-point sweep uses parameter peak LR=.00005/.0005/.005 at initial hidden
eta=.01, plus eta=.001/.1 at LR=.0005. LR applies to all parameter groups,
including the readout; this is not a separate head-LR sweep or full Cartesian
grid. Three additional center controls are fixed-final-eta, pure cosine and
NTP-only. Eight independent single-GPU jobs in total. All start from seed1337,
six blocks, width288, six heads, vocab32000, same TinyStories cache/order,
microbatch16x256, accumulation8, and 1000 steps / 32,768,000 tokens. The original
100-step warmup / 6104-step schedule horizon is retained. Native full-vocabulary
CE and fixed 32-batch CE / 8-batch cosine validation match historical diagnostics.
Different models still have different gradient targets; cosine is a
self-consistency diagnostic, not a score on one common teacher target set.

Use the existing uv-locked llama2-uv-v4.sif and immutable code bundle
llama2-cosine-head-v1.tar.gz. Submit the center manifest first; after its GPU
checks and 50-step recovery gate pass, submit the remaining manifest together.
No model computation runs on the access point. Every job records to
yibozhong657-none/nanochat, group gradformer-cosine-head-lr-20261028.
Each 50-step checkpoint is resumed to 1000; final finite-state, norm/eta routing
and actual LR-schedule audits are recorded in the archive and W&B.
