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

## Completed results (2026-10-28)

All eight jobs, 7334936.0 and 7334940.0–6, completed successfully with training
source `b89b225dfaf2770b0f42470b6ccd453dec0f4790`. Configuration, scheduled LR,
finite final model/optimizer state, intended norm/eta routing, and final W&B
metrics were checked. These are single-seed, fixed-sample validation results
after 1000 steps, not exhaustive validation or a complete LR grid.

| Objective / variant | Initial hidden eta | Parameter peak LR | Validation CE | Validation cosine | Run |
| --- | ---: | ---: | ---: | ---: | --- |
| Cosine + head, center | .01 | .0005 | 2.678559 | .140343 | h749wjta |
| Cosine + head, low LR | .01 | .00005 | 5.054484 | .508242 | yn4o32w9 |
| Cosine + head, high LR | .01 | .005 | 3.164355 | .108421 | awly8vqh |
| Cosine + head, low eta | .001 | .0005 | 5.773774 | .633116 | ago796lo |
| Cosine + head, high eta | .1 | .0005 | 2.693906 | .098573 | mja5q7fs |
| Cosine + head, fixed final eta | .01 | .0005 | 2.808591 | .180079 | endg0y5x |
| Pure cosine, no head NTP | .01 | .0005 | 9.081894 | .583240 | s85axjjl |
| End-to-end NTP-only | .01 | .0005 | 1.932952 | .006001 | gwl8l6pl |

Training the readout resolves much of the historical CE~9 failure, even with
the last eta fixed. The best cosine+head configuration still trails NTP-only
by about .746 CE. High target alignment does not imply good prediction CE;
each model generates its own targets. The pure-cosine control exactly
reproduces its historical final CE and cosine.

The high-LR run stayed finite but had severe direction-scale growth: final
training layer-0 prediction norm ~389415 versus target norm ~.00537, with mean
raw MSE across layers ~9.25e7. MSE has zero training weight. Cosine alone leaves
scale uncontrolled; finite checkpoints do not establish healthy optimization.

Blackwell median training-step time was .185764 seconds for the center versus
.114785 seconds for NTP-only (1.62x). These were different allocations of the
same GPU model, not an isolated simultaneous benchmark; the pure-cosine run
used H200. Times exclude queueing, setup, evaluation and archival.

Independent GPU reference checks covered four routing modes, FP32/BF16,
accumulation 1/8, separate clipping and exact checkpoint recovery. Maximum
FP32 parameter-gradient absolute error was 8.20e-8; BF16 matched exactly.
There were no block-gradient hook events during the NTP target/readout VJP
for cosine modes. Every training job also passed its own launch/recovery gates.

[W&B experiment group](https://wandb.ai/yibozhong657-none/nanochat/groups/gradformer-cosine-head-lr-20261028)
contains all training and check runs.
[Report run 00t5trrt](https://wandb.ai/yibozhong657-none/nanochat/runs/00t5trrt)
was generated by CPU job 7335449, exited 0, and was verified finished remotely.
Its `gradformer-cosine-head-lr-report-and-selected-models` artifact contains
the curves (PNG/PDF/SVG), complete records, rendering script, and final model
weights/configs for the best cosine+head, pure-cosine and NTP-only runs.
Plots were visually inspected. Metadata collectors, verifiers and archival
scripts are preserved in `gradformer/chtc/experiments/`; their AP paths reflect
the documented CHTC workspace layout.
