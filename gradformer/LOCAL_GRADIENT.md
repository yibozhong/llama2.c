# Local CE with input-gradient constraints

The user explicitly selected auxiliary supervision of each layer output by a
real local CE gradient, not an architecture constrained to be an energy gradient.
Each block predicts a direction and uses the existing hidden SGD update. The
target is the gradient of that block's POST-update shared-head CE with respect
to its INPUT hidden, including the current block's Jacobian. It is detached.
This differs from the old final-NTP gradient target and from a direct-head CE
gradient computed before the block.

The objective is the layer mean of CE + alpha*raw_MSE + beta*(1-cosine). All
tokens and the full vocabulary remain supervised using pinned CCE. A CE
backward pass supplies both parameter gradients and the hidden target; an
auxiliary backward pass updates the current block and (through its input)
the first-layer embedding only. No second-order derivatives are computed.
Targets use the unweighted per-microbatch mean-token CE, before layer or
accumulation scaling. Raw MSE therefore retains the original batch-dependent
units; cosine is scale insensitive. Targets are not normalized or EMA-smoothed.

Hidden states are detached between blocks. Each block receives its local CE
plus auxiliary. Its eta receives local CE only: the prediction precedes the
eta update and targets are detached. Downstream losses never train that block.
Head/norm readout branches receive only final-layer CE. Tied embedding/head
also receives the first block's input-path CE and auxiliary gradients. Targets
are detached, so auxiliary losses do not train readout parameters through target
construction. Parameters update once after eight microbatches, using AdamW.
Clipping remains separate norm-1 groups per block+eta and for head/norm.

Six independent 1000-step / 32,768,000-token runs, seed1337, six blocks, width288,
six heads, vocabulary32000, batch16x256, accumulation8, initial hidden eta=1.
This eta and the central parameter LR=.0005 came from the earlier local SGD
sweep. Keep its original 100-step warmup and 6104-step LR horizon. The new
readout/validation definition is unchanged from the local SGD experiment.

| Variant | alpha (MSE) | beta (cosine) | Parameter peak LR |
| --- | ---: | ---: | ---: |
| Local CE control | 0 | 0 | .0005 |
| Weak direction constraint | 0 | .1 | .0005 |
| Center | 0 | 1 | .0005 |
| Low parameter LR | 0 | 1 | .00005 |
| High parameter LR | 0 | 1 | .005 |
| Scale and direction constraint | 1 | 1 | .0005 |

Use the immutable llama2-local-gradient-v2 bundle and uv-locked llama2-uv-v4
container with CCE source 3de376c. Submit the center first, then submit the
remaining five together after the full GPU correctness and 50-step gates pass.
Every job has a GPU check and exact-configuration 50-to-1000-step resume.
All runs go to yibozhong657-none/nanochat, group gradformer-local-gradient-20261028.

Fixed validation uses seed987654, 32 microbatches for CE and eight for local
gradient diagnostics. Record per-layer CE improvement, raw MSE, cosine, prediction
and target norms, step size, time, and peak memory. On the first microbatch of
logged steps, capture the weighted block parameter gradients from the CE and
auxiliary backward passes separately, reporting their norms, ratio and angle.
These are microbatch block-matrix diagnostics, not complete accumulated-gradient
ratios and not optimizer-preconditioned update ratios. Timing includes their
overhead. Each trained model supplies its own gradient targets.

The independent GPU reference explicitly computes the local target VJP, then
differentiates the combined objective. It checks FP32/BF16, accumulation1/8,
ignored-token masking, no cross-block gradients, exact equivalence to the old
local baseline at zero auxiliary, CCE versus dense derivatives, diagnostic
side effects, checkpoint recovery, and a full 16x256x288/32k-vocabulary batch.
Final checkpoint audits verify finite model/optimizer, actual LR schedule,
trained final norm and positive hidden step sizes. No training runs on the AP.
