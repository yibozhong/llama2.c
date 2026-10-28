# Paired hidden-SGD learning-rate sweep

All 26 single-GPU training runs completed 1000 steps / 32,768,000 tokens each,
with the same seed 1337, data order and schedule. Both objectives received the
same 13 candidates. Training source: 645b012e5d30ab3b032a48d1535399810a2553bc.
This experiment uses hidden SGD, no EMA and no gradient-prediction auxiliary.
The historical hidden-Adam NTP-only and gradient-supervised versions were NOT
LR-swept; this experiment does not settle their LR sensitivity.

The initial grid crossed eta 0.001/0.01/0.1 with parameter AdamW peak LR
0.00005/0.0005/0.005. Additional paired points were (eta, LR) =
(0.01, 0.015), (0.1, 0.015), (1, 0.0005), (10, 0.0005).
Eta is learnable; these are initial values. The queued eta=1 pair was changed
from LR=0.005 to 0.0005 before either ran; the final manifests match executed
configurations. No poor-but-finite candidate was pruned.

| Protocol | Best local CE | Best e2e CE | Gap |
| --- | ---: | ---: | ---: |
| Fixed validation sample used for selection | 2.020580 | 1.787003 | 0.233577 |
| Full validation, 1,999,872 targets, same A100 | 2.060319 | 1.828854 | 0.231464 |

Both winners use eta=1, LR=0.0005. Training runs:
[local g67o54rx](https://wandb.ai/yibozhong657-none/nanochat/runs/g67o54rx),
[e2e k1pg6pt6](https://wandb.ai/yibozhong657-none/nanochat/runs/k1pg6pt6).
The original local center repeated at 2.403103; tuning improves CE by 0.382524.
Eta=10 and parameter LR=0.015 did not improve the winners. A finite one-seed,
1000-step search does not exclude every schedule or establish asymptotic limits.
Exhaustive evaluation uses the same validation split, not an untouched test set.

All exit codes, source/configs, actual LR schedules, GPU checks, model/optimizer
finite audits, final token counts and W&B values were verified. The two selected
final checkpoints are archived externally with the full-validation artifact:
[evaluation jf7ec1sj](https://wandb.ai/yibozhong657-none/nanochat/runs/jf7ec1sj).
The [report oh7hxzmu](https://wandb.ai/yibozhong657-none/nanochat/runs/oh7hxzmu)
contains grid, convergence and per-layer PNG/PDF/SVG figures and input evidence.

The previous same-L40 full-step benchmark measured e2e SGD + CCE at 0.203812 s
and 0.705427 GiB peak allocated, versus strict local at 0.299760 s and 0.375947 GiB.
Local saves activation memory; ordinary e2e is faster here. Mixed GPU types in
the training sweep are not a matched speed comparison. The user's efficiency
priority is training time / total compute to the same validation CE.

Reproduction files are in `chtc/experiments/`: four trial manifests, the training
launcher, evaluation/report launchers, summary CSV, full-validation JSON and
metadata selection/verification/render scripts. Copy launchers/manifests to the
workspace `cluster/`, and the report tools plus `local-lr-extra-manifests.json` to `reports/`, as referenced by
the submit files. Use `monitor_local_lr.py 7333513:9 7333514:9 7333653:6 7333966:2` to extract metadata from the original archives (substitute IDs for reruns). Run verification after training output extraction, then select
models, then submit evaluation and report jobs. The training bundle is
`llama2-local-lr-v1.tar.gz` from 645b012; evaluation bundle is
`llama2-local-lr-eval-v1.tar.gz` from 28f09c8. Container/CCE/data remain pinned as
described in LOCAL_CE.md and the launchers. Credentials are injected separately.

The result supports a sample-efficiency cost of strict per-layer CE in this
setup, not failure of optimizer-based hidden refinement in general. Grouped
local losses and cheaper forward refinements remain proposals; no new-method
training was launched as part of this report.
