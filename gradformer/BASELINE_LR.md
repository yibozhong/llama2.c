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

Submission and results pending at this commit.
