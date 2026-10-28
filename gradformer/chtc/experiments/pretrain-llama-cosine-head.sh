#!/bin/bash
set -euo pipefail
GRADFORMER_WORKDIR="$PWD"
GRADFORMER_ETA="$1"
GRADFORMER_LR="$2"
GRADFORMER_LABEL="$3"
GRADFORMER_MODE="$4"
GRADFORMER_LAST_ETA="$5"
GRADFORMER_CHECKS="$6"
mkdir -p results
trap 'tar -cf "$GRADFORMER_WORKDIR/results.tar" -C "$GRADFORMER_WORKDIR" results' EXIT
export NETRC="$PWD/.netrc" WANDB_DIR="$PWD/results"
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none OMP_NUM_THREADS=1
cp /opt/gradformer-environment.txt results/container-environment.txt
cp /opt/gradformer-locks.txt results/container-locks.txt
tar -xzf llama2-cosine-head-v1.tar.gz
tar -xf dataset-v1.tar
sha256sum llama2-cosine-head-v1.tar.gz dataset-v1.tar > results/input-hashes.txt
cd llama2.c
if [[ "$GRADFORMER_CHECKS" == "yes" ]]; then
    python -m gradformer.check_cosine_only --output ../results/legacy-cosine-only-check.json
    python -m gradformer.check_cosine_head --output ../results/cosine-head-check.json
else
    python -m gradformer.check_cosine_head --quick --output ../results/cosine-head-check.json
fi
GRADFORMER_ARGS=(--data ../dataset --output ../results/cosine-head
    --hidden-step-size "$GRADFORMER_ETA" --parameter-lr "$GRADFORMER_LR"
    --mode "$GRADFORMER_MODE" --last-eta "$GRADFORMER_LAST_ETA" --label "$GRADFORMER_LABEL")
python -m gradformer.train_cosine_head "${GRADFORMER_ARGS[@]}" --stop-after-step 50
python - <<'VERIFY'
import json, math
from pathlib import Path
p=Path('../results/cosine-head')
assert json.loads((p/'latest.json').read_text())['step']==50
rows=[json.loads(s) for s in (p/'metrics.jsonl').read_text().splitlines()]
assert rows[-1]['step']==50 and 'val_cosine/mean' in rows[-1]
assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
print('Passed cosine/readout 50-step gate',flush=True)
VERIFY
python -m gradformer.train_cosine_head "${GRADFORMER_ARGS[@]}" --stop-after-step 1000 --resume
python -m gradformer.audit_cosine_head --folder ../results/cosine-head --output ../results/cosine-head-summary.json
