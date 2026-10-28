#!/bin/bash
set -euo pipefail
GRADFORMER_WORKDIR="$PWD"
GRADFORMER_ALPHA="$1"
GRADFORMER_BETA="$2"
GRADFORMER_LR="$3"
GRADFORMER_LABEL="$4"
GRADFORMER_CHECKS="$5"
mkdir -p results
trap 'tar -cf "$GRADFORMER_WORKDIR/results.tar" -C "$GRADFORMER_WORKDIR" results' EXIT
export NETRC="$PWD/.netrc" WANDB_DIR="$PWD/results"
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none OMP_NUM_THREADS=1
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TRITON_CACHE_DIR="$PWD/triton-cache" TORCHINDUCTOR_CACHE_DIR="$PWD/inductor-cache"
cp /opt/gradformer-environment.txt results/container-environment.txt
cp /opt/gradformer-locks.txt results/container-locks.txt
tar -xzf llama2-local-gradient-v1.tar.gz
tar -xzf cce-3de376c.tar.gz
tar -xf dataset-v1.tar
export PYTHONPATH="$PWD/cce-source${PYTHONPATH:+:$PYTHONPATH}"
sha256sum cce-3de376c.tar.gz llama2-local-gradient-v1.tar.gz > results/input-hashes.txt
python - <<'ENVIRONMENT'
import json,sys,torch,triton,cut_cross_entropy
from pathlib import Path
assert torch.__version__.startswith('2.9.1') and triton.__version__=='3.5.1'
assert cut_cross_entropy.__version__=='25.9.3'
record=dict(python=sys.version,torch=torch.__version__,triton=triton.__version__,cce=cut_cross_entropy.__version__,
            gpu=torch.cuda.get_device_name(),container_sha256='245f31bfbf83473593e127a56b7040b80bb9dbaf48e5387bc71aecd6bd0b7baf')
Path('results/runtime.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
ENVIRONMENT
cd llama2.c
GRADFORMER_CHECK_ARGS=()
if [[ "$GRADFORMER_CHECKS" != "full" ]]; then GRADFORMER_CHECK_ARGS+=(--quick); fi
python -m gradformer.check_local_gradient --output ../results/local-gradient-check.json "${GRADFORMER_CHECK_ARGS[@]}"
GRADFORMER_ARGS=(--data ../dataset --output ../results/local-gradient --alpha "$GRADFORMER_ALPHA"
    --beta "$GRADFORMER_BETA" --parameter-lr "$GRADFORMER_LR" --hidden-step-size 1 --label "$GRADFORMER_LABEL")
python -m gradformer.train_local_gradient "${GRADFORMER_ARGS[@]}" --stop-after-step 50
python - <<'VERIFY'
import json,math
from pathlib import Path
p=Path('../results/local-gradient')
assert json.loads((p/'latest.json').read_text())['step']==50
rows=[json.loads(line) for line in (p/'metrics.jsonl').read_text().splitlines()]
assert rows[-1]['step']==50 and 'val_gradient/cosine' in rows[-1]
assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v,(int,float)))
print('Passed local-gradient 50-step gate',flush=True)
VERIFY
python -m gradformer.train_local_gradient "${GRADFORMER_ARGS[@]}" --stop-after-step 1000 --resume
python -m gradformer.audit_local_gradient --directory ../results/local-gradient --output ../results/local-gradient-summary.json
