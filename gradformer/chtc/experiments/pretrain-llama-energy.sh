#!/bin/bash
set -euo pipefail
GRADFORMER_WORKDIR="$PWD"
GRADFORMER_KIND="$1"
GRADFORMER_ETA="$2"
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
tar -xzf llama2-energy-v1.tar.gz
tar -xzf cce-3de376c.tar.gz
tar -xf dataset-v1.tar
export PYTHONPATH="$PWD/cce-source${PYTHONPATH:+:$PYTHONPATH}"
sha256sum cce-3de376c.tar.gz llama2-energy-v1.tar.gz > results/input-hashes.txt
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
if [[ "$GRADFORMER_CHECKS" == "full" ]]; then
    python -m gradformer.check_energy --output ../results/energy-check.json
fi
GRADFORMER_ARGS=(--data ../dataset --output ../results/energy --kind "$GRADFORMER_KIND"
    --parameter-lr "$GRADFORMER_LR" --hidden-step-size "$GRADFORMER_ETA" --label "$GRADFORMER_LABEL")
python -m gradformer.train_energy "${GRADFORMER_ARGS[@]}" --stop-after-step 50
python - <<'VERIFY'
import json,math
from pathlib import Path
p=Path('../results/energy')
assert json.loads((p/'latest.json').read_text())['step']==50
s=json.loads((p/'summary.json').read_text());assert s['checkpoint_exact']
rows=[json.loads(line) for line in (p/'metrics.jsonl').read_text().splitlines()]
assert rows[-1]['step']==50 and 'val_ce' in rows[-1]
assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v,(int,float)))
print('Passed energy 50-step gate',flush=True)
VERIFY
python -m gradformer.train_energy "${GRADFORMER_ARGS[@]}" --stop-after-step 1000 --resume
if [[ "$GRADFORMER_CHECKS" == "full" ]]; then
    # Real standard residual Transformer, trained on the SAME physical GPU.
    python -m gradformer.train_energy --data ../dataset --output ../results/baseline \
        --kind baseline --parameter-lr .0005 --hidden-step-size .01 --label same-gpu-baseline --stop-after-step 1000
fi
