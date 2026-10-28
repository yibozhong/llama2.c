#!/bin/bash
set -euo pipefail
variant="${1:?variant required}"
case "$variant" in
    cosine-B-fixed1) extra=(--gradient-balance fixed --aux-weight 1);;
    cosine-C-balanced1) extra=(--gradient-balance global --aux-weight 1 --balance-ratio 1);;
    *) exit 2;;
esac
GRADFORMER_WORKDIR="$PWD"
mkdir -p results
trap 'tar -cf "$GRADFORMER_WORKDIR/results.tar" -C "$GRADFORMER_WORKDIR" results' EXIT
export NETRC="$PWD/.netrc" WANDB_DIR="$PWD/results"
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none OMP_NUM_THREADS=1
cp /opt/gradformer-environment.txt results/environment.txt
cp /opt/gradformer-locks.txt results/environment-locks.txt
cp "$0" results/continuation-launcher.sh
tar -xzf llama2-cosine-v1.tar.gz
tar -xf dataset-v1.tar
tar -xf "$variant-step1000.tar"
cd llama2.c
python - "$variant" <<'VERIFY'
import json,sys
from pathlib import Path
out=Path('../results')/sys.argv[1]
assert json.loads((out/'latest.json').read_text())['step']==1000
cfg=json.loads((out/'config.json').read_text())
source=json.loads(Path('gradformer/source.json').read_text())
assert cfg['source']==source
assert source['git_commit']=='2e52c7375352dcb73382d9634f17ba1209a44d0e'
assert cfg['tokens']==200000000 and cfg['alpha']==0 and cfg['beta']==1
assert cfg['batch_tokens']==32768 and cfg['seed']==1337
VERIFY
python -m gradformer.train --backend llama2 --update adam --variant "$variant" \
    --data ../dataset --output "../results/$variant" --resume \
    --depth 6 --width 288 --heads 6 --sequence 256 --batch 16 \
    --batch-tokens 32768 --tokens 200000000 --seed 1337 \
    --eval-every 250 --eval-batches 32 --save-every 1000 \
    --backward reuse --alpha 0 --beta 1 "${extra[@]}" \
    --cosine-probe-every 25 --cosine-probe-refresh-every 250 --cosine-eval-batches 8
python - "$variant" <<'VERIFY'
import json,math,sys
from pathlib import Path
out=Path('../results')/sys.argv[1]
done=json.loads((out/'completed.json').read_text())
assert done['tokens']==200015872 and math.isfinite(done['final_val_ce'])
assert json.loads((out/'latest.json').read_text())['step']==6104
rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
assert len([r for r in rows if 'probe/fixed_loss_delta' in r])==244
if sys.argv[1]=='cosine-C-balanced1':
    measured=[r for r in rows if 'balance/aux_to_ntp_ratio' in r]
    assert all(r['balance/active']==1 and abs(r['balance/aux_to_ntp_ratio']-1)<2e-5 for r in measured)
print('Completed full cosine-only pretraining:',sys.argv[1],json.dumps(done),flush=True)
VERIFY
