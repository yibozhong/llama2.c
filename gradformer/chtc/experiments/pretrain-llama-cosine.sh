#!/bin/bash
set -euo pipefail
GRADFORMER_WORKDIR="$PWD"
mkdir -p results
trap 'tar -cf "$GRADFORMER_WORKDIR/results.tar" -C "$GRADFORMER_WORKDIR" results' EXIT
export NETRC="$PWD/.netrc" WANDB_DIR="$PWD/results"
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none
export OMP_NUM_THREADS=1
cp /opt/gradformer-environment.txt results/environment.txt
cp /opt/gradformer-locks.txt results/environment-locks.txt
tar -xzf llama2-cosine-v1.tar.gz
tar -xf dataset-v1.tar
cd llama2.c
python -m gradformer.check_cosine --output ../results/cosine-check.json
common=(--backend llama2 --update adam --data ../dataset
        --depth 6 --width 288 --heads 6 --sequence 256 --batch 16
        --batch-tokens 32768 --tokens 200000000 --seed 1337
        --eval-every 250 --eval-batches 32 --save-every 1000
        --backward reuse --alpha 0 --beta 1
        --cosine-probe-every 25 --cosine-probe-refresh-every 250 --cosine-eval-batches 8)
for variant in cosine-A-ntp-only cosine-B-fixed1 cosine-C-balanced1; do
    case "$variant" in
        cosine-A-ntp-only) extra=(--gradient-balance fixed --aux-weight 0);;
        cosine-B-fixed1) extra=(--gradient-balance fixed --aux-weight 1);;
        cosine-C-balanced1) extra=(--gradient-balance global --aux-weight 1 --balance-ratio 1);;
    esac
    python -m gradformer.train "${common[@]}" "${extra[@]}" --variant "$variant" --output "../results/$variant" --stop-after-step 50
    python - "$variant" <<'VERIFY'
import json, math, sys
from pathlib import Path
folder=Path('../results')/sys.argv[1]
assert json.loads((folder/'latest.json').read_text())['step']==50
rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
assert rows[-1]['step']==50 and math.isfinite(rows[-1]['val_ce'])
probes=[r for r in rows if 'probe/fixed_loss_delta' in r]
assert len(probes)==2
for r in rows:
    assert all(math.isfinite(v) for v in r.values() if isinstance(v,(int,float)))
    if sys.argv[1]=='cosine-C-balanced1' and 'balance/aux_to_ntp_ratio' in r:
        assert r['balance/active']==1 and abs(r['balance/aux_to_ntp_ratio']-1)<2e-5
print('Passed 50-step finite-loss/gradient-budget/diagnostic gates:',sys.argv[1],flush=True)
VERIFY
    python -m gradformer.train "${common[@]}" "${extra[@]}" --variant "$variant" --output "../results/$variant" --stop-after-step 1000 --resume
done
python - <<'SUMMARY'
import json, math, statistics
from pathlib import Path
summary={'pilot_steps':1000,'pilot_tokens':32768000,'full_schedule_tokens':200015872,'variants':{}}
for folder in sorted(Path('../results').glob('cosine-*-*')):
    if not folder.is_dir(): continue
    rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
    probes=[r for r in rows if 'probe/fixed_loss_delta' in r]
    vals=[r for r in rows if 'val_ce' in r]
    assert len(probes)==40 and vals[-1]['step']==1000
    assert json.loads((folder/'latest.json').read_text())['step']==1000
    assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
    if folder.name=='cosine-C-balanced1':
        assert all(r['balance/active']==1 and abs(r['balance/aux_to_ntp_ratio']-1)<2e-5 for r in rows if 'balance/aux_to_ntp_ratio' in r)
    row={'wandb_run':(folder/'wandb-id.txt').read_text().strip(), 'final_validation':vals[-1],
         'probes':len(probes),'decreased':sum(r['probe/decreased'] for r in probes),
         'flat':sum(r['probe/flat'] for r in probes),'increased':sum(r['probe/increased'] for r in probes),
         'mean_fixed_loss_delta':statistics.mean(r['probe/fixed_loss_delta'] for r in probes),
         'median_fixed_loss_delta':statistics.median(r['probe/fixed_loss_delta'] for r in probes),
         'negative_grad_dot_update':sum(r['probe/all/raw_cosine_grad_dot_update']<0 for r in probes),
         'max_noop_error':max(r['probe/noop_error'] for r in probes),
         'probe_seconds':sum(r['probe/seconds'] for r in probes),
         'median_nonprobe_step_seconds':statistics.median(r['step_seconds'] for r in rows if 'step_seconds' in r and 'probe/seconds' not in r),
         'last_probe':probes[-1]}
    summary['variants'][folder.name]=row
assert len(summary['variants'])==3
Path('../results/cosine-pilot-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
SUMMARY
