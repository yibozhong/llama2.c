#!/bin/bash
set -euo pipefail
GRADFORMER_WORKDIR="$PWD"
mkdir -p results
trap 'tar -cf "$GRADFORMER_WORKDIR/results.tar" -C "$GRADFORMER_WORKDIR" results' EXIT
export NETRC="$PWD/.netrc" WANDB_DIR="$PWD/results"
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none OMP_NUM_THREADS=1
cp /opt/gradformer-environment.txt results/environment.txt
cp /opt/gradformer-locks.txt results/environment-locks.txt
tar -xzf llama2-ema-v1.tar.gz
tar -xf dataset-v1.tar
cd llama2.c
python -m gradformer.check_cosine --output ../results/joint-regression-check.json
python -m gradformer.check_cosine_only --output ../results/pure-regression-check.json
python -m gradformer.check_ema --output ../results/ema-check.json
python -m gradformer.benchmark_ema --output ../results/ema-benchmark.json
common=(--backend llama2 --update adam --data ../dataset
        --depth 6 --width 288 --heads 6 --sequence 256 --batch 16
        --batch-tokens 32768 --tokens 200000000 --seed 1337
        --eval-every 250 --eval-batches 32 --save-every 1000
        --gradient-balance fixed --aux-weight 1 --alpha 0 --beta 1
        --target-source ema --ema-decay .99
        --cosine-probe-every 25 --cosine-probe-refresh-every 250 --cosine-eval-batches 8)
for variant in ema-joint-fixed1 ema-cosine-only; do
    case "$variant" in
        ema-joint-fixed1) extra=(--parameter-objective joint --backward reuse);;
        ema-cosine-only) extra=(--parameter-objective cosine-only --backward reference);;
    esac
    python -m gradformer.train "${common[@]}" "${extra[@]}" --variant "$variant" --output "../results/$variant" --stop-after-step 50
    python - "$variant" <<'VERIFY'
import json,math,sys
from pathlib import Path
import torch
folder=Path('../results')/sys.argv[1]
latest=json.loads((folder/'latest.json').read_text())
assert latest['step']==50
teacher=torch.load(folder/latest['directory']/'teacher.pt',map_location='cpu',weights_only=False)
assert teacher['updates']==50 and teacher['decay']==.99
rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
assert rows[-1]['step']==50 and math.isfinite(rows[-1]['val_ce'])
assert 'val_cosine/mean' in rows[-1] and 'val_ema_cosine/mean' in rows[-1]
assert len([r for r in rows if 'probe/fixed_loss_delta' in r])==2
for r in rows:
    assert all(math.isfinite(v) for v in r.values() if isinstance(v,(int,float)))
    if 'train_ce' in r:
        assert r['ema/updates']==r['step'] and r['ema/decay']==.99
print('Passed EMA first 50 steps:',sys.argv[1],flush=True)
VERIFY
    python -m gradformer.train "${common[@]}" "${extra[@]}" --variant "$variant" --output "../results/$variant" --stop-after-step 1000 --resume
done
python - <<'SUMMARY'
import json,math,statistics
from pathlib import Path
import torch,wandb
summary={'pilot_steps':1000,'pilot_tokens':32768000,'ema_decay':.99,'variants':{}}
for name in ('ema-joint-fixed1','ema-cosine-only'):
    folder=Path('../results')/name
    rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
    vals=[r for r in rows if 'val_ce' in r]
    probes=[r for r in rows if 'probe/fixed_loss_delta' in r]
    latest=json.loads((folder/'latest.json').read_text())
    assert latest['step']==1000 and vals[-1]['step']==1000 and len(probes)==40
    assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
    teacher=torch.load(folder/latest['directory']/'teacher.pt',map_location='cpu',weights_only=False)
    assert teacher['updates']==1000 and teacher['decay']==.99
    assert all(torch.isfinite(v).all().item() for v in teacher['model'].values() if torch.is_floating_point(v))
    assert all(r['ema/updates']==r['step'] for r in rows if 'train_ce' in r)
    row={'wandb_run':(folder/'wandb-id.txt').read_text().strip(),'final_validation':vals[-1],
         'probes':len(probes),'decreased':sum(r['probe/decreased'] for r in probes),
         'flat':sum(r['probe/flat'] for r in probes),'increased':sum(r['probe/increased'] for r in probes),
         'mean_fixed_loss_delta':statistics.mean(r['probe/fixed_loss_delta'] for r in probes),
         'negative_grad_dot_update':sum(r['probe/all/raw_cosine_grad_dot_update']<0 for r in probes),
         'max_noop_error':max(r['probe/noop_error'] for r in probes),
         'median_nonprobe_step_seconds':statistics.median(r['step_seconds'] for r in rows if 'step_seconds' in r and 'probe/seconds' not in r),
         'last_probe':probes[-1],'teacher_checkpoint_audit':{'passed':True,'updates':1000,'decay':.99,'finite':True}}
    summary['variants'][name]=row
    run=wandb.Api().run('yibozhong657-none/nanochat/'+row['wandb_run'])
    run.summary['teacher_checkpoint_audit']=row['teacher_checkpoint_audit'];run.summary.update()
Path('../results/ema-pilot-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
SUMMARY
