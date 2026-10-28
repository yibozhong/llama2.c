#!/bin/bash
set -euo pipefail
GRADFORMER_WORKDIR="$PWD"
mkdir -p results
trap 'tar -cf "$GRADFORMER_WORKDIR/results.tar" -C "$GRADFORMER_WORKDIR" results' EXIT
export NETRC="$PWD/.netrc" WANDB_DIR="$PWD/results"
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none OMP_NUM_THREADS=1
cp /opt/gradformer-environment.txt results/environment.txt
cp /opt/gradformer-locks.txt results/environment-locks.txt
tar -xzf llama2-cosine-only-v1.tar.gz
tar -xf dataset-v1.tar
cd llama2.c
python -m gradformer.check_cosine --output ../results/joint-regression-check.json
python -m gradformer.check_cosine_only --output ../results/cosine-only-check.json
common=(--backend llama2 --update adam --data ../dataset
        --variant cosine-D-no-ntp --output ../results/cosine-D-no-ntp
        --depth 6 --width 288 --heads 6 --sequence 256 --batch 16
        --batch-tokens 32768 --tokens 200000000 --seed 1337
        --eval-every 250 --eval-batches 32 --save-every 1000
        --parameter-objective cosine-only --backward reference
        --gradient-balance fixed --aux-weight 1 --alpha 0 --beta 1
        --cosine-probe-every 25 --cosine-probe-refresh-every 250 --cosine-eval-batches 8)
python -m gradformer.train "${common[@]}" --stop-after-step 50
python - <<'VERIFY'
import json,math
from pathlib import Path
folder=Path('../results/cosine-D-no-ntp')
assert json.loads((folder/'latest.json').read_text())['step']==50
rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
assert rows[-1]['step']==50 and math.isfinite(rows[-1]['val_ce'])
assert len([r for r in rows if 'probe/fixed_loss_delta' in r])==2
for r in rows:
    assert all(math.isfinite(v) for v in r.values() if isinstance(v,(int,float)))
    if 'train_ce' in r:
        assert r['objective/ntp_weight']==0 and r['objective/cosine_weight']==1
        assert r['balance/effective_mse_weight']==0
print('Passed strict cosine-only first 50 steps.',flush=True)
VERIFY
python -m gradformer.train "${common[@]}" --stop-after-step 1000 --resume
python - <<'SUMMARY'
import json,math,statistics
from pathlib import Path
import torch
import wandb
from gradformer.runtime import construct
folder=Path('../results/cosine-D-no-ntp')
rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
probes=[r for r in rows if 'probe/fixed_loss_delta' in r]
vals=[r for r in rows if 'val_ce' in r]
latest=json.loads((folder/'latest.json').read_text())
assert latest['step']==1000 and vals[-1]['step']==1000 and len(probes)==40
assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
cfg=json.loads((folder/'config.json').read_text())
assert cfg['parameter_objective']=='cosine-only' and cfg['ntp_parameter_update_weight']==0
assert cfg['alpha']==0 and cfg['beta']==1 and cfg['aux_weight']==1
assert all(r['objective/ntp_weight']==0 and r['objective/cosine_weight']==1 for r in rows if 'train_ce' in r)
# Read-only checkpoint audit: no parameter update is performed here.
torch.manual_seed(1337)
model=construct('llama2','adam',depth=6,width=288,heads=6,vocab=cfg['dataset']['vocab_size'],sequence=256,step_size=.01,device='cpu')
initial_norm=model.norm.weight.detach().clone()
initial_eta=model.hidden_step_raw[-1].detach().clone()
initial_embedding=model.tok_embeddings.weight.detach().clone()
checkpoint=folder/latest['directory']
model.load_state_dict(torch.load(checkpoint/'model.pt',map_location='cpu',weights_only=False))
optimizer=model.configure_optimizers(.1,5e-4,(.9,.95),'cpu')
state=torch.load(checkpoint/'rank-0.pt',map_location='cpu',weights_only=False)
optimizer.load_state_dict(state['optimizer'])
assert state['step']==1000
assert torch.equal(model.norm.weight,initial_norm)
assert torch.equal(model.hidden_step_raw[-1],initial_eta)
assert not torch.equal(model.tok_embeddings.weight,initial_embedding)
assert model.norm.weight not in optimizer.state
moments=optimizer.state[model.hidden_step_raw]
assert moments['exp_avg'][-1].item()==0 and moments['exp_avg_sq'][-1].item()==0
summary={'pilot_steps':1000,'pilot_tokens':32768000,'full_schedule_tokens':200015872,
         'wandb_run':(folder/'wandb-id.txt').read_text().strip(),'final_validation':vals[-1],
         'probes':len(probes),'decreased':sum(r['probe/decreased'] for r in probes),
         'flat':sum(r['probe/flat'] for r in probes),'increased':sum(r['probe/increased'] for r in probes),
         'mean_fixed_loss_delta':statistics.mean(r['probe/fixed_loss_delta'] for r in probes),
         'median_fixed_loss_delta':statistics.median(r['probe/fixed_loss_delta'] for r in probes),
         'negative_grad_dot_update':sum(r['probe/all/raw_cosine_grad_dot_update']<0 for r in probes),
         'max_noop_error':max(r['probe/noop_error'] for r in probes),
         'median_nonprobe_step_seconds':statistics.median(r['step_seconds'] for r in rows if 'step_seconds' in r and 'probe/seconds' not in r),
         'last_probe':probes[-1],
         'checkpoint_audit':{'passed':True,'final_norm_unchanged':True,'final_norm_has_optimizer_state':False,
                             'last_hidden_step_unchanged':True,'last_hidden_step_moments_zero':True,
                             'tied_embedding_changed':True}}
Path('../results/cosine-only-pilot-summary.json').write_text(json.dumps(summary,indent=2))
run=wandb.Api().run('yibozhong657-none/nanochat/'+summary['wandb_run'])
run.summary['cosine_only_checkpoint_audit']=summary['checkpoint_audit']
run.summary.update()
print(json.dumps(summary),flush=True)
SUMMARY
