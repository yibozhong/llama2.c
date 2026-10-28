#!/bin/bash
set -euo pipefail
GRADFORMER_WORKDIR="$PWD"
mkdir -p results
trap 'tar -cf "$GRADFORMER_WORKDIR/results.tar" -C "$GRADFORMER_WORKDIR" results' EXIT
export NETRC="$PWD/.netrc" WANDB_DIR="$PWD/results"
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none OMP_NUM_THREADS=1
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TRITON_CACHE_DIR="$PWD/triton-cache" TORCHINDUCTOR_CACHE_DIR="$PWD/inductor-cache"
cp /opt/gradformer-environment.txt results/container-environment.txt
cp /opt/gradformer-locks.txt results/container-locks.txt
tar -xzf llama2-local-v1.tar.gz
tar -xzf cce-3de376c.tar.gz
tar -xf dataset-v1.tar
export PYTHONPATH="$PWD/cce-source${PYTHONPATH:+:$PYTHONPATH}"
sha256sum cce-3de376c.tar.gz llama2-local-v1.tar.gz > results/input-hashes.txt
python - <<'ENVIRONMENT'
import json,sys,torch,triton,cut_cross_entropy
from pathlib import Path
assert torch.__version__.startswith('2.9.1')
assert triton.__version__=='3.5.1'
assert cut_cross_entropy.__version__=='25.9.3'
record=dict(python=sys.version,torch=torch.__version__,triton=triton.__version__,
            cce=cut_cross_entropy.__version__,cce_commit='3de376c106a1916bc5e1b619f9c77c87a461ee1c',
            cce_module=cut_cross_entropy.__file__,gpu=torch.cuda.get_device_name(),
            container_sha256='245f31bfbf83473593e127a56b7040b80bb9dbaf48e5387bc71aecd6bd0b7baf')
Path('results/runtime.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
ENVIRONMENT
cd llama2.c
python -m gradformer.check_local --output ../results/local-check.json
python -m gradformer.benchmark_local --output ../results/local-benchmark.json
python -m gradformer.train_local --data ../dataset --output ../results/local-ce --stop-after-step 50
python - <<'VERIFY'
import json,math
from pathlib import Path
p=Path('../results/local-ce')
assert json.loads((p/'latest.json').read_text())['step']==50
rows=[json.loads(s) for s in (p/'metrics.jsonl').read_text().splitlines()]
assert rows[-1]['step']==50 and 'val_local_ce/layer_5' in rows[-1]
assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
print('Passed strict-local all-token 50-step gate',flush=True)
VERIFY
python -m gradformer.train_local --data ../dataset --output ../results/local-ce --stop-after-step 1000 --resume
python - <<'SUMMARY'
import json,math,statistics
from pathlib import Path
import torch,wandb
p=Path('../results/local-ce')
latest=json.loads((p/'latest.json').read_text());assert latest['step']==1000
rows=[json.loads(s) for s in (p/'metrics.jsonl').read_text().splitlines()]
assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
model=torch.load(p/latest['directory']/'model.pt',map_location='cpu',weights_only=False)
assert all(torch.isfinite(v).all().item() for v in model.values() if torch.is_floating_point(v))
state=torch.load(p/latest['directory']/'rank-0.pt',map_location='cpu',weights_only=False)
assert state['step']==1000
audit=dict(passed=True,step=1000,finite_model=True,optimizer_saved=bool(state['optimizer']['state']))
values=[r for r in rows if 'val_ce' in r]
result=dict(run_id=(p/'wandb-id.txt').read_text().strip(),pilot_steps=1000,pilot_tokens=32768000,
            final_validation=values[-1],checkpoint_audit=audit,
            median_step_seconds=statistics.median(r['step_seconds'] for r in rows if 'step_seconds' in r))
Path('../results/local-pilot-summary.json').write_text(json.dumps(result,indent=2))
run=wandb.Api().run('yibozhong657-none/nanochat/'+result['run_id'])
run.summary['local_checkpoint_audit']=audit;run.summary.update()
print(json.dumps(result),flush=True)
SUMMARY
