"""Read-only audit of completed standard-Transformer LR experiment metadata."""
import base64
import json
import math
import netrc
from pathlib import Path
import urllib.request

ROOT=Path('/home/yzhong239/gradformer')


def main():
    manifest=json.loads((ROOT/'reports/baseline-lr-submissions.json').read_text())
    login,_,password=netrc.netrc().authenticators('api.wandb.ai')
    filters=json.dumps({'group':manifest['group']})
    query='query { project(name: "nanochat", entityName: "yibozhong657-none") { runs(first: 100, filters: '+json.dumps(filters)+') { edges { node { name state summaryMetrics config } } } } }'
    request=urllib.request.Request('https://api.wandb.ai/graphql',data=json.dumps({'query':query}).encode(),
        headers={'Content-Type':'application/json','Authorization':'Basic '+base64.b64encode((login+':'+password).encode()).decode()})
    with urllib.request.urlopen(request,timeout=25) as response:data=json.load(response)
    if 'errors' in data:raise RuntimeError(data['errors'])
    remote={}
    for edge in data['data']['project']['runs']['edges']:
        row=edge['node'];row['summary']=json.loads(row.pop('summaryMetrics'));row['config']=json.loads(row['config']);remote[row['name']]=row
    records=[];common=None
    for trial in manifest['trials']:
        job=trial['job'];folder=ROOT/f'results/pretrain-llama-energy-{job}-metadata'/trial['model_directory']
        c=json.loads((folder/'config.json').read_text());s=json.loads((folder/'summary.json').read_text())
        rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        assert c['kind']==c['model']['hidden_update']=='baseline'
        assert c['parameters']==15191712 and c['parameter_lr']==trial['lr']
        assert c['source']==dict(git_commit=manifest['training_source'],dirty=False)
        assert c['seed']==manifest['seed'] and s['last_step']==1000 and s['checkpoint_exact'] and s['pilot_complete']
        fields={k:c[k] for k in ('model','dataset','batch_tokens','batch','sequence','accumulation','schedule_steps','warmup_steps','objective','clipping','ce_implementation','compile')}
        if common is None:common=fields
        else:assert common==fields
        assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v,(int,float)))
        assert not any(k.startswith('hidden_eta/') for row in rows for k in row)
        for row in rows:
            assert row['tokens']==row['step']*32768
            if 'parameter_lr' in row:
                i=row['step']-1;mult=(i+1)/100 if i<100 else .5*(1+math.cos(math.pi*(i-100)/6004))
                assert math.isclose(row['parameter_lr'],trial['lr']*mult,rel_tol=1e-12)
        vals=[row for row in rows if 'val_ce' in row]
        assert {250,500,750,1000}.issubset({row['step'] for row in vals})
        error=None
        if not trial['reused']:
            fifty=[row['val_ce'] for row in vals if row['step']==50];assert len(fifty)==2
            error=abs(fifty[0]-fifty[1]);assert error<=1e-6
        r=remote[s['run_id']]
        assert r['state']=='finished' and r['summary']['last_step']==1000
        assert abs(r['summary']['val_ce']-vals[-1]['val_ce'])<1e-7
        assert abs(r['summary']['best_val_ce']-s['best_val_ce'])<1e-7
        assert r['config']['parameter_lr']['value']==trial['lr'] and r['config']['kind']['value']=='baseline'
        log=(ROOT/f'logs/pretrain-llama-energy.{job}.log').read_text()
        assert 'Normal termination (return value 0)' in log
        records.append(dict(**trial,run_id=s['run_id'],val_ce=vals[-1]['val_ce'],best_val_ce=s['best_val_ce'],
                            training_seconds=s['training_seconds'],wall_seconds=s['wall_seconds'],gpu=c['gpu'],
                            resume_val_absolute_difference=error,checks_passed=True,wandb_finished=True,
                            curve=[dict(step=v['step'],ce=v['val_ce']) for v in vals]))
    records.sort(key=lambda row:row['lr'])
    (ROOT/'reports/baseline-lr-audit.json').write_text(json.dumps(dict(all_passed=True,runs=records,common=common),indent=2))
    (ROOT/'reports/baseline-lr-wandb.json').write_text(json.dumps({r['run_id']:remote[r['run_id']] for r in records},indent=2))
    for row in records:print(json.dumps({k:row[k] for k in ('job','lr','run_id','val_ce','gpu')}))


if __name__=='__main__':main()
