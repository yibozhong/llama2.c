"""Verify archived sweep metadata and W&B; does not load model tensors on AP."""
import base64
import concurrent.futures
import csv
import json
import math
import netrc
import re
from pathlib import Path
import urllib.request

ROOT=Path('/home/yzhong239/gradformer')
SOURCE='645b012e5d30ab3b032a48d1535399810a2553bc'
QUERY='query($entity:String!,$project:String!,$run:String!){project(name:$project,entityName:$entity){run(name:$run){name state config summaryMetrics}}}'


def main():
    manifests={7333513:ROOT/'cluster/local-lr-grid.tsv',7333514:ROOT/'cluster/local-lr-e2e.tsv'}
    # Optional refinement manifest records cluster and trials filename.
    extra=ROOT/'reports/local-lr-extra-manifests.json'
    if extra.exists():
        for key,value in json.loads(extra.read_text()).items():manifests[int(key)]=ROOT/'cluster'/value
    records=[]
    previous=json.loads((ROOT/'results/pretrain-llama-local-sgd-7332874-metadata/local-ce/config.json').read_text())
    for cluster,manifest in manifests.items():
        for proc,line in enumerate(manifest.read_text().splitlines()):
            eta,lr,label,objective,checks=line.split();eta,lr=float(eta),float(lr)
            folder=ROOT/f'results/pretrain-llama-local-lr-{cluster}.{proc}-metadata'
            if not (folder/'local-pilot-summary.json').exists():continue
            events=(ROOT/f'logs/pretrain-llama-local-lr.{cluster}.{proc}.log').read_text()
            exits=re.findall(r'Normal termination \(return value (\d+)\)',events)
            assert exits and exits[-1]=='0',(cluster,proc,exits)
            s=json.loads((folder/'local-pilot-summary.json').read_text())
            c=s['config'];rows=[json.loads(x) for x in (folder/'local-ce/metrics.jsonl').read_text().splitlines()]
            assert c['hidden_step_size']==eta and c['model']['hidden_step_size']==eta and c['parameter_lr']==lr
            assert c['training_objective']==objective and c['readout_training']=='final' and c['hidden_update']=='sgd'
            assert c['source']['git_commit']==SOURCE and not c['source']['dirty']
            for key in ('seed','dataset','batch','sequence','batch_tokens','accumulation','schedule_steps'): assert c[key]==previous[key],key
            assert c['final_ce_coefficient']==1/6
            assert all(g['initial_lr']==lr for g in c['optimizer_groups'])
            assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
            for r in rows:
                if 'parameter_lr' not in r: continue
                step=r['step']-1
                mult=(step+1)/100 if step<100 else .5*(1+math.cos(math.pi*(step-100)/(6104-100)))
                assert math.isclose(r['parameter_lr'],lr*mult,rel_tol=1e-12,abs_tol=0)
            audit=s['checkpoint_audit'];assert audit['passed'] and audit['finite_model'] and audit['finite_optimizer'] and audit['step']==1000
            assert json.loads((folder/'local-lr-check.json').read_text())['passed']
            vals=[r for r in rows if 'val_ce' in r];assert vals[-1]['step']==1000 and vals[-1]['tokens']==32768000
            assert vals[-1]==s['final_validation']
            train=[r for r in rows if 'train_ce' in r]
            records.append({'job':f'{cluster}.{proc}','run_id':s['run_id'],'eta':eta,'lr':lr,'label':label,'objective':objective,'gpu':c['gpu'],'source':c['source'],'final':vals[-1],'validation':vals,'last_training':train[-1],'median_step_seconds':s['median_step_seconds'],'checkpoint_audit':audit})
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    auth='Basic '+base64.b64encode(('api:'+key).encode()).decode()
    def verify(row):
        data=json.dumps({'query':QUERY,'variables':{'entity':'yibozhong657-none','project':'nanochat','run':row['run_id']}}).encode()
        req=urllib.request.Request('https://api.wandb.ai/graphql',data=data,headers={'Content-Type':'application/json','Authorization':auth})
        with urllib.request.urlopen(req,timeout=30) as stream: obj=json.load(stream)
        assert not obj.get('errors'),obj.get('errors')
        remote=obj['data']['project']['run'];remote['summaryMetrics']=json.loads(remote['summaryMetrics']);remote['config']=json.loads(remote['config'])
        assert remote['state']=='finished',(row['run_id'],remote['state'])
        for k,v in row['final'].items(): assert math.isclose(remote['summaryMetrics'][k],v,rel_tol=0,abs_tol=1e-8),(row['run_id'],k)
        for k in ('passed','finite_model','finite_optimizer'): assert remote['summaryMetrics']['local_checkpoint_audit.'+k]
        assert remote['config']['parameter_lr']['value']==row['lr'] and remote['config']['hidden_step_size']['value']==row['eta']
        return row['run_id'],remote
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: remote=dict(pool.map(verify,records))
    (ROOT/'reports/local-lr-api-verification.json').write_text(json.dumps(remote,indent=2))
    (ROOT/'reports/sweep-records.json').write_text(json.dumps(records,indent=2))
    with (ROOT/'reports/local-lr-summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['job','run_id','objective','eta','lr','val_ce','gpu'],lineterminator='\n');writer.writeheader()
        for r in records:writer.writerow({**{k:r[k] for k in ('job','run_id','objective','eta','lr','gpu')},'val_ce':r['final']['val_ce']})
    print(f'Verified {len(records)} completed candidates against metadata, LR schedule, checkpoint audits and W&B API.')
    for r in sorted(records,key=lambda r:(r['objective'],r['final']['val_ce'])):
        print(json.dumps({k:r[k] for k in ('job','run_id','objective','eta','lr')},ensure_ascii=False),r['final']['val_ce'])

if __name__=='__main__':main()
