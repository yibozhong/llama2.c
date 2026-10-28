"""Verify all completed experiment metadata against queue logs and W&B."""
import base64
import concurrent.futures
import csv
import json
import math
import netrc
from pathlib import Path
import re
import urllib.request

ROOT=Path('/home/yzhong239/gradformer')
QUERY='query($entity:String!,$project:String!,$run:String!){project(name:$project,entityName:$entity){run(name:$run){name state config summaryMetrics}}}'


def main():
    manifests=json.loads((ROOT/'reports/local-gradient-manifests.json').read_text())
    records=[]
    for cluster,entry in manifests.items():
        for proc,line in enumerate((ROOT/'cluster'/entry['manifest']).read_text().splitlines()):
            alpha,beta,lr,label,checks=line.split();job=f'{cluster}.{proc}'
            folder=ROOT/f'results/pretrain-llama-local-gradient-{job}-metadata'
            summary=json.loads((folder/'local-gradient-summary.json').read_text());config=summary['config']
            assert (config['alpha'],config['beta'],config['parameter_lr'],config['label'])==(float(alpha),float(beta),float(lr),label)
            assert config['hidden_step_size']==1 and config['seed']==1337 and config['schedule_steps']==6104
            assert config['source']['git_commit']==entry['source'] and not config['source']['dirty']
            log=(ROOT/f'logs/pretrain-llama-local-gradient.{job}.log').read_text()
            assert re.findall(r'Normal termination \(return value (\d+)\)',log)[-1]=='0'
            assert json.loads((folder/'local-gradient-check.json').read_text())['passed']
            assert summary['checkpoint_audit']['passed']
            final=summary['final_validation'];assert final['step']==1000 and final['tokens']==32768000
            rows=[json.loads(line) for line in (folder/'local-gradient/metrics.jsonl').read_text().splitlines()]
            assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v,(float,int)))
            records.append(dict(job=job,run_id=summary['run_id'],alpha=config['alpha'],beta=config['beta'],lr=config['parameter_lr'],
                                label=label,gpu=config['gpu'],source=config['source'],final=final,
                                validation=[r for r in rows if 'val_ce' in r],last_training=[r for r in rows if 'train_ce' in r][-1],
                                median_step_seconds=summary['median_step_seconds'],audit=summary['checkpoint_audit']))
    assert len(records)==6
    auth='Basic '+base64.b64encode(('api:'+netrc.netrc().authenticators('api.wandb.ai')[2]).encode()).decode()
    def verify(row):
        payload={'query':QUERY,'variables':{'entity':'yibozhong657-none','project':'nanochat','run':row['run_id']}}
        req=urllib.request.Request('https://api.wandb.ai/graphql',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':auth})
        with urllib.request.urlopen(req,timeout=30) as response:obj=json.load(response)
        assert not obj.get('errors');run=obj['data']['project']['run']
        for key in ('config','summaryMetrics'):
            if isinstance(run[key],str):run[key]=json.loads(run[key])
        assert run['state']=='finished'
        for key,value in row['final'].items():assert math.isclose(run['summaryMetrics'][key],value,rel_tol=0,abs_tol=1e-8),(row['run_id'],key)
        assert run['summaryMetrics']['local_gradient_checkpoint_audit.passed']
        for key,value in (('alpha',row['alpha']),('beta',row['beta']),('parameter_lr',row['lr'])):
            assert run['config'][key]['value']==value
        return row['run_id'],run
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:remote=dict(pool.map(verify,records))
    (ROOT/'reports/local-gradient-api-verification.json').write_text(json.dumps(remote,indent=2))
    (ROOT/'reports/local-gradient-records.json').write_text(json.dumps(records,indent=2))
    with (ROOT/'reports/local-gradient-summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['job','run_id','alpha','beta','lr','label','val_ce','val_cosine','val_mse','gpu'],lineterminator='\n');writer.writeheader()
        for row in records:writer.writerow({**{k:row[k] for k in ['job','run_id','alpha','beta','lr','label','gpu']},'val_ce':row['final']['val_ce'],'val_cosine':row['final']['val_gradient/cosine'],'val_mse':row['final']['val_gradient/mse']})
    print('Verified completed runs:',len(records))
    for row in sorted(records,key=lambda r:r['final']['val_ce']):
        print(json.dumps({k:row[k] for k in ['job','run_id','label','alpha','beta','lr','gpu']}),row['final']['val_ce'],row['final']['val_gradient/cosine'])


if __name__=='__main__':main()
