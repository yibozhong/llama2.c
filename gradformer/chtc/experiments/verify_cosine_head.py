"""Metadata and W&B verification for the cosine/readout sweep (no model compute)."""
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
    manifests=json.loads((ROOT/'reports/cosine-head-manifests.json').read_text())
    records=[]
    for cluster,filename in manifests.items():
        for proc,line in enumerate((ROOT/'cluster'/filename).read_text().splitlines()):
            eta,lr,label,mode,last_eta,checks=line.split();job=f'{cluster}.{proc}'
            folder=ROOT/f'results/pretrain-llama-cosine-head-{job}-metadata'
            summary_path=folder/'cosine-head-summary.json'
            if not summary_path.exists():continue
            summary=json.loads(summary_path.read_text());config=summary['config']
            assert config['hidden_step_size']==float(eta) and config['parameter_lr']==float(lr)
            assert config['mode']==mode and config['last_eta']==last_eta and config['label']==label
            assert config['seed']==1337 and config['schedule_steps']==6104 and config['batch_tokens']==32768
            assert config['target_source']=='online; no EMA' and config['mse_coefficient']==0
            assert config['source']['git_commit']=='b89b225dfaf2770b0f42470b6ccd453dec0f4790' and not config['source']['dirty']
            logs=(ROOT/f'logs/pretrain-llama-cosine-head.{job}.log').read_text()
            assert re.findall(r'Normal termination \(return value (\d+)\)',logs)[-1]=='0'
            assert json.loads((folder/'cosine-head-check.json').read_text())['passed']
            assert summary['checkpoint_audit']['passed']
            final=summary['final_validation'];assert final['step']==1000 and final['tokens']==32768000
            rows=[json.loads(line) for line in (folder/'cosine-head/metrics.jsonl').read_text().splitlines()]
            assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v,(float,int)))
            records.append(dict(job=job,eta=float(eta),lr=float(lr),label=label,mode=mode,last_eta=last_eta,
                                source=config['source'],run_id=summary['run_id'],gpu=config['gpu'],
                                final=final,validation=[r for r in rows if 'val_ce' in r],
                                last_training=[r for r in rows if 'train_ce' in r][-1],
                                median_step_seconds=summary['median_step_seconds'],audit=summary['checkpoint_audit']))
    auth='Basic '+base64.b64encode(('api:'+netrc.netrc().authenticators('api.wandb.ai')[2]).encode()).decode()
    def verify(row):
        req=urllib.request.Request('https://api.wandb.ai/graphql',data=json.dumps({'query':QUERY,'variables':{'entity':'yibozhong657-none','project':'nanochat','run':row['run_id']}}).encode(),headers={'Content-Type':'application/json','Authorization':auth})
        with urllib.request.urlopen(req,timeout=30) as response: obj=json.load(response)
        assert not obj.get('errors')
        run=obj['data']['project']['run'];run['config']=json.loads(run['config']);run['summaryMetrics']=json.loads(run['summaryMetrics'])
        assert run['state']=='finished'
        for k,v in row['final'].items():assert math.isclose(run['summaryMetrics'][k],v,rel_tol=0,abs_tol=1e-8),(row['run_id'],k)
        assert run['summaryMetrics']['cosine_head_checkpoint_audit.passed']
        assert run['config']['parameter_lr']['value']==row['lr'] and run['config']['hidden_step_size']['value']==row['eta']
        return row['run_id'],run
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: remote=dict(pool.map(verify,records))
    (ROOT/'reports/cosine-head-api-verification.json').write_text(json.dumps(remote,indent=2))
    (ROOT/'reports/cosine-head-records.json').write_text(json.dumps(records,indent=2))
    with (ROOT/'reports/cosine-head-summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['job','run_id','mode','last_eta','eta','lr','val_ce','val_cosine','gpu'],lineterminator='\n');writer.writeheader()
        for r in records:
            writer.writerow({**{k:r[k] for k in ['job','run_id','mode','last_eta','eta','lr','gpu']},'val_ce':r['final']['val_ce'],'val_cosine':r['final']['val_cosine/mean']})
    print('Verified completed runs:',len(records))
    for r in sorted(records,key=lambda r:r['final']['val_ce']):
        print(json.dumps({k:r[k] for k in ['job','run_id','mode','last_eta','eta','lr','gpu']}),r['final']['val_ce'],r['final']['val_cosine/mean'])

if __name__=='__main__':main()
