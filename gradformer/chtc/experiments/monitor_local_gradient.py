"""Collect queue/log/archive metadata without running model computation."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path('/home/yzhong239/gradformer')


def inspect(cluster,proc,state):
    key=f'{cluster}.{proc}';record={'job':key,'job_status':state.get('JobStatus'),'host':state.get('RemoteHost')}
    archive=ROOT/f'results/pretrain-llama-local-gradient-{key}.tar'
    if state.get('JobStatus')==2:
        try:out=subprocess.run(['condor_tail',key,'-maxbytes','150000'],capture_output=True,text=True,timeout=8).stdout
        except subprocess.TimeoutExpired:out=''
    else:
        path=ROOT/f'logs/pretrain-llama-local-gradient.{key}.out';out=path.read_text() if path.exists() else ''
    rows=[]
    for line in out.splitlines():
        try:
            item=json.loads(line)
            if isinstance(item,dict):rows.append(item)
        except ValueError:pass
    steps=[r for r in rows if 'step' in r];values=[r for r in steps if 'val_ce' in r]
    if steps:record['step']=steps[-1]['step']
    if values:record.update(val_step=values[-1]['step'],val_ce=values[-1]['val_ce'],val_cosine=values[-1].get('val_gradient/cosine'))
    record['checks']=[r['check'] for r in rows if 'check' in r]
    record['gate50']='Passed local-gradient 50-step gate' in out
    if rows:(ROOT/f'reports/local-gradient-live-{key}.json').write_text(json.dumps(rows,indent=2))
    if archive.exists() and not state:
        folder=ROOT/f'results/pretrain-llama-local-gradient-{key}-metadata';folder.mkdir(exist_ok=True)
        if not (folder/'archive-entries.json').exists():
            entries=[]
            with tarfile.open(archive) as tf:
                for m in tf:
                    entries.append({'name':m.name,'size':m.size});p=Path(m.name)
                    if not m.isfile() or '..' in p.parts or p.parts[0]!='results':continue
                    rel=Path(*p.parts[1:])
                    keep=(len(rel.parts)==1 and rel.suffix in ('.json','.txt')) or (rel.parts[0]=='local-gradient' and rel.name in ('config.json','latest.json','metrics.jsonl','wandb-id.txt'))
                    if keep:
                        dest=folder/rel;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(tf.extractfile(m).read())
            (folder/'archive-entries.json').write_text(json.dumps(entries,indent=2))
        summary=folder/'local-gradient-summary.json'
        if summary.exists():
            data=json.loads(summary.read_text());config=data['config'];v=data['final_validation']
            record.update(finished=True,run_id=data['run_id'],alpha=config['alpha'],beta=config['beta'],lr=config['parameter_lr'],
                          gpu=config['gpu'],step=v['step'],val_step=v['step'],val_ce=v['val_ce'],val_cosine=v['val_gradient/cosine'],
                          checkpoint_passed=data['checkpoint_audit']['passed'])
        else:
            err=ROOT/f'logs/pretrain-llama-local-gradient.{key}.err'
            if err.exists():record['error_tail']=err.read_text()[-4500:]
    return record


def main():
    counts=[tuple(map(int,s.split(':'))) for s in sys.argv[1:]]
    queue=json.loads(subprocess.check_output(['condor_q','-json','-attributes','ClusterId,ProcId,JobStatus,RemoteHost'],text=True))
    states={(r['ClusterId'],r['ProcId']):r for r in queue}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(inspect,c,p,states.get((c,p),{})) for c,n in counts for p in range(n)]
        records=[f.result() for f in futures]
    (ROOT/'reports/local-gradient-status.json').write_text(json.dumps(records,indent=2))
    for row in records:print(json.dumps(row))


if __name__=='__main__':main()
