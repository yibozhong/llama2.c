"""Read scheduler/log/archive metadata only; no model computation on AP."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path('/home/yzhong239/gradformer')


def inspect(cluster,proc,state):
    key=f'{cluster}.{proc}';record={'job':key,'status':state.get('JobStatus'),'host':state.get('RemoteHost')}
    if state.get('JobStatus')==2:
        try:
            out=subprocess.run(['condor_tail',key,'-maxbytes','180000'],capture_output=True,text=True,timeout=8).stdout
        except subprocess.TimeoutExpired:
            out=''
    else:
        p=ROOT/f'logs/pretrain-llama-energy.{key}.out';out=p.read_text() if p.exists() else ''
    rows=[]
    for line in out.splitlines():
        try:
            r=json.loads(line)
            if isinstance(r,dict):rows.append(r)
        except ValueError:pass
    if rows:(ROOT/f'reports/energy-live-{key}.json').write_text(json.dumps(rows,indent=2))
    record['checks']=[r['check'] for r in rows if 'check' in r]
    record['gate50']='Passed energy 50-step gate' in out
    phases=[];previous=-1
    for row in rows:
        if 'step' not in row:continue
        if row['step']<previous or not phases:phases.append({})
        phases[-1]['step']=row['step'];previous=row['step']
        if 'val_ce' in row:phases[-1].update(val_step=row['step'],val_ce=row['val_ce'])
        if 'step_seconds' in row:phases[-1]['step_seconds']=row['step_seconds']
    record['phases']=phases
    archive=ROOT/f'results/pretrain-llama-energy-{key}.tar'
    if archive.exists() and not state:
        folder=ROOT/f'results/pretrain-llama-energy-{key}-metadata';folder.mkdir(exist_ok=True)
        if not (folder/'archive-entries.json').exists():
            entries=[]
            with tarfile.open(archive) as tf:
                for m in tf:
                    entries.append({'name':m.name,'size':m.size});p=Path(m.name)
                    if not m.isfile() or '..' in p.parts or p.parts[0]!='results':continue
                    rel=Path(*p.parts[1:])
                    keep=(len(rel.parts)==1 and rel.suffix in ('.json','.txt')) or (rel.parts[0] in ('energy','baseline') and len(rel.parts)==2 and rel.name in ('config.json','latest.json','metrics.jsonl','wandb-id.txt','summary.json'))
                    if keep:
                        dest=folder/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(tf.extractfile(m).read())
            (folder/'archive-entries.json').write_text(json.dumps(entries,indent=2))
        record['runs']=[]
        for name in ('energy','baseline'):
            summary=folder/name/'summary.json'
            if not summary.exists():continue
            data=json.loads(summary.read_text());cfg=data['config']
            metrics=[json.loads(s) for s in (folder/name/'metrics.jsonl').read_text().splitlines()]
            vals=[r for r in metrics if 'val_ce' in r]
            record['runs'].append(dict(kind=cfg['kind'],eta=cfg['hidden_step_size'],lr=cfg['parameter_lr'],
                                      step=data['last_step'],val_ce=vals[-1]['val_ce'],run_id=data['run_id'],
                                      training_seconds=data['training_seconds'],wall_seconds=data['wall_seconds'],
                                      checkpoint_exact=data['checkpoint_exact'],gpu=cfg['gpu']))
        record['archive_bytes']=archive.stat().st_size
        if not record['runs']:
            err=ROOT/f'logs/pretrain-llama-energy.{key}.err'
            if err.exists():record['error_tail']=err.read_text()[-5000:]
    return record


def main():
    counts=[tuple(map(int,s.split(':'))) for s in sys.argv[1:]]
    queue=json.loads(subprocess.check_output(['condor_q','-json','-attributes','ClusterId,ProcId,JobStatus,RemoteHost'],text=True))
    states={(r['ClusterId'],r['ProcId']):r for r in queue}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        jobs=[pool.submit(inspect,c,p,states.get((c,p),{})) for c,n in counts for p in range(n)]
        records=[f.result() for f in jobs]
    (ROOT/'reports/energy-status.json').write_text(json.dumps(records,indent=2))
    for row in records:print(json.dumps(row))


if __name__=='__main__':main()
