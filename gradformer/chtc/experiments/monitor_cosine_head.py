"""Read-only live/archive metadata collector; no model computation on the AP."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path('/home/yzhong239/gradformer')

def inspect(cluster, proc, state):
    key=f'{cluster}.{proc}'
    archive=ROOT/f'results/pretrain-llama-cosine-head-{key}.tar'
    folder=ROOT/f'results/pretrain-llama-cosine-head-{key}-metadata'
    result={'job':key,'job_status':state.get('JobStatus'),'host':state.get('RemoteHost')}
    output=''
    if state.get('JobStatus')==2:
        try:
            output=subprocess.run(['condor_tail',key,'-maxbytes','250000'],capture_output=True,text=True,timeout=8).stdout
        except subprocess.TimeoutExpired:
            result['tail_timeout']=True
    elif (ROOT/f'logs/pretrain-llama-cosine-head.{key}.out').exists():
        output=(ROOT/f'logs/pretrain-llama-cosine-head.{key}.out').read_text()
    rows=[]
    for line in output.splitlines():
        try:
            row=json.loads(line)
            if isinstance(row,dict): rows.append(row)
        except ValueError: pass
    if rows:
        metrics=[r for r in rows if 'step' in r]
        vals=[r for r in metrics if 'val_ce' in r]
        checks=[r['check'] for r in rows if 'check' in r]
        result.update(checks=checks)
        if metrics: result['step']=metrics[-1]['step']
        if vals: result.update(val_step=vals[-1]['step'],val_ce=vals[-1]['val_ce'])
        runtime=[r for r in rows if 'container_sha256' in r]
        if runtime: result['gpu']=runtime[-1]['gpu']
        (ROOT/f'reports/cosine-head-live-{key}.json').write_text(json.dumps(rows,indent=2))
    # Only harvest after Condor has removed the completed job from its queue.
    if archive.exists() and not state:
        folder.mkdir(exist_ok=True)
        if not (folder/'archive-entries.json').exists():
            entries=[]
            with tarfile.open(archive) as stream:
                for member in stream:
                    entries.append({'name':member.name,'size':member.size})
                    path=Path(member.name)
                    if not member.isfile() or path.parts[0]!='results' or '..' in path.parts: continue
                    rel=Path(*path.parts[1:])
                    keep=(len(rel.parts)==1 and rel.suffix in ('.json','.txt')) or (rel.parts[0]=='cosine-head' and rel.name in ('config.json','latest.json','metrics.jsonl','wandb-id.txt'))
                    if keep:
                        dest=folder/rel;dest.parent.mkdir(exist_ok=True,parents=True)
                        dest.write_bytes(stream.extractfile(member).read())
            (folder/'archive-entries.json').write_text(json.dumps(entries,indent=2))
        summary=folder/'cosine-head-summary.json'
        if summary.exists():
            record=json.loads(summary.read_text());config=record['config']
            result.update(finished=True,run_id=record['run_id'],eta=config['hidden_step_size'],lr=config['parameter_lr'],objective=config['mode'],gpu=config['gpu'],val_ce=record['final_validation']['val_ce'],val_step=1000,step=1000,checkpoint_passed=record['checkpoint_audit']['passed'],last_eta=config['last_eta'],val_cosine=record['final_validation']['val_cosine/mean'])
        else:
            result['finished']=False
            err=ROOT/f'logs/pretrain-llama-cosine-head.{key}.err'
            if err.exists(): result['error_tail']=err.read_text()[-2200:]
    return result


def main():
    old_path=ROOT/'reports/cosine-head-status.json'
    previous={r['job']:r for r in json.loads(old_path.read_text())} if old_path.exists() else {}
    counts=[tuple(map(int,s.split(':'))) for s in sys.argv[1:]]
    queue=json.loads(subprocess.check_output(['condor_q','-json','-attributes','ClusterId,ProcId,JobStatus,RemoteHost'],text=True))
    states={(r['ClusterId'],r['ProcId']):r for r in queue}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        jobs=[pool.submit(inspect,c,p,states.get((c,p),{})) for c,n in counts for p in range(n)]
        results=[job.result() for job in jobs]
    (ROOT/'reports/cosine-head-status.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({'total':len(results),'finished':sum(bool(r.get('finished')) for r in results),'running':sum(r.get('job_status')==2 for r in results),'idle':sum(r.get('job_status')==1 for r in results),'held':sum(r.get('job_status')==5 for r in results)}))
    for row in results:
        old=previous.get(row['job'],{})
        if row.get('job_status') in (2,5) or row.get('error_tail') or (row.get('finished') and not old.get('finished')):
            print(json.dumps({k:v for k,v in row.items() if k not in ('checks','host')}))

if __name__=='__main__': main()
