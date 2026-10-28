"""Token-weighted exhaustive validation for selected LR-sweep checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from gradformer.local_ce import validation_metrics
from gradformer.runtime import construct


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--selection',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    selection_path=Path(args.selection);selection=json.loads(selection_path.read_text())
    data=np.memmap(Path(args.data)/'val.bin',dtype=np.uint16,mode='r')
    manifest=json.loads((Path(args.data)/'manifest.json').read_text())
    sequence=256;sequences=(len(data)-1)//sequence
    import wandb
    run=wandb.init(project='nanochat',entity='yibozhong657-none',job_type='evaluation',
        name='gradformer-local-lr-selected-full-validation',group='gradformer-local-sgd-lr-sweep-20261028',
        config={'selection':selection,'dataset':manifest,'gpu':torch.cuda.get_device_name(),
                'evaluation_source':json.loads(Path('gradformer/source.json').read_text()),
                'protocol':'every full nonoverlapping 256-token sequence; all-token CCE, token-weighted batches',
                'validation_tokens':sequences*sequence,'excluded_tail_tokens':len(data)-1-sequences*sequence})
    results=[];exit_code=1
    try:
        for candidate in selection:
            folder=selection_path.parent/candidate['directory']
            config=json.loads((folder/'config.json').read_text())
            assert config['dataset']==manifest
            assert config['model']['hidden_update']=='sgd' and config['ce_implementation'].startswith('CCE full vocabulary')
            m=config['model']
            model=construct('llama2','sgd',depth=m['n_layers'],width=m['dim'],heads=m['n_heads'],vocab=m['vocab_size'],sequence=sequence,step_size=m['hidden_step_size'],device='cuda')
            model.load_state_dict(torch.load(folder/'model.pt',map_location='cuda',weights_only=True));model.eval()
            totals={};tokens=0;start=time.perf_counter()
            for first in range(0,sequences,16):
                n=min(16,sequences-first);begin=first*sequence;end=(first+n)*sequence
                x=torch.from_numpy(np.asarray(data[begin:end],dtype=np.int64).copy().reshape(n,sequence)).cuda()
                y=torch.from_numpy(np.asarray(data[begin+1:end+1],dtype=np.int64).copy().reshape(n,sequence)).cuda()
                metrics=validation_metrics(model,[(x,y)])
                weight=n*sequence;tokens+=weight
                for key,value in metrics.items():totals[key]=totals.get(key,0.)+weight*value
                if first%1024==0:
                    run.log({'candidate':candidate['run_id'],'evaluated_tokens':tokens,'partial_full_val_ce':totals['val_ce']/tokens})
            torch.cuda.synchronize()
            record={**candidate,'tokens':tokens,'seconds':time.perf_counter()-start,
                    'model_sha256':hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest(),
                    'metrics':{k:v/tokens for k,v in totals.items()}}
            results.append(record);print(json.dumps(record),flush=True)
            del model;torch.cuda.empty_cache()
        assert len(results)==2 and {r['objective'] for r in results}=={'local','e2e'}
        final={r['objective']:r['metrics']['val_ce'] for r in results}
        result={'gpu':torch.cuda.get_device_name(),'results':results,'local_minus_e2e':final['local']-final['e2e']}
        (out/'full-validation.json').write_text(json.dumps(result,indent=2))
        (out/'evaluation-run-url.txt').write_text(run.url)
        run.summary.update({'full_val_ce/'+k:v for k,v in final.items()})
        run.summary['local_minus_e2e']=result['local_minus_e2e']
        artifact=wandb.Artifact('gradformer-local-lr-selected-models-and-validation',type='evaluation')
        artifact.add_dir(str(out));artifact.add_file(str(selection_path));artifact.add_file(__file__)
        for candidate in selection:
            artifact.add_dir(str(selection_path.parent/candidate['directory']),name=candidate['directory'])
        run.log_artifact(artifact).wait();exit_code=0
    finally:run.finish(exit_code=exit_code)

if __name__=='__main__':main()
