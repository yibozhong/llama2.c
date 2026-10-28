"""Audit final local-gradient checkpoints and publish metadata to the same run."""
import argparse
import json
import math
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    folder=Path(args.directory);latest=json.loads((folder/'latest.json').read_text());assert latest['step']==1000
    config=json.loads((folder/'config.json').read_text())
    model=torch.load(folder/latest['directory']/'model.pt',map_location='cpu',weights_only=True)
    state=torch.load(folder/latest['directory']/'rank-0.pt',map_location='cpu',weights_only=False)
    assert state['step']==1000
    assert all(torch.isfinite(v).all() for v in model.values() if torch.is_floating_point(v))
    assert state['optimizer']['state']
    for item in state['optimizer']['state'].values():
        for value in item.values():
            if isinstance(value,torch.Tensor) and torch.is_floating_point(value):
                assert torch.isfinite(value).all()
    rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
    assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v,(int,float)))
    for row in rows:
        if 'parameter_lr' in row:
            step=row['step']-1
            factor=(step+1)/100 if step<100 else .5*(1+math.cos(math.pi*(step-100)/(6104-100)))
            assert math.isclose(row['parameter_lr'],config['parameter_lr']*factor,rel_tol=1e-12)
    final=[row for row in rows if 'val_ce' in row][-1]
    assert final['step']==1000 and final['tokens']==32768000
    assert not torch.equal(model['norm.weight'],torch.ones_like(model['norm.weight']))
    eta=F.softplus(model['hidden_step_raw'])
    assert torch.isfinite(eta).all() and (eta>0).all()
    audit=dict(passed=True,step=1000,finite_model=True,finite_optimizer=True,lr_schedule_checked=True,
               final_readout_norm_trained=True,final_etas=eta.tolist())
    result=dict(run_id=(folder/'wandb-id.txt').read_text().strip(),config=config,final_validation=final,
                checkpoint_audit=audit,median_step_seconds=statistics.median(r['step_seconds'] for r in rows if 'step_seconds' in r))
    Path(args.output).write_text(json.dumps(result,indent=2))
    import wandb
    remote=wandb.Api().run('yibozhong657-none/nanochat/'+result['run_id'])
    remote.summary['local_gradient_checkpoint_audit']=audit;remote.summary.update()
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
