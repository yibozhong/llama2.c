"""CPU-job report and external checkpoint archive for the local constraint runs."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb


def main():
    out=Path('results');out.mkdir(exist_ok=True)
    source=Path('local-gradient-records.json');rows=json.loads(source.read_text());assert len(rows)==6
    best=min((r for r in rows if r['alpha'] or r['beta']),key=lambda r:r['final']['val_ce'])
    base=next(r for r in rows if not r['alpha'] and not r['beta'])
    run=wandb.init(project='nanochat',entity='yibozhong657-none',job_type='report',
                   name='local-ce-gradient-constraint-results',group='gradformer-local-gradient-20261028',
                   config={'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           'records_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'training_runs':6,
                           'seed':1337,'tokens_per_run':32768000})
    code=1
    try:
        api=wandb.Api()
        for row in rows:
            remote=api.run('yibozhong657-none/nanochat/'+row['run_id'])
            assert remote.state=='finished' and abs(remote.summary['val_ce']-row['final']['val_ce'])<1e-8
            assert remote.summary['local_gradient_checkpoint_audit.passed']
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
        for row in rows:
            label=f"{row['label']} (MSE={row['alpha']:g}, cos={row['beta']:g}, LR={row['lr']:g})"
            for ax,key in zip(axes,['val_ce','val_gradient/cosine','val_gradient/prediction_norm']):
                ax.plot([r['step'] for r in row['validation']],[r[key] for r in row['validation']],label=label)
        for ax in axes:ax.set_xlabel('Training step');ax.grid(alpha=.2)
        axes[0].set_ylabel('Validation CE (nats/token)')
        axes[1].set_ylabel('Local input-gradient cosine');axes[1].set_ylim(-1,1)
        axes[2].set_ylabel('Predicted direction norm');axes[2].set_yscale('log')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',fontsize=8,ncol=2)
        fig.suptitle('Local CE + local input-gradient supervision: 1000 steps, one seed')
        for ext in ('png','pdf','svg'):fig.savefig(out/f'local-gradient-curves.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
        summary={'verified_training_runs':6,'best_constraint_run':best['run_id'],'best_constraint_ce':best['final']['val_ce'],
                 'best_constraint_cosine':best['final']['val_gradient/cosine'],'control_ce':base['final']['val_ce'],
                 'control_cosine':base['final']['val_gradient/cosine'],'constraint_minus_control_ce':best['final']['val_ce']-base['final']['val_ce']}
        (out/'summary.json').write_text(json.dumps(summary,indent=2));run.summary.update(summary)
        run.log({'curves':wandb.Image(str(out/'local-gradient-curves.png'))})
        artifact=wandb.Artifact('gradformer-local-gradient-report-and-selected-models',type='report')
        for path in out.iterdir():
            if path.is_file():artifact.add_file(str(path))
        artifact.add_file(str(source));artifact.add_file(__file__);artifact.add_file('local-gradient-selected-models.tar')
        run.log_artifact(artifact).wait()
        (out/'report-url.txt').write_text(run.url);print(json.dumps(summary),flush=True);code=0
    finally:run.finish(exit_code=code)


if __name__=='__main__':main()
