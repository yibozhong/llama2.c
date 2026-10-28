"""Render and archive the completed cosine/readout experiment on a CPU job."""
import hashlib
import json
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb


def main():
    out=Path('results');out.mkdir(exist_ok=True)
    source=Path('cosine-head-records.json');rows=json.loads(source.read_text());assert len(rows)==8
    best=min((r for r in rows if r['mode']=='cosine-head'),key=lambda r:r['final']['val_ce'])
    pure=next(r for r in rows if r['mode']=='cosine-only')
    ntp=next(r for r in rows if r['mode']=='ntp-only')
    center=next(r for r in rows if r['label']=='center')
    fixed=next(r for r in rows if r['last_eta']=='fixed' and r['mode']=='cosine-head')
    run=wandb.init(project='nanochat',entity='yibozhong657-none',job_type='report',
        name='cosine-blocks-ntp-readout-lr-results',group='gradformer-cosine-head-lr-20261028',
        config={'training_source':'b89b225dfaf2770b0f42470b6ccd453dec0f4790',
                'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'records_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'training_runs':8,'seed':1337,'tokens_per_run':32768000})
    code=1
    try:
        api=wandb.Api()
        for row in rows:
            remote=api.run('yibozhong657-none/nanochat/'+row['run_id'])
            assert remote.state=='finished'
            assert abs(remote.summary['val_ce']-row['final']['val_ce'])<1e-8
            assert remote.summary['cosine_head_checkpoint_audit.passed']
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        colors={best['run_id']:'#dc2626',pure['run_id']:'#16a34a',ntp['run_id']:'#2563eb',fixed['run_id']:'#d97706'}
        for row in rows:
            color=colors.get(row['run_id'],'#9ca3af')
            label=(row['mode']+'; '+row['label']+f" (eta={row['eta']:g}, lr={row['lr']:g})")
            for ax,key in zip(axes,['val_ce','val_cosine/mean']):
                ax.plot([r['step'] for r in row['validation']],[r[key] for r in row['validation']],color=color,
                        alpha=1 if row['run_id'] in colors else .6,lw=2 if row['run_id'] in colors else 1,label=label)
        for ax in axes:ax.set_xlabel('Training step');ax.grid(alpha=.2)
        axes[0].set_ylabel('Validation CE (nats/token)');axes[1].set_ylabel('Validation prediction / target cosine')
        axes[1].set_ylim(-.02,1.)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',fontsize=7,ncol=2)
        fig.suptitle('Cosine blocks + NTP readout: 1000-step sweep and controls')
        for ext in ['png','pdf','svg']:fig.savefig(out/f'cosine-head-curves.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for row,label,color in [(best,'Best cosine + head','#dc2626'),(pure,'Pure cosine','#16a34a'),(ntp,'NTP-only','#2563eb')]:
            ax.plot(range(1,7),[row['final'][f'val_cosine/layer_{i}/mean'] for i in range(6)],marker='o',color=color,label=label)
        ax.set(xlabel='Layer',ylabel='Validation cosine',ylim=(-.02,1.),title="Each model uses its own gradient targets")
        ax.grid(alpha=.2);ax.legend()
        for ext in ['png','pdf','svg']:fig.savefig(out/f'cosine-head-layers.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
        summary={'verified_training_runs':8,'best_cosine_head_run':best['run_id'],'best_cosine_head_ce':best['final']['val_ce'],
                 'best_cosine_head_cosine':best['final']['val_cosine/mean'],'pure_cosine_ce':pure['final']['val_ce'],
                 'pure_cosine_cosine':pure['final']['val_cosine/mean'],'ntp_only_ce':ntp['final']['val_ce'],
                 'center_ce':center['final']['val_ce'],'fixed_last_eta_ce':fixed['final']['val_ce']}
        (out/'summary.json').write_text(json.dumps(summary,indent=2));run.summary.update(summary)
        run.log({p.stem:wandb.Image(str(p)) for p in out.glob('*.png')})
        artifact=wandb.Artifact('gradformer-cosine-head-lr-report-and-selected-models',type='report')
        artifact.add_dir(str(out));artifact.add_file(str(source));artifact.add_file(__file__)
        artifact.add_file('cosine-head-selected-models.tar')
        run.log_artifact(artifact).wait()
        (out/'report-url.txt').write_text(run.url);print(json.dumps(summary),flush=True);code=0
    finally:run.finish(exit_code=code)

if __name__=='__main__':main()
