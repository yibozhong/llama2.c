"""Render a paired LR sweep from verified metadata on a CHTC CPU job."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import wandb


def main():
    output=Path('results'); output.mkdir(exist_ok=True)
    data=json.loads(Path('sweep-records.json').read_text())
    run=wandb.init(project='nanochat',entity='yibozhong657-none',job_type='report',
        name='gradformer-local-sgd-lr-sweep-report',group='gradformer-local-sgd-lr-sweep-20261028',
        config={'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'inputs_sha256':hashlib.sha256(Path('sweep-records.json').read_bytes()).hexdigest(),
                'training_runs':len(data),'seed':1337,'tokens_per_run':32768000})
    code=1
    try:
        api=wandb.Api(); verified=[]
        for row in data:
            remote=api.run('yibozhong657-none/nanochat/'+row['run_id'])
            assert remote.state=='finished'
            assert abs(remote.summary['val_ce']-row['final']['val_ce'])<1e-8
            assert remote.summary['local_checkpoint_audit.passed']
            verified.append({'run_id':row['run_id'],'val_ce':remote.summary['val_ce'],'state':remote.state})
        (output/'verified.json').write_text(json.dumps(verified,indent=2))
        etas=[.001,.01,.1]; lrs=[.00005,.0005,.005]
        matrices={objective:np.full((3,3),np.nan) for objective in ('local','e2e')}
        for row in data:
            if row['eta'] in etas and row['lr'] in lrs:
                matrices[row['objective']][etas.index(row['eta']),lrs.index(row['lr'])]=row['final']['val_ce']
        assert all(np.isfinite(m).all() for m in matrices.values())
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
        lo=min(m.min() for m in matrices.values());hi=max(m.max() for m in matrices.values())
        gap=matrices['local']-matrices['e2e'];bound=max(abs(gap).max(),.001)
        for ax,title,m in zip(axes,['Local CE','Connected final CE','Local minus connected CE'],[matrices['local'],matrices['e2e'],gap]):
            im=ax.imshow(m,cmap='coolwarm' if ax==axes[2] else 'viridis_r',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound) if ax==axes[2] else None,vmin=None if ax==axes[2] else lo,vmax=None if ax==axes[2] else hi)
            for i in range(3):
                for j in range(3): ax.text(j,i,f'{m[i,j]:.3f}',ha='center',va='center',color='black',bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=2))
            ax.set(xticks=range(3),xticklabels=[f'{v:g}' for v in lrs],yticks=range(3),yticklabels=[f'{v:g}' for v in etas],xlabel='Peak parameter AdamW learning rate',ylabel='Initial learnable hidden step size',title=title)
            fig.colorbar(im,ax=ax,shrink=.8,label='CE (nats/token)')
        fig.suptitle('Paired 3×3 LR sweep · 1000 steps / 32.77M tokens · seed 1337')
        figures=[]
        def save(fig,name):
            for ext in ('png','pdf','svg'): fig.savefig(output/f'{name}.{ext}',dpi=180,bbox_inches='tight')
            figures.append(output/f'{name}.png');plt.close(fig)
        save(fig,'paired-lr-grid')
        best={obj:min([r for r in data if r['objective']==obj],key=lambda r:r['final']['val_ce']) for obj in ('local','e2e')}
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for obj,color in (('local','#d97706'),('e2e','#2563eb')):
            for row in [r for r in data if r['objective']==obj]:
                vals=row['validation'];is_best=row['run_id']==best[obj]['run_id']
                for ax in axes:
                    ax.plot([r['step'] for r in vals],[r['val_ce'] for r in vals],color=color,alpha=1 if is_best else .18,lw=2.2 if is_best else 1,label=f"Best {obj}: eta={row['eta']:g}, lr={row['lr']:g}" if is_best else None)
        for ax in axes: ax.set(xlabel='Training step',ylabel='Validation CE');ax.grid(alpha=.2)
        axes[0].set_title('All candidates; best final-CE runs highlighted');axes[0].legend(fontsize=8)
        axes[1].set(xlim=(250,1000),ylim=(max(0,min(r['final']['val_ce'] for r in data)-.1),max(v['val_ce'] for r in best.values() for v in r['validation'] if v['step']>=250)+.2),title='Promising region (zoom; poor runs may be off-scale)')
        save(fig,'lr-sweep-curves')
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for obj,color in (('local','#d97706'),('e2e','#2563eb')):
            row=best[obj]
            ax.plot(range(1,7),[row['final'][f'val_local_ce/layer_{i}'] for i in range(6)],marker='o',color=color,label=f"Best {obj}: eta={row['eta']:g}, lr={row['lr']:g}")
        ax.set(xlabel='Layer (shared final head evaluation)',ylabel='Validation CE',title='Per-layer task effect at 1000 steps');ax.grid(alpha=.2);ax.legend(fontsize=8)
        save(fig,'lr-sweep-best-layers')
        run.log({p.stem:wandb.Image(str(p)) for p in figures})
        run.summary.update({'verified_training_runs':len(data),'best_local_ce':best['local']['final']['val_ce'],'best_e2e_ce':best['e2e']['final']['val_ce'],'best_local_run':best['local']['run_id'],'best_e2e_run':best['e2e']['run_id']})
        artifact=wandb.Artifact('gradformer-local-lr-sweep-report',type='report')
        artifact.add_dir(str(output));artifact.add_file('sweep-records.json');artifact.add_file(__file__)
        run.log_artifact(artifact).wait();(output/'report-url.txt').write_text(run.url)
        code=0
    finally: run.finish(exit_code=code)

if __name__=='__main__':main()
