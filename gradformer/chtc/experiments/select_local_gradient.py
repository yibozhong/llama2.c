"""Package selected final checkpoint files using archive I/O only on the AP."""
import io
import json
from pathlib import Path
import tarfile

root=Path('/home/yzhong239/gradformer')
rows=json.loads((root/'reports/local-gradient-records.json').read_text());assert len(rows)==6
chosen=[min((r for r in rows if r['alpha'] or r['beta']),key=lambda r:r['final']['val_ce']),
        next(r for r in rows if not r['alpha'] and not r['beta'])]
output=root/'bundles/local-gradient-selected-models.tar';assert not output.exists()
selection=[]
with tarfile.open(output,'w') as dest:
    for row in chosen:
        folder=root/f"results/pretrain-llama-local-gradient-{row['job']}-metadata/local-gradient"
        latest=json.loads((folder/'latest.json').read_text());assert latest['step']==1000
        selection.append({k:row[k] for k in ('job','run_id','label','alpha','beta','lr','final','source')})
        with tarfile.open(root/f"results/pretrain-llama-local-gradient-{row['job']}.tar") as source:
            member=source.getmember(f"results/local-gradient/{latest['directory']}/model.pt")
            target=tarfile.TarInfo(row['label']+'/model.pt');target.size=member.size;dest.addfile(target,source.extractfile(member))
        raw=(folder/'config.json').read_bytes();target=tarfile.TarInfo(row['label']+'/config.json');target.size=len(raw);dest.addfile(target,io.BytesIO(raw))
    raw=json.dumps(selection,indent=2).encode();target=tarfile.TarInfo('selection.json');target.size=len(raw);dest.addfile(target,io.BytesIO(raw))
(root/'reports/local-gradient-selected.json').write_text(json.dumps(selection,indent=2))
print('Selected final models:',[r['run_id'] for r in chosen])
