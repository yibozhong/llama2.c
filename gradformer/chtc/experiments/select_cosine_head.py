"""Archive the three selected final checkpoints using file I/O only."""
import io
import json
from pathlib import Path
import tarfile

root=Path('/home/yzhong239/gradformer')
rows=json.loads((root/'reports/cosine-head-records.json').read_text());assert len(rows)==8
chosen=[min((r for r in rows if r['mode']=='cosine-head'),key=lambda r:r['final']['val_ce']),
        next(r for r in rows if r['mode']=='cosine-only'),next(r for r in rows if r['mode']=='ntp-only')]
output=root/'bundles/cosine-head-selected-models.tar';assert not output.exists()
selection=[]
with tarfile.open(output,'w') as dest:
    for row in chosen:
        folder=root/f"results/pretrain-llama-cosine-head-{row['job']}-metadata/cosine-head"
        latest=json.loads((folder/'latest.json').read_text());assert latest['step']==1000
        selection.append({k:row[k] for k in ['job','run_id','mode','last_eta','eta','lr','label','final','source']})
        with tarfile.open(root/f"results/pretrain-llama-cosine-head-{row['job']}.tar") as source:
            m=source.getmember(f"results/cosine-head/{latest['directory']}/model.pt")
            member=tarfile.TarInfo(row['mode']+'/model.pt');member.size=m.size
            dest.addfile(member,source.extractfile(m))
        raw=(folder/'config.json').read_bytes();member=tarfile.TarInfo(row['mode']+'/config.json');member.size=len(raw)
        dest.addfile(member,io.BytesIO(raw))
    raw=json.dumps(selection,indent=2).encode();member=tarfile.TarInfo('selection.json');member.size=len(raw)
    dest.addfile(member,io.BytesIO(raw))
(root/'reports/cosine-head-selected.json').write_text(json.dumps(selection,indent=2))
print('Selected final model checkpoints:',[r['run_id'] for r in chosen])
