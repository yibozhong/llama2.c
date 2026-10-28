"""Copy selected checkpoint files from archives without loading model tensors."""
import io
import json
from pathlib import Path
import tarfile

root=Path('/home/yzhong239/gradformer')
records=json.loads((root/'reports/sweep-records.json').read_text())
assert len(records)==26
selection=[]
output=root/'bundles/local-lr-selected-models.tar'
assert not output.exists(), 'Keep selected-model input bundles immutable'
with tarfile.open(output,'w') as target:
    for objective in ('local','e2e'):
        row=min([r for r in records if r['objective']==objective],key=lambda r:r['final']['val_ce'])
        metadata=root/f"results/pretrain-llama-local-lr-{row['job']}-metadata/local-ce"
        latest=json.loads((metadata/'latest.json').read_text());assert latest['step']==1000
        selection.append({k:row[k] for k in ('job','run_id','objective','eta','lr','source')})
        selection[-1].update(directory=objective,selection_val_ce=row['final']['val_ce'])
        with tarfile.open(root/f"results/pretrain-llama-local-lr-{row['job']}.tar") as source:
            member=source.getmember(f"results/local-ce/{latest['directory']}/model.pt")
            dest=tarfile.TarInfo(f'selected/{objective}/model.pt');dest.size=member.size
            target.addfile(dest,source.extractfile(member))
        data=(metadata/'config.json').read_bytes()
        dest=tarfile.TarInfo(f'selected/{objective}/config.json');dest.size=len(data);target.addfile(dest,io.BytesIO(data))
    data=json.dumps(selection,indent=2).encode()
    dest=tarfile.TarInfo('selected/selection.json');dest.size=len(data);target.addfile(dest,io.BytesIO(data))
(root/'reports/local-lr-selected.json').write_text(json.dumps(selection,indent=2))
print(json.dumps(selection,indent=2))
