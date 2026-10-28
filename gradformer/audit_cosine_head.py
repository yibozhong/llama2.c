"""Audit final checkpoints inside an allocated compute job, then log to W&B."""
import argparse
import json
import math
from pathlib import Path
import statistics

import torch


def main():
    p = argparse.ArgumentParser(); p.add_argument("--folder",required=True); p.add_argument("--output",required=True)
    args = p.parse_args(); folder = Path(args.folder)
    config = json.loads((folder/"config.json").read_text())
    latest = json.loads((folder/"latest.json").read_text()); assert latest["step"] == 1000
    model = torch.load(folder/latest["directory"]/"model.pt",map_location="cpu",weights_only=True)
    state = torch.load(folder/latest["directory"]/"rank-0.pt",map_location="cpu",weights_only=False)
    assert state["step"] == 1000
    assert all(torch.isfinite(v).all() for v in model.values() if v.is_floating_point())
    for entry in state["optimizer"]["state"].values():
        assert all(torch.isfinite(v).all() for v in entry.values() if isinstance(v,torch.Tensor))
    rows = [json.loads(line) for line in (folder/"metrics.jsonl").read_text().splitlines()]
    assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
    for r in rows:
        if "parameter_lr" in r:
            step=r["step"]-1; mult=(step+1)/100 if step<100 else .5*(1+math.cos(math.pi*(step-100)/6004))
            assert math.isclose(r["parameter_lr"],config["parameter_lr"]*mult,rel_tol=1e-12)
    final = [r for r in rows if "val_ce" in r][-1]; assert final["step"]==1000 and final["tokens"]==32768000
    norm_changed = not torch.equal(model["norm.weight"],torch.ones_like(model["norm.weight"]))
    initial = model["hidden_step_raw"].new_tensor(math.log(math.expm1(config["hidden_step_size"])))
    eta_changed = not torch.equal(model["hidden_step_raw"][-1],initial)
    assert norm_changed == (config["mode"] != "cosine-only")
    assert eta_changed == (config["last_eta"] == "ntp")
    audit = dict(passed=True,step=1000,finite_model=True,finite_optimizer=True,
                 norm_changed=norm_changed,last_eta_changed=eta_changed,lr_schedule_verified=True)
    result = dict(config=config,run_id=(folder/"wandb-id.txt").read_text().strip(),
                  final_validation=final,checkpoint_audit=audit,
                  median_step_seconds=statistics.median(r["step_seconds"] for r in rows if "step_seconds" in r))
    Path(args.output).write_text(json.dumps(result,indent=2))
    import wandb
    run = wandb.Api().run("yibozhong657-none/nanochat/"+result["run_id"])
    run.summary["cosine_head_checkpoint_audit"]=audit;run.summary.update()
    print(json.dumps(result),flush=True)


if __name__ == "__main__": main()
