"""Matched ordinary Transformer / conditional-energy pretraining pilot."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from gradformer.energy import make_model, forward_hidden, enable_compile
from gradformer.local_ce import readout_loss
from gradformer.runtime import optimizer_for
from gradformer.train import atomic_save, batch
from gradformer.train_local import configure_parameter_lr


def main():
    begin_wall = time.perf_counter()
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True);p.add_argument("--output",required=True)
    p.add_argument("--kind",choices=["baseline","sgd","adam"],required=True)
    p.add_argument("--parameter-lr",type=float,default=.0005)
    p.add_argument("--hidden-step-size",type=float,default=.01)
    p.add_argument("--stop-after-step",type=int,default=1000)
    p.add_argument("--seed",type=int,default=1337);p.add_argument("--label",default="")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if not all(math.isfinite(v) and v>0 for v in (args.parameter_lr,args.hidden_step_size)):
        p.error("LR and eta must be positive finite")
    if not 0<args.stop_after_step<=6104:p.error("Invalid endpoint")
    torch.cuda.set_device(0);torch.manual_seed(args.seed)
    torch._dynamo.config.recompile_limit=64
    rng=np.random.default_rng(args.seed);device=torch.device("cuda")
    data=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data/"manifest.json").read_text())
    train=np.memmap(data/"train.bin",dtype=np.uint16,mode="r")
    val=np.memmap(data/"val.bin",dtype=np.uint16,mode="r")
    model=make_model(args.kind,vocab=manifest["vocab_size"],step_size=args.hidden_step_size)
    opt=optimizer_for(model,"llama2",32768);configure_parameter_lr(opt,args.parameter_lr)
    config={**vars(args),"source":json.loads(Path("gradformer/source.json").read_text()),
            "dataset":manifest,"model":asdict(model.params),"gpu":torch.cuda.get_device_name(),
            "torch":torch.__version__,"parameters":sum(p.numel() for p in model.parameters()),
            "batch_tokens":32768,"batch":16,"sequence":256,"accumulation":8,
            "schedule_steps":6104,"warmup_steps":100,"objective":"final NTP CE only; coefficient 1",
            "clipping":"global norm 1","ce_implementation":"CCE full vocabulary; no filtering",
            "compile":"per block fullgraph; exact explicit energy derivative",
            "energy":"sum_j a_j SiLU(Wz RMS(z_t) + Wc c_t); c=RMS(causal_attention(RMS(h)))",
            "inner_derivative":"partial in independent z, at z=h; context fixed only in inner derivative",
            "outer_derivative":"exact e2e through z and context; no detach, no gradient target, no local CE",
            "adam_state":"reset each forward; per-token moments across depth; b1=.9 b2=.999 eps=.01"}
    latest_path=out/"latest.json";start=0;elapsed=0.;prior_wall=0.;best=float("inf")
    if args.resume:
        old=json.loads((out/"config.json").read_text())
        for k in ("kind","parameter_lr","hidden_step_size","seed","source","dataset","model","clipping","compile"):
            if old[k]!=config[k]:raise ValueError("Resume mismatch: "+k)
        folder=out/json.loads(latest_path.read_text())["directory"]
        model.load_state_dict(torch.load(folder/"model.pt",map_location=device,weights_only=True))
        state=torch.load(folder/"state.pt",map_location=device,weights_only=False)
        opt.load_state_dict(state["optimizer"]);rng.bit_generator.state=state["data_rng"]
        torch.set_rng_state(state["torch_rng"].cpu());torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        start=state["step"];elapsed=state["training_seconds"];best=state["best_val"]
        prior_wall=state["wall_seconds"]
    elif latest_path.exists():raise ValueError("Existing checkpoint requires --resume")
    if start>=args.stop_after_step:raise ValueError("Endpoint must exceed saved step")
    import wandb
    run_id=(out/"wandb-id.txt").read_text().strip() if args.resume else wandb.util.generate_id()
    (out/"wandb-id.txt").write_text(run_id);(out/"config.json").write_text(json.dumps(config,indent=2))
    run=wandb.init(project="nanochat",entity="yibozhong657-none",id=run_id,resume="allow",
                   name=f"layer-energy-{args.kind}-eta{args.hidden_step_size:g}-lr{args.parameter_lr:g}-{args.label}",
                   group="gradformer-layer-energy-20261028",job_type="pretrain",config=config)
    run.define_metric("step");run.define_metric("*",step_metric="step")
    handle=(out/"metrics.jsonl").open("a")
    def emit(row):
        if any(not math.isfinite(v) for v in row.values() if isinstance(v,(float,int))):
            raise RuntimeError("Nonfinite metric")
        handle.write(json.dumps(row)+"\n");handle.flush();run.log(row);print(json.dumps(row),flush=True)

    def evaluate(step):
        model.eval();vrng=np.random.default_rng(987654);losses=[];stats=[]
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            for i in range(32):
                x,y=batch(val,vrng,16,256,device)
                h,s=forward_hidden(model,x,diagnostics=i<4)
                losses.append(readout_loss(model,h,y,"cce").detach())
                if s is not None:stats.append(s)
        value=torch.stack(losses).mean().item()
        row=dict(step=step,tokens=step*32768,val_ce=value,training_seconds=elapsed,
                 wall_seconds=prior_wall+time.perf_counter()-begin_wall)
        for i,values in enumerate(torch.stack(stats).mean(0).tolist()):
            row.update({f"hidden/layer_{i}/{k}":v for k,v in zip(("rms","energy_grad_rms","update_rms","update_ratio"),values)})
        emit(row);model.train();return value

    def save(step):
        old=json.loads(latest_path.read_text())["directory"] if latest_path.exists() else "checkpoint-1"
        folder=out/("checkpoint-1" if old=="checkpoint-0" else "checkpoint-0");folder.mkdir(exist_ok=True)
        atomic_save(model.state_dict(),folder/"model.pt")
        atomic_save(dict(optimizer=opt.state_dict(),data_rng=rng.bit_generator.state,
                         torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                         step=step,training_seconds=elapsed,best_val=best,
                         wall_seconds=prior_wall+time.perf_counter()-begin_wall),folder/"state.pt")
        tmp=out/"latest.json.tmp";tmp.write_text(json.dumps(dict(directory=folder.name,step=step)));tmp.replace(latest_path)

    code=1
    try:
        enable_compile(model)
        # Compile and populate kernels without changing parameters or data RNG.
        x,y=batch(val,np.random.default_rng(456789),16,256,device)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            h,_=forward_hidden(model,x);warmup=readout_loss(model,h,y,"cce")
        warmup.backward();opt.zero_grad(set_to_none=True);del h,warmup
        best=min(best,evaluate(start));torch.cuda.reset_peak_memory_stats()
        for step in range(start,args.stop_after_step):
            mult=(step+1)/100 if step<100 else .5*(1+math.cos(math.pi*(step-100)/6004))
            for g in opt.param_groups:g["lr"]=g["initial_lr"]*mult
            opt.zero_grad(set_to_none=True);torch.cuda.synchronize();begin=time.perf_counter();ces=[]
            for _ in range(8):
                x,y=batch(train,rng,16,256,device)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    h,_=forward_hidden(model,x);ce=readout_loss(model,h,y,"cce")
                (ce/8).backward();ces.append(ce.detach())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();torch.cuda.synchronize();duration=time.perf_counter()-begin;elapsed+=duration
            if step==start or (step+1)%10==0:
                row=dict(step=step+1,tokens=(step+1)*32768,train_ce=torch.stack(ces).mean().item(),
                         parameter_lr=opt.param_groups[0]["lr"],parameter_grad_norm=norm.item(),
                         training_seconds=elapsed,step_seconds=duration,
                         peak_memory_gb=torch.cuda.max_memory_allocated()/2**30,
                         wall_seconds=prior_wall+time.perf_counter()-begin_wall)
                if args.kind!="baseline":
                    row.update({f"hidden_eta/layer_{i}":v for i,v in enumerate(torch.nn.functional.softplus(model.hidden_step_raw).detach().tolist())})
                emit(row)
            if (step+1)%250==0 or step+1==args.stop_after_step:best=min(best,evaluate(step+1))
            if (step+1)%1000==0 or step+1==args.stop_after_step:save(step+1)
        # Check saved tensors on the allocated GPU; the full resume equivalence
        # and dense-vs-CCE gradient checks live in check_energy.
        folder=out/json.loads(latest_path.read_text())["directory"]
        saved=torch.load(folder/"model.pt",map_location=device,weights_only=True)
        for name,value in model.state_dict().items():
            torch.testing.assert_close(saved[name],value,rtol=0,atol=0)
            assert torch.isfinite(value).all(),name
        del saved
        summary=dict(last_step=args.stop_after_step,best_val_ce=best,training_seconds=elapsed,
                     wall_seconds=prior_wall+time.perf_counter()-begin_wall,
                     pilot_complete=args.stop_after_step==1000,checkpoint_exact=True,run_id=run.id,config=config)
        (out/"summary.json").write_text(json.dumps(summary,indent=2));run.summary.update(summary);code=0
    finally:
        handle.close();run.finish(exit_code=code)


if __name__=="__main__":main()
