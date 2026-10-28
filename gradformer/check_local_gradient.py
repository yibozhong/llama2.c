"""Allocated-GPU independent reference, local routing and recovery checks."""
import argparse
import copy
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from gradformer.check_cosine import assert_tree_equal
from gradformer.check_local import relative_error
from gradformer.local_ce import local_backward, clip_local_gradients, validation_metrics
from gradformer.local_gradient import backward, gradient_validation
from gradformer.runtime import construct, optimizer_for
from gradformer.train_local_gradient import validate_resume


def tiny():
    torch.manual_seed(123)
    return construct("llama2", "sgd", depth=3, width=64, heads=2, vocab=256,
                     sequence=16, step_size=1., device="cuda")


def reference_graphs(model, x, y, alpha, beta, amp):
    h = model.tok_embeddings(x)
    for i, block in enumerate(model.layers):
        if i:
            h = h.detach().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            direction = -block(h, model.freqs_cos[:x.size(1)], model.freqs_sin[:x.size(1)], return_delta=True)
            out = h - F.softplus(model.hidden_step_raw[i]) * direction
            nw = model.norm.weight if i == 2 else model.norm.weight.detach()
            weight = model.output.weight if i == 2 else model.output.weight.detach()
            normalized = (out.float()*torch.rsqrt(out.float().square().mean(-1, keepdim=True)+model.norm.eps)).type_as(out)*nw
            logits = F.linear(normalized, weight)
            ce = F.cross_entropy(logits.float().reshape(-1,256), y.reshape(-1), ignore_index=-1)
        target, = torch.autograd.grad(ce, h, retain_graph=True, create_graph=False)
        assert not target.requires_grad and target.grad_fn is None
        pred, target = direction.float(), target.float()
        valid = y != -1
        mse = (pred-target).square().mean(-1)[valid].mean()
        good = valid & (target.norm(dim=-1) > 1e-8)
        cosine = (pred*target).sum(-1)/(pred.norm(dim=-1).clamp_min(1e-8)*target.norm(dim=-1).clamp_min(1e-8))
        cos_loss = (1-cosine)[good].mean()
        loss = ce+alpha*mse+beta*cos_loss
        stats = torch.stack((mse.detach(), cosine[good].mean().detach(),
                             pred.norm(dim=-1)[valid].mean().detach(), target.norm(dim=-1)[valid].mean(), good.float().sum()/valid.sum()))
        yield i, loss, ce, stats
        h = out


def reference():
    x = torch.arange(32, device="cuda").reshape(2,16)
    results = {}
    for amp in (False, True):
        errors = []
        for alpha,beta in ((0.,0.), (0.,.1), (0.,1.), (1.,1.)):
            for accumulation in (1,8):
                expected,actual=tiny(),tiny()
                for micro in range(accumulation):
                    xx,yy=x+micro,x+micro+1
                    yy=yy.clone();yy[0,-1]=-1
                    measured,stats,_=backward(actual,xx,yy,alpha=alpha,beta=beta,accumulation=accumulation,
                                              implementation="dense",amp=amp,diagnostics=True)
                    for i,loss,ce,ref_stats in reference_graphs(expected,xx,yy,alpha,beta,amp):
                        (loss/(3*accumulation)).backward()
                        torch.testing.assert_close(measured[i,0],ce,rtol=0,atol=0)
                        torch.testing.assert_close(stats[i],ref_stats,rtol=.02 if amp else 2e-5,atol=2e-5)
                for (name,p),(_,q) in zip(expected.named_parameters(),actual.named_parameters()):
                    assert p.grad is not None and q.grad is not None,name
                    if amp:
                        error=relative_error(q.grad,p.grad)
                        assert error<.06,(name,error)
                    else:
                        torch.testing.assert_close(p.grad,q.grad,rtol=5e-4,atol=3e-6,msg=name)
                        error=(p.grad-q.grad).abs().max().item()
                    errors.append(error)
        results['bf16_max_relative_error' if amp else 'fp32_max_absolute_error']=max(errors)
    return results


def routing_and_control():
    x=torch.arange(32,device="cuda").reshape(2,16)
    model=tiny();params=list(model.parameters());index={id(p):i for i,p in enumerate(params)}
    for layer,loss,_,_ in reference_graphs(model,x,x+1,1.,1.,False):
        grads=torch.autograd.grad(loss,params,allow_unused=True)
        for i,block in enumerate(model.layers):
            values=[grads[index[id(p)]] for p in block.parameters()]
            assert any(g is not None and g.abs().sum()>0 for g in values) if i==layer else all(g is None for g in values)
        eta=grads[index[id(model.hidden_step_raw)]]
        assert eta[layer].abs()>0 and (eta!=0).sum()==1
        assert (grads[index[id(model.norm.weight)]] is not None)==(layer==2)
        assert (grads[index[id(model.output.weight)]] is not None)==(layer in (0,2))
    controls=[]
    for amp in (False,True):
        old,new=tiny(),tiny()
        local_backward(old,x,x+1,implementation="dense",amp=amp,readout_training="final")
        backward(new,x,x+1,alpha=0.,beta=0.,implementation="dense",amp=amp,diagnostics=True)
        for p,q in zip(old.parameters(),new.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        controls.append(True)
    a,b=tiny(),tiny()
    for model,diag in ((a,False),(b,True)):
        backward(model,x,x+1,alpha=1.,beta=1.,implementation="dense",amp=False,diagnostics=diag)
    for p,q in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    old=tiny();local_backward(old,x,x+1,implementation='dense',amp=False,readout_training='final')
    torch.testing.assert_close(a.hidden_step_raw.grad,old.hidden_step_raw.grad,rtol=0,atol=0)
    torch.testing.assert_close(a.norm.weight.grad,old.norm.weight.grad,rtol=0,atol=0)
    return dict(strict_local_block_and_eta=True,final_only_readout=True,tied_input_path_first_layer=True,
                zero_aux_matches_old_local_exactly=all(controls),diagnostics_do_not_change_gradients=True,
                auxiliary_does_not_update_local_eta_or_final_norm=True)


def fused_and_resume():
    x=torch.arange(32,device="cuda").reshape(2,16)
    errors=[]
    for alpha,beta in ((0.,0.),(0.,1.),(1.,1.)):
        dense,fused=tiny(),tiny()
        a,stats,_=backward(dense,x,x+1,alpha=alpha,beta=beta,implementation="dense")
        b,fstats,_=backward(fused,x,x+1,alpha=alpha,beta=beta)
        torch.testing.assert_close(a,b,rtol=.005,atol=5e-4)
        torch.testing.assert_close(stats,fstats,rtol=.04,atol=5e-4)
        for (name,p),(_,q) in zip(dense.named_parameters(),fused.named_parameters()):
            error=relative_error(q.grad,p.grad);assert error<.07,(name,error);errors.append(error)
    model=tiny();opt=optimizer_for(model,"llama2",32768)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        backward(model,x+step,x+step+1,alpha=1.,beta=1.)
        clip_local_gradients(model);opt.step()
    stream=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),stream)
    stream.seek(0);saved=torch.load(stream,weights_only=False)
    resumed=tiny();resumed.load_state_dict(saved['model']);ropt=optimizer_for(resumed,"llama2",32768)
    ropt.load_state_dict(copy.deepcopy(saved['optimizer']))
    assert_tree_equal(model.state_dict(),resumed.state_dict());assert_tree_equal(opt.state_dict(),ropt.state_dict())
    for net,optimizer in ((model,opt),(resumed,ropt)):
        optimizer.zero_grad(set_to_none=True);backward(net,x+2,x+3,alpha=1.,beta=1.)
        clip_local_gradients(net);optimizer.step()
    assert_tree_equal(model.state_dict(),resumed.state_dict());assert_tree_equal(opt.state_dict(),ropt.state_dict())
    before=copy.deepcopy(model.state_dict()),[p.grad.clone() for p in model.parameters()],torch.cuda.get_rng_state().clone()
    metrics=gradient_validation(model,[(x,x+1)])
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert_tree_equal(model.state_dict(),before[0]);assert_tree_equal([p.grad for p in model.parameters()],before[1])
    assert_tree_equal(torch.cuda.get_rng_state(),before[2])
    keys=("alpha","beta","parameter_lr","hidden_step_size","seed","model","dataset","source","schedule_steps","clipping","batch_tokens","target","ce_implementation")
    config={key:1 for key in keys};validate_resume(config,config)
    for key in keys:
        try:validate_resume(config,{**config,key:2})
        except ValueError:pass
        else:raise AssertionError('Resume accepted changed '+key)
    return dict(cce_max_relative_parameter_error=max(errors),checkpoint_and_resumed_step_exact=True,
                validation_no_side_effects=True,resume_rejects_changed_semantics=True)


def real_shape():
    torch.manual_seed(4321)
    model=construct("llama2","sgd",depth=6,width=288,heads=6,vocab=32000,sequence=256,step_size=1.,device="cuda")
    x=torch.randint(0,32000,(16,256),device="cuda");y=torch.randint(0,32000,(16,256),device="cuda")
    loss,stats,signals=backward(model,x,y,alpha=1.,beta=1.,accumulation=8,diagnostics=True)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.isfinite(t).all() for t in (loss,stats,signals))
    clip_local_gradients(model)
    return dict(batch=16,sequence=256,width=288,vocabulary=32000,all_finite=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    import wandb
    run=wandb.init(project='nanochat',entity='yibozhong657-none',job_type='correctness',
                    name='local-gradient-reference-routing-recovery',group='gradformer-local-gradient-20261028',
                    config={'source':json.loads(Path('gradformer/source.json').read_text()),'gpu':torch.cuda.get_device_name(),'quick':args.quick})
    result,code={},1
    try:
        checks=(('reference',reference),('routing_and_control',routing_and_control),('fused_and_resume',fused_and_resume),('real_shape',real_shape))
        for name,fn in checks:
            if args.quick and name in ('reference','fused_and_resume'):
                continue
            result[name]=fn();print(json.dumps({'check':name,'result':result[name]}),flush=True)
        result['passed']=True;code=0
    except Exception as error:
        result.update(passed=False,failure=f'{type(error).__name__}: {error}');raise
    finally:
        Path(args.output).write_text(json.dumps(result,indent=2));run.summary.update(result);run.finish(exit_code=code)


if __name__=='__main__':main()
