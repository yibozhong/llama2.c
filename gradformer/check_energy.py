"""GPU correctness gate and same-device timing for conditional energy layers."""
import argparse
import copy
import io
import json
from pathlib import Path
import time

import torch

from gradformer.energy import ConditionalEnergy, make_model, forward_hidden, enable_compile
from gradformer.local_ce import readout_loss
from gradformer.runtime import optimizer_for


def relative(a, b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-12)).item()


def potential_check():
    torch.manual_seed(7)
    p = ConditionalEnergy(8, 24, 1e-5).cuda().double()
    z = torch.randn(2, 3, 8, device="cuda", dtype=torch.float64, requires_grad=True)
    context = torch.randn_like(z, requires_grad=True)
    actual, reference = p.gradient(z, context), p.reference_gradient(z, context)
    torch.testing.assert_close(actual, reference, rtol=2e-12, atol=2e-12)
    variables = [z, context, *p.parameters()]
    a = torch.autograd.grad(actual.square().sum(), variables)
    b = torch.autograd.grad(reference.square().sum(), variables)
    for aa, bb in zip(a, b):
        torch.testing.assert_close(aa, bb, rtol=2e-11, atol=2e-11)
    direction = torch.randn_like(p.z_proj.weight)
    old = p.z_proj.weight.detach().clone()
    def probe(sign):
        with torch.no_grad():
            p.z_proj.weight.copy_(old + sign*1e-5*direction)
            return p.gradient(z, context).square().sum().item()
    fd = (probe(1)-probe(-1))/2e-5
    with torch.no_grad():
        p.z_proj.weight.copy_(old)
    # Parameter order is amplitude, norm_weight, z_proj.weight, c_proj.weight.
    ad = (a[4]*direction).sum().item()
    assert abs(fd-ad) < 1e-7 * max(1, abs(ad)), (fd, ad)
    return dict(gradient_relative=relative(actual, reference),
                outer_max_relative=max(relative(x, y) for x, y in zip(a, b)),
                outer_finite_difference_error=abs(fd-ad))


def tiny(kind):
    torch.manual_seed(31)
    return make_model(kind, depth=2, width=64, heads=2, vocab=256, sequence=16, step_size=.01)


def loss_and_grads(model, x, y, amp, implementation="dense"):
    model.train(); model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        h, _ = forward_hidden(model, x)
        loss = readout_loss(model, h, y, implementation)
    loss.backward()
    gradients = {}
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.norm().item() > 0, name
        gradients[name] = p.grad.detach().clone()
    return h.detach(), loss.detach(), gradients


def compare(a, b, tolerance):
    err = max(relative(a[k], b[k]) for k in a)
    assert err < tolerance, (err, sorted(((relative(a[k],b[k]), k) for k in a), reverse=True)[:4])
    return err


def full_graph_checks():
    x = torch.arange(32, device="cuda").reshape(2,16)
    y = (x+1) % 256
    results = {}
    for kind in ("sgd", "adam"):
        for amp in (False, True):
            actual = tiny(kind); reference = copy.deepcopy(actual)
            for block in reference.layers:
                block.reference = True
            h, loss, ga = loss_and_grads(actual, x, y, amp)
            hr, lr, gr = loss_and_grads(reference, x, y, amp)
            torch.testing.assert_close(h, hr, rtol=.03 if amp else 1e-4, atol=2e-4 if amp else 1e-7)
            torch.testing.assert_close(loss, lr, rtol=.003 if amp else 1e-6, atol=1e-6)
            results[f"{kind}_{amp}_autograd_relative"] = compare(ga, gr, .10 if amp else .001)
            actual.eval()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                with torch.no_grad():
                    he, _ = forward_hidden(actual, x)
                with torch.inference_mode():
                    hi, _ = forward_hidden(actual, x)
                    changed = x.clone(); changed[:, 8:] += 50
                    hc, _ = forward_hidden(actual, changed)
                    repeated, _ = forward_hidden(actual, x.repeat(2,1))
            torch.testing.assert_close(he, hi, rtol=0, atol=0)
            torch.testing.assert_close(hi[:,:8], hc[:,:8], rtol=0, atol=0)
            torch.testing.assert_close(hi, repeated[:2], rtol=1e-5, atol=1e-7)
            results[f"{kind}_{amp}_causal_inference"] = True
        actual = tiny(kind); compiled = copy.deepcopy(actual); enable_compile(compiled)
        h, l, ga = loss_and_grads(actual, x, y, True)
        hc, lc, gc = loss_and_grads(compiled, x, y, True)
        torch.testing.assert_close(l, lc, rtol=.003, atol=1e-5)
        results[f"{kind}_compiled_relative"] = compare(gc, ga, .10)
        _, lcce, gcc = loss_and_grads(actual, x, y, True, "cce")
        torch.testing.assert_close(l, lcce, rtol=.003, atol=1e-5)
        results[f"{kind}_cce_relative"] = compare(gcc, ga, .06)
        # Restoring parameters, AdamW states and the next step must be exact.
        model = tiny(kind); opt = optimizer_for(model, "llama2", 32768)
        loss_and_grads(model,x,y,False); opt.step()
        buf = io.BytesIO(); torch.save((model.state_dict(),opt.state_dict()),buf); buf.seek(0)
        restored = tiny(kind); ropt = optimizer_for(restored,"llama2",32768)
        ms,os = torch.load(buf,weights_only=False); restored.load_state_dict(ms); ropt.load_state_dict(os)
        loss_and_grads(model,x,y,False); opt.step()
        loss_and_grads(restored,x,y,False); ropt.step()
        for a,b in zip(model.parameters(),restored.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        results[f"{kind}_resume_exact"] = True
    return results


def benchmark():
    results = {}
    for kind in ("baseline", "sgd", "adam"):
        torch.manual_seed(1337)
        model = make_model(kind); enable_compile(model)
        opt = optimizer_for(model,"llama2",32768)
        x = torch.randint(32000,(16,256),device="cuda"); y = torch.randint(32000,x.shape,device="cuda")
        times = []
        for iteration in range(6):
            opt.zero_grad(set_to_none=True)
            if iteration == 2:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); start = time.perf_counter()
            for _ in range(8):
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    h,_ = forward_hidden(model,x)
                    loss = readout_loss(model,h,y,"cce")/8
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step(); torch.cuda.synchronize()
            if iteration >= 2:
                times.append(time.perf_counter()-start)
        results[kind] = dict(step_seconds=sorted(times)[len(times)//2], timed_steps=times,
                             peak_memory_gb=torch.cuda.max_memory_allocated()/2**30,
                             parameters=sum(p.numel() for p in model.parameters()))
        del model,opt,h,loss; torch.cuda.empty_cache()
    return results


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",required=True); args=p.parse_args()
    torch.cuda.set_device(0)
    torch._dynamo.config.recompile_limit=64
    import wandb
    run=wandb.init(project="nanochat",entity="yibozhong657-none",group="gradformer-layer-energy-20261028",
                   job_type="correctness-benchmark",name="layer-energy-math-causal-compile-check",
                   config=dict(source=json.loads(Path('gradformer/source.json').read_text()),
                               gpu=torch.cuda.get_device_name(),torch=torch.__version__))
    result={"gpu":torch.cuda.get_device_name(),"run_id":run.id};code=1
    try:
        for name,fn in (("potential",potential_check),("full_graph",full_graph_checks),("benchmark",benchmark)):
            result[name]=fn();print(json.dumps(dict(check=name,result=result[name])),flush=True)
            run.summary[name]=result[name]
        result["passed"]=True;code=0
    finally:
        Path(args.output).write_text(json.dumps(result,indent=2));run.finish(exit_code=code)


if __name__=="__main__":main()
