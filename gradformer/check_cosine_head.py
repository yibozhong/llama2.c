"""GPU-only independent gradient routing, clipping and recovery checks."""
import argparse
import copy
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from gradformer.check import make_model
from gradformer.cosine_head import backward, clip_gradients, forward_trace
from gradformer.runtime import loss_forward
from gradformer.train_cosine_head import validate_resume


def reference(model, x, y, mode, last_eta, accumulation):
    prior = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    hidden = []
    handle = model.norm.register_forward_pre_hook(lambda module, args: hidden.append(args[0]))
    try:
        ntp, inputs, preds = loss_forward(model, "llama2", x, y, True)
    finally:
        handle.remove()
    if mode == "ntp-only":
        for p, g in zip(model.parameters(), prior):
            p.grad = g
        (ntp / accumulation).backward()
        return ntp.detach(), None
    targets = [None] * len(inputs)
    def capture(i):
        def hook(g):
            targets[i] = g.detach().clone()
        return hook
    handles = [h.register_hook(capture(i)) for i,h in enumerate(inputs)]
    ntp.backward(retain_graph=True)
    for handle in handles:
        handle.remove()
    eta_gradient = model.hidden_step_raw.grad[-1].detach().clone()
    for p, g in zip(model.parameters(), prior):
        p.grad = g
    terms = []
    for pred, target in zip(preds, targets):
        valid = target.float().norm(dim=-1) > 1e-8
        terms.append((1-F.cosine_similarity(pred.float(), target.float(), dim=-1, eps=1e-8))[valid].mean())
    (torch.stack(terms).mean() / accumulation).backward()
    if mode == "cosine-head":
        # Independent second head, explicitly detached at the final hidden.
        detached_loss = model._training_loss(model.norm(hidden[0].detach()), y)
        torch.testing.assert_close(ntp, detached_loss, rtol=0, atol=0)
        (detached_loss / accumulation).backward()
        if last_eta:
            model.hidden_step_raw.grad[-1].add_(eta_gradient / accumulation)
    return ntp.detach(), targets


def routing_checks(quick=False):
    cases = [("cosine-head", True), ("cosine-head", False), ("cosine-only", False), ("ntp-only", True)]
    results = []
    for mode, last_eta in cases:
        for bf16 in ([False] if quick else [False, True]):
            for accumulation in ([1] if quick else [1, 8]):
                ref = make_model("llama2", "adam", "cuda"); actual = copy.deepcopy(ref)
                x = torch.arange(16, device="cuda").reshape(2,8)
                maximum = 0.; events = []
                for iteration in range(2):
                    ref.zero_grad(set_to_none=True); actual.zero_grad(set_to_none=True)
                    for micro in range(accumulation):
                        xx, yy = x+micro+iteration, x+micro+iteration+1
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                            ce, expected_targets = reference(ref, xx, yy, mode, last_eta, accumulation)
                            phase = {"target": True}
                            handles = [p.register_hook(lambda g: events.append("CE reached a block") if phase["target"] else None)
                                       for b in actual.layers for p in b.parameters()] if mode != "ntp-only" else []
                            def observe(trace, targets, extra):
                                assert not events
                                for g, target in zip(targets, expected_targets):
                                    assert not g.requires_grad and g.grad_fn is None
                                    torch.testing.assert_close(g, target, rtol=5e-4, atol=5e-6)
                                phase["target"] = False
                            try:
                                measured, _, _ = backward(actual, xx, yy, mode=mode, train_last_eta=last_eta,
                                                          accumulation=accumulation, observer=observe)
                            finally:
                                for handle in handles: handle.remove()
                        torch.testing.assert_close(ce, measured, rtol=0, atol=0)
                    for (name, a), (_, b) in zip(actual.named_parameters(), ref.named_parameters()):
                        if a.grad is None or b.grad is None:
                            assert a.grad is None and b.grad is None, name
                        else:
                            error = (a.grad-b.grad).abs().max().item(); maximum = max(maximum, error)
                            torch.testing.assert_close(a.grad, b.grad, rtol=1e-3 if bf16 else 5e-4,
                                                       atol=1e-4 if bf16 else 5e-6, msg=name)
                    if mode == "cosine-only":
                        assert actual.norm.weight.grad is None and actual.hidden_step_raw.grad[-1].item() == 0
                    else:
                        assert actual.norm.weight.grad.norm().item() > 0
                        if last_eta: assert actual.hidden_step_raw.grad[-1].abs().item() > 0
                        else: assert actual.hidden_step_raw.grad[-1].item() == 0
                    for model in (ref, actual):
                        with torch.no_grad():
                            for param in model.parameters():
                                if param.grad is not None: param.add_(param.grad, alpha=-.001)
                results.append(dict(mode=mode, last_eta=last_eta, bf16=bf16, accumulation=accumulation,
                                    maximum_gradient_error=maximum, block_ntp_hook_events=len(events)))
    return results


def clipping_check():
    a = make_model("llama2", "adam", "cuda")
    x = torch.arange(16, device="cuda").reshape(2,8)
    backward(a, x, x+1)
    b = copy.deepcopy(a)
    for p,q in zip(a.parameters(), b.parameters()): q.grad = None if p.grad is None else p.grad.clone()
    b.output.weight.grad.mul_(1e6); b.norm.weight.grad.mul_(1e6); b.hidden_step_raw.grad[-1].mul_(1e6)
    clip_gradients(a); clip_gradients(b)
    for left, right in zip(a.layers.parameters(), b.layers.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    torch.testing.assert_close(a.hidden_step_raw.grad[:-1], b.hidden_step_raw.grad[:-1], rtol=0, atol=0)
    return True


def recovery_check():
    a = make_model("llama2", "adam", "cuda"); b = copy.deepcopy(a)
    opta = torch.optim.AdamW(a.parameters(), lr=.0005)
    optb = torch.optim.AdamW(b.parameters(), lr=.0005)
    for step in range(3):
        x = torch.arange(16, device="cuda").reshape(2,8)+step
        for model, opt in ((a,opta),(b,optb)):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): backward(model, x, x+1)
            clip_gradients(model); opt.step()
        if step == 0:
            stream = io.BytesIO(); torch.save(dict(model=b.state_dict(), optimizer=optb.state_dict()), stream)
            stream.seek(0); state = torch.load(stream, weights_only=False)
            b = make_model("llama2", "adam", "cuda"); b.load_state_dict(state["model"])
            optb = torch.optim.AdamW(b.parameters(), lr=.0005); optb.load_state_dict(state["optimizer"])
    for p,q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(opta.state[p][name], optb.state[q][name], rtol=0,atol=0)
    keys = ("mode", "last_eta", "parameter_lr", "hidden_step_size", "seed", "model", "dataset",
            "source", "schedule_steps", "clipping", "batch_tokens", "ce_implementation")
    config = {k:1 for k in keys}; validate_resume(config,config)
    for key in keys:
        changed = {**config,key:2}
        try: validate_resume(config,changed)
        except ValueError: pass
        else: raise AssertionError("Resume accepted change to " + key)
    return True


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    import wandb
    run = wandb.init(project="nanochat", entity="yibozhong657-none", job_type="correctness",
                     group="gradformer-cosine-head-lr-20261028", name="cosine-head-gradient-routing-check",
                     config={**vars(args), "gpu":torch.cuda.get_device_name()})
    code = 1
    try:
        result = dict(routing=routing_checks(args.quick), clipping_isolated=clipping_check(),
                      recovery_exact=recovery_check(), passed=True)
        Path(args.output).write_text(json.dumps(result,indent=2)); run.summary.update(result)
        print(json.dumps(result),flush=True); code = 0
    finally: run.finish(exit_code=code)


if __name__ == "__main__": main()
