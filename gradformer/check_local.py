"""GPU checks: CCE full-vocabulary gradients, local routing, clipping, resume."""
import argparse
import copy
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from gradformer.check_cosine import assert_tree_equal
from gradformer.core import hidden_step
from gradformer.local_ce import (amp_context, local_losses, local_backward,
                                 clip_local_gradients, validation_metrics)
from gradformer.runtime import construct


def tiny():
    torch.manual_seed(123)
    return construct("llama2", "adam", depth=3, width=64, heads=2,
                     vocab=256, sequence=16, step_size=.01, device="cuda")


def relative_error(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def fused_checks():
    from cut_cross_entropy import linear_cross_entropy
    result = {}
    for n, d, v in ((32, 64, 256), (4096, 288, 32000)):
        torch.manual_seed(42)
        e = torch.randn(n, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        c = (torch.randn(v, d, device="cuda", dtype=torch.bfloat16) * .02).requires_grad_()
        y = torch.randint(v, (n,), device="cuda"); y[0] = -1
        reference = F.cross_entropy((e @ c.T).float(), y, ignore_index=-1)
        expected = torch.autograd.grad(reference, (e, c))
        actual = linear_cross_entropy(e, c, y, ignore_index=-1, shift=0, impl="cce",
                                      filter_eps=None, filter_e_grad=False, filter_c_grad=False,
                                      accum_e_fp32=True, accum_c_fp32=True)
        gradients = torch.autograd.grad(actual, (e, c))
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=2e-5)
        errors = [relative_error(a, b) for a, b in zip(gradients, expected)]
        assert max(errors) < .015, errors
        assert torch.count_nonzero(gradients[0][0]) == 0
        result[f"tokens_{n}_width_{d}_vocab_{v}"] = {
            "loss_abs_error": (actual - reference).abs().item(),
            "hidden_gradient_relative_l2_error": errors[0], "head_gradient_relative_l2_error": errors[1],
            "all_vocab_no_filter": True, "ignored_token_gradient_zero": True}
    return result


def routing_checks():
    x = torch.arange(32, device="cuda").reshape(2, 16)
    model = tiny()
    terms = list(local_losses(model, x, x + 1, "dense", amp=False))
    parameters = list(model.parameters())
    by_id = {id(p): i for i, p in enumerate(parameters)}
    for i, loss in terms:
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        for j, block in enumerate(model.layers):
            these = [grads[by_id[id(p)]] for p in block.parameters()]
            if j == i:
                assert any(g is not None and g.abs().sum() > 0 for g in these)
            else:
                assert all(g is None for g in these), (i, j)
        eta = grads[by_id[id(model.hidden_step_raw)]]
        assert eta is not None and eta[i].abs() > 0
        assert torch.count_nonzero(torch.cat((eta[:i], eta[i + 1:]))) == 0
        assert grads[by_id[id(model.output.weight)]] is not None
        assert grads[by_id[id(model.norm.weight)]] is not None
    # Negative control: detaching h ALONE still leaks through Adam moments.
    h, state = model.tok_embeddings(x), None
    cos, sin = model.freqs_cos[:16], model.freqs_sin[:16]
    for i, block in enumerate(model.layers):
        if i:
            h = h.detach()
        pred = -block(h, cos, sin, return_delta=True)
        h, state = hidden_step(h, pred, model.hidden_step_raw[i], state, i, model.params)
    loss = F.cross_entropy(model.output(model.norm(h)).reshape(-1, 256), (x + 1).reshape(-1))
    leaked = torch.autograd.grad(loss, tuple(model.layers[0].parameters()), allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in leaked)
    return {"every_block_receives_only_own_ce": True, "eta_receives_only_own_ce": True,
            "shared_head_and_norm_receive_all_losses": True, "moments_leak_negative_control_detected": True}


def reference_checks():
    result = {}
    x = torch.arange(32, device="cuda").reshape(2, 16)
    for accumulation in (1, 3, 8):
        reference, actual = tiny(), tiny()
        reference_terms = []
        for micro in range(accumulation):
            xx, yy = x + micro, x + micro + 1
            local_backward(actual, xx, yy, accumulation=accumulation, implementation="dense", amp=False)
            h, state = reference.tok_embeddings(xx), None
            for i, block in enumerate(reference.layers):
                if i:
                    h = h.detach(); state = tuple(t.detach() for t in state)
                pred = -block(h, reference.freqs_cos[:16], reference.freqs_sin[:16], return_delta=True)
                h, state = hidden_step(h, pred, reference.hidden_step_raw[i], state, i, reference.params)
                logits = reference.output(reference.norm(h))
                reference_terms.append(F.cross_entropy(logits.reshape(-1, 256), yy.reshape(-1)))
        torch.stack(reference_terms).mean().backward()
        largest = 0.
        for (name, a), (_, b) in zip(reference.named_parameters(), actual.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6, msg=name)
            largest = max(largest, (a.grad - b.grad).abs().max().item())
        result[f"accumulation_{accumulation}"] = largest
    return result


def clipping_checks():
    a, b = tiny(), tiny()
    x = torch.arange(32, device="cuda").reshape(2, 16)
    local_backward(a, x, x + 1, implementation="dense", amp=False)
    for p, q in zip(a.parameters(), b.parameters()):
        q.grad = p.grad.clone()
    # Change another block and shared-head gradients drastically.
    for p in b.layers[-1].parameters():
        p.grad.mul_(1e6)
    b.hidden_step_raw.grad[-1].mul_(1e6)
    b.output.weight.grad.mul_(1e6)
    clip_local_gradients(a); clip_local_gradients(b)
    for p, q in zip(a.layers[0].parameters(), b.layers[0].parameters()):
        assert_tree_equal(p.grad, q.grad)
    assert_tree_equal(a.hidden_step_raw.grad[0], b.hidden_step_raw.grad[0])
    return {"other_block_and_head_do_not_change_local_clipping": True}


def fused_model_and_resume():
    x = torch.arange(32, device="cuda").reshape(2, 16)
    dense, fused = tiny(), tiny()
    expected = local_backward(dense, x, x + 1, implementation="dense")
    actual = local_backward(fused, x, x + 1)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    errors = []
    for (name, p), (_, q) in zip(dense.named_parameters(), fused.named_parameters()):
        error = relative_error(q.grad, p.grad)
        assert error < .04, (name, error)
        errors.append(error)
    # Compare the graph boundary change in identical precision/grad modes.
    # Also capture hidden states, so this does not confound a CE kernel's
    # numerical reduction with a change to the hidden update itself.
    forward_errors = {}
    for enabled in (True, False):
        captured, logits = [], []
        hook = fused.norm.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
        head_hook = fused.output.register_forward_hook(lambda module, inputs, output: logits.append(output.detach().clone()))
        try:
            with torch.set_grad_enabled(enabled):
                with amp_context(fused):
                    original = fused(x, x + 1, return_loss=True)
                native_hidden = captured[-1]
                native_logits = logits[-1]
                local = list(local_losses(fused, x, x + 1, "dense"))[-1][1]
                local_hidden = captured[-1]
                local_logits = logits[-1]
                fp32_reference = F.cross_entropy(native_logits.float().reshape(-1, 256), (x + 1).reshape(-1))
            forward_errors[f"grad_enabled_{enabled}"] = {
                "hidden_max_error": (native_hidden - local_hidden).abs().max().item(),
                "logits_max_error": (native_logits - local_logits).abs().max().item(),
                "fp32_ce_abs_error": (fp32_reference - local).abs().item(),
                "native_ce_abs_difference": (original - local).abs().item(),
                "native_ce_dtype": str(original.dtype), "native_logits_dtype": str(native_logits.dtype),
                "native_ce": original.item(), "local_fp32_ce": local.item()}
            print(json.dumps({"forward_equivalence": forward_errors}), flush=True)
            torch.testing.assert_close(native_hidden, local_hidden, rtol=0, atol=0)
            torch.testing.assert_close(native_logits, local_logits, rtol=0, atol=0)
            # Local CCE is compared to explicitly fp32 CE throughout. Record
            # the native loss separately instead of conflating its reduction
            # precision with preservation of the forward hidden/logit values.
            torch.testing.assert_close(fp32_reference, local, rtol=0, atol=0)
        finally:
            hook.remove(); head_hook.remove()
    # Local fused/dense CE agreement was checked above in the training mode.
    model = tiny(); optimizer = model.configure_optimizers(.1, 5e-4, (.9, .95), "cuda")
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        local_backward(model, x + step, x + step + 1)
        clip_local_gradients(model); optimizer.step()
    stream = io.BytesIO()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), stream)
    stream.seek(0); state = torch.load(stream, weights_only=False)
    resumed = tiny(); resumed.load_state_dict(state["model"])
    resumed_optimizer = resumed.configure_optimizers(.1, 5e-4, (.9, .95), "cuda")
    resumed_optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    assert_tree_equal(model.state_dict(), resumed.state_dict())
    assert_tree_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
    # CCE uses parallel reductions: resumed trajectories are checked within
    # numerical tolerance, not claimed bitwise deterministic across launches.
    for net, opt in ((model, optimizer), (resumed, resumed_optimizer)):
        opt.zero_grad(set_to_none=True); local_backward(net, x + 4, x + 5)
        clip_local_gradients(net); opt.step()
    for p, q in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-4, atol=2e-6)
    before = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    gradients = [p.grad.clone() for p in model.parameters()]
    rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    validation_metrics(model, [(x, x + 1)])
    assert_tree_equal(model.state_dict(), before[0]); assert_tree_equal(optimizer.state_dict(), before[1])
    assert_tree_equal([p.grad for p in model.parameters()], gradients)
    assert_tree_equal((torch.get_rng_state(), torch.cuda.get_rng_state()), rng)
    return {"max_parameter_gradient_relative_l2_error": max(errors), "forward_matches_original_hidden_adam": True,
            "forward_equivalence": forward_errors,
            "checkpoint_roundtrip_exact": True, "next_step_resume_within_tolerance": True,
            "validation_preserves_training_state_and_rng": True}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); args = p.parse_args()
    import wandb
    run = wandb.init(project="nanochat", entity="yibozhong657-none", job_type="correctness",
                     name="gradformer-local-ce-fused-routing-check", config={**vars(args), "gpu": torch.cuda.get_device_name(),
                     "source": json.loads(Path('gradformer/source.json').read_text())})
    result, exit_code = {}, 1
    try:
        for name, function in (("fused", fused_checks), ("routing", routing_checks),
                               ("reference", reference_checks), ("clipping", clipping_checks),
                               ("model_and_resume", fused_model_and_resume)):
            result[name] = function(); print(json.dumps({"check": name, "result": result[name]}), flush=True)
        result["passed"] = True
        exit_code = 0
    except Exception as error:
        result.update(passed=False, failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        Path(args.output).write_text(json.dumps(result, indent=2)); run.summary.update(result)
        run.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
