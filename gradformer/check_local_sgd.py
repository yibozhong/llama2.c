"""SGD local-CE and final-only readout gradient checks on an allocated GPU."""
import argparse
import copy
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from gradformer.check_cosine import assert_tree_equal
from gradformer.check_local import relative_error
from gradformer.local_ce import local_losses, local_backward, clip_local_gradients, readout_loss, validation_metrics
from gradformer.runtime import construct


def tiny():
    torch.manual_seed(123)
    return construct("llama2", "sgd", depth=3, width=64, heads=2, vocab=256,
                     sequence=16, step_size=.01, device="cuda")


def routing():
    x = torch.arange(32, device="cuda").reshape(2, 16)
    model = tiny()
    # An isolated readout must retain hidden gradients, without a head or
    # norm parameter gradient. No embedding input path is present here.
    for implementation in ("dense", "cce"):
        model.zero_grad(set_to_none=True)
        h = torch.randn(2, 16, 64, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = readout_loss(model, h, x + 1, implementation, train_readout=False)
        loss.backward()
        assert h.grad.abs().sum() > 0
        assert model.output.weight.grad is None and model.norm.weight.grad is None
    model.zero_grad(set_to_none=True)
    params = list(model.parameters()); indexes = {id(p): i for i, p in enumerate(params)}
    for layer, loss in local_losses(model, x, x + 1, "dense", amp=False, readout_training="final"):
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for i, block in enumerate(model.layers):
            selected = [grads[indexes[id(p)]] for p in block.parameters()]
            if i == layer:
                assert any(g is not None and g.abs().sum() > 0 for g in selected)
            else:
                assert all(g is None for g in selected)
        eta = grads[indexes[id(model.hidden_step_raw)]]
        assert eta[layer].abs() > 0
        assert torch.count_nonzero(torch.cat((eta[:layer], eta[layer + 1:]))) == 0
        norm_grad = grads[indexes[id(model.norm.weight)]]
        assert (norm_grad is not None) == (layer == 2)
        tied_grad = grads[indexes[id(model.output.weight)]]
        # Layer 0 still trains the tied matrix through the input embedding.
        assert (tied_grad is not None) == (layer in (0, 2))
        if tied_grad is not None:
            assert tied_grad.abs().sum() > 0
    # Removing ONLY the input-embedding path exposes the final-only readout
    # routing for the tied matrix itself.
    hook = model.tok_embeddings.register_forward_hook(lambda module, inputs, output: output.detach())
    try:
        for layer, loss in local_losses(model, x, x + 1, "dense", amp=False, readout_training="final"):
            g, = torch.autograd.grad(loss, (model.output.weight,), allow_unused=True)
            assert (g is not None) == (layer == 2)
    finally:
        hook.remove()
    return dict(local_block_and_eta_routing=True, frozen_readout_preserves_hidden_derivative=True,
                norm_only_final_ce=True, readout_matrix_only_final_ce=True,
                tied_matrix_also_has_first_layer_embedding_path=True)


def reference():
    result = {}
    x = torch.arange(32, device="cuda").reshape(2, 16)
    for accumulation in (1, 3, 8):
        expected, actual = tiny(), tiny()
        losses = []
        for micro in range(accumulation):
            xx, yy = x + micro, x + micro + 1
            local_backward(actual, xx, yy, accumulation=accumulation, implementation="dense",
                           amp=False, readout_training="final")
            h = expected.tok_embeddings(xx)
            for layer, block in enumerate(expected.layers):
                if layer:
                    h = h.detach()
                prediction = -block(h, expected.freqs_cos[:16], expected.freqs_sin[:16], return_delta=True)
                h = h - F.softplus(expected.hidden_step_raw[layer]) * prediction
                nw = expected.norm.weight if layer == 2 else expected.norm.weight.detach()
                weight = expected.output.weight if layer == 2 else expected.output.weight.detach()
                normalized = (h.float() * torch.rsqrt(h.float().square().mean(-1, keepdim=True) + expected.norm.eps)).type_as(h) * nw
                logits = F.linear(normalized, weight)
                losses.append(F.cross_entropy(logits.reshape(-1, 256), yy.reshape(-1)))
        torch.stack(losses).mean().backward()
        errors = []
        for (name, p), (_, q) in zip(expected.named_parameters(), actual.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6, msg=name)
            errors.append((p.grad - q.grad).abs().max().item())
        result[f"accumulation_{accumulation}"] = max(errors)
    return result


def numerical_and_resume():
    x = torch.arange(32, device="cuda").reshape(2, 16)
    dense, fused = tiny(), tiny()
    a = local_backward(dense, x, x + 1, implementation="dense", readout_training="final")
    b = local_backward(fused, x, x + 1, readout_training="final")
    torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4)
    errors = []
    for (name, p), (_, q) in zip(dense.named_parameters(), fused.named_parameters()):
        err = relative_error(q.grad, p.grad); assert err < .04, (name, err)
        errors.append(err)
    # Readout parameter detachment cannot change its numerical values or the
    # current block's gradient; compare both policies at identical weights.
    all_model, final_model = tiny(), tiny()
    all_loss = local_backward(all_model, x, x + 1, implementation="dense", amp=False)
    final_loss = local_backward(final_model, x, x + 1, implementation="dense", amp=False, readout_training="final")
    assert_tree_equal(all_loss, final_loss)
    for all_block, final_block in zip(all_model.layers, final_model.layers):
        for p, q in zip(all_block.parameters(), final_block.parameters()):
            assert_tree_equal(p.grad, q.grad)
    assert_tree_equal(all_model.hidden_step_raw.grad, final_model.hidden_step_raw.grad)
    model = tiny(); opt = model.configure_optimizers(.1, 5e-4, (.9, .95), "cuda")
    for step in range(4):
        opt.zero_grad(set_to_none=True)
        local_backward(model, x + step, x + step + 1, readout_training="final")
        clip_local_gradients(model); opt.step()
    stream = io.BytesIO(); torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), stream)
    stream.seek(0); checkpoint = torch.load(stream, weights_only=False)
    resumed = tiny(); resumed.load_state_dict(checkpoint["model"])
    ropt = resumed.configure_optimizers(.1, 5e-4, (.9, .95), "cuda"); ropt.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    assert_tree_equal(model.state_dict(), resumed.state_dict()); assert_tree_equal(opt.state_dict(), ropt.state_dict())
    for net, optimizer in ((model, opt), (resumed, ropt)):
        optimizer.zero_grad(set_to_none=True); local_backward(net, x + 4, x + 5, readout_training="final")
        clip_local_gradients(net); optimizer.step()
    for p, q in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-4, atol=2e-6)
    before = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    gradients = [p.grad.clone() for p in model.parameters()]
    rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    metrics = validation_metrics(model, [(x, x + 1)])
    assert all(metrics[f"val_hidden/layer_{i}/update_rms"] > 0 for i in range(3))
    assert_tree_equal(model.state_dict(), before[0]); assert_tree_equal(opt.state_dict(), before[1])
    assert_tree_equal([p.grad for p in model.parameters()], gradients)
    assert_tree_equal((torch.get_rng_state(), torch.cuda.get_rng_state()), rng)
    return dict(max_parameter_gradient_relative_l2_error=max(errors),
                readout_policy_preserves_values_and_block_gradients=True,
                checkpoint_roundtrip_exact=True, resumed_step_within_tolerance=True,
                validation_preserves_state_rng_and_gradients=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); args = p.parse_args()
    import wandb
    run = wandb.init(project="nanochat", entity="yibozhong657-none", job_type="correctness",
                     name="gradformer-local-sgd-final-readout-check",
                     config={**vars(args), "source": json.loads(Path('gradformer/source.json').read_text()),
                             "gpu": torch.cuda.get_device_name(), "hidden_step_size": .01,
                             "parameter_lr": .0005, "parameter_optimizer": "AdamW"})
    result, exit_code = {}, 1
    try:
        for name, function in (("routing", routing), ("reference", reference), ("numerical_and_resume", numerical_and_resume)):
            result[name] = function(); print(json.dumps({"check": name, "result": result[name]}), flush=True)
        result["passed"] = True; exit_code = 0
    except Exception as error:
        result.update(passed=False, failure=f"{type(error).__name__}: {error}"); raise
    finally:
        Path(args.output).write_text(json.dumps(result, indent=2)); run.summary.update(result); run.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
