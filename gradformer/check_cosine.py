"""GPU correctness gates for cosine-only gradients and nonintrusive probes."""
import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from gradformer.balance import for_model
from gradformer.check import make_model
from gradformer.check_balance import check
from gradformer.cosine_probe import CosineProbe, diagnostic_context, validation_metrics
from gradformer.runtime import loss_forward


def assert_tree_equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def trajectory(weight):
    models = [make_model("llama2", "adam", "cuda") for _ in range(2)]
    optimizers = [m.configure_optimizers(.1, 5e-4, (.9, .95), "cuda") for m in models]
    balancers = [for_model(m, "llama2") for m in models]
    x = torch.arange(16, device="cuda").reshape(2, 8)
    last_metrics = {}
    for step in range(10):
        for enabled, (model, optimizer, balancer) in enumerate(zip(models, optimizers, balancers)):
            optimizer.zero_grad(set_to_none=True)
            balancer.reset()
            probe = CosineProbe(model, balancer, refresh=True) if enabled else None
            for offset in (step, step + 2):
                xx, yy = x + offset, (x + offset + 1) % 128
                with diagnostic_context(model):
                    ntp, states, predictions = loss_forward(model, "llama2", xx, yy, True)
                observer = (lambda hs, ps, gs, a, m: probe.observe(xx, yy, hs, ps, gs, a, m)) if probe else None
                balancer.backward(ntp, states, predictions, alpha=0, beta=1, accumulation=2,
                                  compute_aux=weight != 0 or bool(probe), observer=observer)
            if probe:
                rng_before = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                state_before = copy.deepcopy(optimizer.state_dict())
                probe.before_step()
                assert_tree_equal(optimizer.state_dict(), state_before)
                assert_tree_equal((torch.get_rng_state(), torch.cuda.get_rng_state()), rng_before)
                # Independent fixed-target loss formula, not prediction_loss.
                with torch.no_grad(), diagnostic_context(model):
                    losses = []
                    for xx, yy, targets, _, _ in probe.batches:
                        _, _, predictions = loss_forward(model, "llama2", xx, yy, True)
                        for p, g in zip(predictions, targets):
                            valid = g.float().norm(dim=-1) > 1e-8
                            losses.append((1 - F.cosine_similarity(p.float(), g.float(), dim=-1, eps=1e-8))[valid].mean())
                    assert abs(torch.stack(losses).mean().item() - probe.before) < 2e-6
                if step == 0:
                    noop = probe.after_step()
                    assert abs(noop["probe/fixed_loss_delta"]) < 1e-7
                    assert noop["probe/all/raw_cosine_grad_dot_update"] == 0
            balancer.finish(accumulation=2, fixed_weight=weight)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if probe:
                state_before = copy.deepcopy(optimizer.state_dict())
                grads_before = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
                targets_before = [[t.clone() for t in b[2]] for b in probe.batches]
                last_metrics = probe.after_step()
                validation_metrics(model, [(x, x + 1)])
                assert_tree_equal([p.grad for p in model.parameters()], grads_before)
                assert_tree_equal(optimizer.state_dict(), state_before)
                assert_tree_equal((torch.get_rng_state(), torch.cuda.get_rng_state()), rng_before)
                assert_tree_equal([b[2] for b in probe.batches], targets_before)
                expected_dot = sum((g.double() * (p.detach() - old).double()).sum().item()
                                   for g, p, old in zip(probe.raw_gradients, model.parameters(), probe.parameters_before) if g is not None)
                assert abs(expected_dot - last_metrics["probe/all/raw_cosine_grad_dot_update"]) < 1e-6
        assert_tree_equal(models[0].state_dict(), models[1].state_dict())
        assert_tree_equal(optimizers[0].state_dict(), optimizers[1].state_dict())
    return dict(steps=10, bitwise_equal_parameters_and_optimizer=True, last_probe=last_metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import wandb
    run = wandb.init(project="nanochat", entity="yibozhong657-none", job_type="correctness",
                     name="gradformer-llama2-cosine-check", config=vars(args))
    try:
        results = {}
        for name, weight in (("ntp_only", 0.), ("fixed_cosine", 1.), ("balanced_cosine", None)):
            results[name] = dict(reference=check("llama2", "cuda", alpha=0, beta=1, fixed_weight=weight),
                                 trajectory=trajectory(weight))
        result = dict(passed=True, results=results)
        Path(args.output).write_text(json.dumps(result, indent=2))
        run.summary.update(result)
        print(json.dumps(result), flush=True)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
