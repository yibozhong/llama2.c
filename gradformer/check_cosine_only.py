"""Independent reference and optimizer checks for zero direct NTP updates."""
import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from gradformer.balance import for_model
from gradformer.check import make_model
from gradformer.check_cosine import trajectory
from gradformer.runtime import loss_forward


def reference_checks():
    result = {}
    x = torch.arange(16, device="cuda").reshape(2, 8)
    for mode in ("sgd", "adam"):
        for accumulation in (1, 3, 8):
            ref = make_model("llama2", mode, "cuda")
            actual = copy.deepcopy(ref)
            engine = for_model(actual, "llama2", cosine_only=True)
            largest, target_error, ntp_leaf_events = 0., 0., []
            for iteration in range(2):
                ref.zero_grad(set_to_none=True)
                actual.zero_grad(set_to_none=True)
                engine.reset()
                expected = [None] * len(list(ref.parameters()))
                for micro in range(accumulation):
                    xx, yy = x + micro + 3 * iteration, x + micro + 3 * iteration + 1
                    ref.zero_grad(set_to_none=True)
                    ntp, inputs, predictions = loss_forward(ref, "llama2", xx, yy, True)
                    targets = [None] * len(inputs)
                    def capture(i):
                        def hook(grad):
                            targets[i] = grad.detach().clone()
                        return hook
                    handles = [h.register_hook(capture(i)) for i, h in enumerate(inputs)]
                    ntp.backward(retain_graph=True)
                    for handle in handles:
                        handle.remove()
                    # NTP would update these parameters if it leaked through.
                    assert ref.norm.weight.grad.norm().item() > 0
                    assert ref.hidden_step_raw.grad[-1].abs().item() > 0
                    for p, g in zip(ref.parameters(), expected):
                        p.grad = g
                    terms = []
                    for p, g in zip(predictions, targets):
                        valid = g.float().norm(dim=-1) > 1e-8
                        assert valid.any().item()
                        terms.append((1 - F.cosine_similarity(p.float(), g.float(), dim=-1, eps=1e-8))[valid].mean())
                    (torch.stack(terms).mean() / accumulation).backward()
                    expected = [p.grad for p in ref.parameters()]

                    ntp, inputs, predictions = loss_forward(actual, "llama2", xx, yy, True)
                    prior = [g.clone() if g is not None else None for g in engine.aux_gradients]
                    phase = {"target": True}
                    def leaf_hook(name):
                        def hook(grad):
                            if phase["target"]:
                                ntp_leaf_events.append(name)
                        return hook
                    handles = [p.register_hook(leaf_hook(name)) for name, p in actual.named_parameters()]
                    def observe(hs, ps, gs, aux, metrics):
                        nonlocal target_error
                        assert not ntp_leaf_events
                        for p, old in zip(actual.parameters(), prior):
                            if old is None:
                                assert p.grad is None
                            else:
                                torch.testing.assert_close(p.grad, old, rtol=0, atol=0)
                        for g, target in zip(gs, targets):
                            assert not g.requires_grad and g.grad_fn is None
                            torch.testing.assert_close(g, target, rtol=5e-4, atol=5e-6)
                            target_error = max(target_error, (g - target).abs().max().item())
                        phase["target"] = False
                    try:
                        engine.backward(ntp, inputs, predictions, accumulation=accumulation, observer=observe)
                    finally:
                        for handle in handles:
                            handle.remove()
                engine.finish(accumulation=accumulation, fixed_weight=1.)
                assert actual.norm.weight.grad is None
                assert actual.hidden_step_raw.grad[-1].item() == 0
                assert actual.tok_embeddings.weight.grad.norm().item() > 0
                assert actual.output.weight is actual.tok_embeddings.weight
                for (name, p), (other, q) in zip(ref.named_parameters(), actual.named_parameters()):
                    assert name == other
                    if p.grad is None:
                        assert q.grad is None, name
                        continue
                    torch.testing.assert_close(p.grad, q.grad, rtol=5e-4, atol=5e-6, msg=name)
                    largest = max(largest, (p.grad - q.grad).abs().max().item())
                    with torch.no_grad():
                        p.add_(p.grad, alpha=-.01)
                        q.add_(q.grad, alpha=-.01)
            result[f"{mode}_accumulation_{accumulation}"] = dict(
                max_abs_gradient_error=largest, max_abs_target_error=target_error,
                ntp_parameter_hook_events=len(ntp_leaf_events), detached_targets=True,
                output_norm_gradient_none=True, last_hidden_step_gradient_zero=True,
                tied_embedding_gradient_nonzero=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import wandb
    run = wandb.init(project="nanochat", entity="yibozhong657-none", job_type="correctness",
                     name="gradformer-llama2-strict-cosine-only-check", config=vars(args))
    try:
        result = dict(reference=reference_checks(), trajectory=trajectory(1., cosine_only=True), passed=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        run.summary.update(result)
        print(json.dumps(result), flush=True)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
