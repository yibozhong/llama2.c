"""GPU checks for EMA targets, gradient routing, and exact teacher resume."""
import argparse
import copy
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from gradformer.balance import for_model
from gradformer.check import make_model
from gradformer.check_cosine import assert_tree_equal
from gradformer.cosine_probe import CosineProbe, diagnostic_context, validation_metrics
from gradformer.ema import EMATeacher
from gradformer.runtime import loss_forward


def identity_checks():
    result = {}
    x = torch.arange(16, device="cuda").reshape(2, 8)
    for mode in ("sgd", "adam"):
        model = make_model("llama2", mode, "cuda")
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state())
        teacher = EMATeacher(model, .99)
        assert_tree_equal((torch.get_rng_state(), torch.cuda.get_rng_state()), rng)
        errors = []
        for amp in (False, True):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                ntp, inputs, predictions = loss_forward(model, "llama2", x, x + 1, True)
                expected = torch.autograd.grad(ntp, inputs, retain_graph=True)
                actual = teacher.targets(inputs, predictions, x + 1)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=2e-3 if amp else 3e-5, atol=2e-5 if amp else 1e-7)
                    errors.append((a - b).abs().max().item())
                    assert not a.requires_grad and a.grad_fn is None
                if mode == "adam":
                    wrong = teacher.targets(inputs, [torch.zeros_like(p) for p in predictions], x + 1)
                    assert (actual[-1] - wrong[-1]).abs().max().item() > 1e-7
            assert all(p.grad is None for p in model.parameters())
            assert all(p.grad is None and not p.requires_grad for p in teacher.model.parameters())
        # A finite-difference check for the actual suffix objective in float64.
        double_model = make_model("llama2", mode, "cuda").double()
        double_teacher = EMATeacher(double_model, .99)
        z = torch.randn(2, 8, 32, device="cuda", dtype=torch.double, requires_grad=True)
        scalar = double_teacher.suffix_loss(z, 0, x + 1, None)
        gradient, = torch.autograd.grad(scalar, z)
        direction = torch.randn_like(z); direction /= direction.norm()
        # llama2 RMSNorm internally computes fp32, so finite difference uses
        # a scale above that rounding noise rather than double machine epsilon.
        eps = 1e-3
        with torch.no_grad():
            numerical = (double_teacher.suffix_loss(z + eps * direction, 0, x + 1, None)
                         - double_teacher.suffix_loss(z - eps * direction, 0, x + 1, None)) / (2 * eps)
        analytic = (gradient * direction).sum()
        torch.testing.assert_close(analytic, numerical, rtol=.03, atol=3e-5)
        old = {n: p.detach().clone() for n, p in teacher.model.named_parameters()}
        with torch.no_grad():
            for p in model.parameters():
                p.add_(.001)
        teacher.update(model)
        for n, p in model.named_parameters():
            torch.testing.assert_close(dict(teacher.model.named_parameters())[n], .99 * old[n] + .01 * p,
                                       rtol=1e-6, atol=1e-8)
        assert teacher.updates == 1 and teacher.model.output.weight is teacher.model.tok_embeddings.weight
        result[mode] = dict(max_identity_target_error=max(errors), finite_difference_error=(analytic - numerical).abs().item(),
                            teacher_has_no_parameter_gradients=True, tied_weights_updated_once=True)
    return result


def gradient_checks():
    result = {}
    x = torch.arange(16, device="cuda").reshape(2, 8)
    for pure in (False, True):
        for accumulation in (1, 3):
            reference = make_model("llama2", "adam", "cuda")
            actual = copy.deepcopy(reference)
            teacher = EMATeacher(actual, .99)
            with torch.no_grad():
                for p in teacher.model.parameters():
                    p.add_(.002 * torch.randn_like(p))
            engine = for_model(actual, "llama2", cosine_only=pure)
            losses = []
            for micro in range(accumulation):
                xx, yy = x + micro, x + micro + 1
                ntp, hs, ps = loss_forward(reference, "llama2", xx, yy, True)
                targets = teacher.targets(hs, ps, yy)
                terms = []
                for p, g in zip(ps, targets):
                    valid = g.norm(dim=-1) > 1e-8
                    terms.append((1 - F.cosine_similarity(p.float(), g.float(), dim=-1, eps=1e-8))[valid].mean())
                cosine = torch.stack(terms).mean()
                losses.append((cosine if pure else ntp + cosine) / accumulation)
                an, ah, ap = loss_forward(actual, "llama2", xx, yy, True)
                at = teacher.targets(ah, ap, yy)
                engine.backward(an, ah, ap, alpha=0, beta=1, accumulation=accumulation, target_gradients=at)
            sum(losses).backward()
            engine.finish(accumulation=accumulation, fixed_weight=1)
            largest = 0.
            for (name, a), (_, b) in zip(reference.named_parameters(), actual.named_parameters()):
                if a.grad is None:
                    assert b.grad is None, name
                    continue
                torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=5e-6, msg=name)
                largest = max(largest, (a.grad - b.grad).abs().max().item())
            assert all(p.grad is None for p in teacher.model.parameters()) and teacher.updates == 0
            if pure:
                assert actual.norm.weight.grad is None and actual.hidden_step_raw.grad[-1].item() == 0
            result[f"{'pure' if pure else 'joint'}_accumulation_{accumulation}"] = largest
    return result


def resume_and_probe(pure):
    def setup():
        model = make_model("llama2", "adam", "cuda")
        return model, model.configure_optimizers(.1, 5e-4, (.9, .95), "cuda"), EMATeacher(model, .99), for_model(model, "llama2", cosine_only=pure)
    pairs = [setup(), setup()]
    x = torch.arange(16, device="cuda").reshape(2, 8)
    for step in range(8):
        for enabled in (0, 1):
            model, optimizer, teacher, engine = pairs[enabled]
            optimizer.zero_grad(set_to_none=True); engine.reset()
            probe = CosineProbe(model, engine, refresh=True, teacher=teacher) if enabled else None
            rng_before = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            for micro in range(2):
                xx, yy = x + step + micro, x + step + micro + 1
                with diagnostic_context(model):
                    ntp, hs, ps = loss_forward(model, "llama2", xx, yy, True)
                    targets = teacher.targets(hs, ps, yy)
                observer = (lambda hs, ps, gs, a, m: probe.observe(xx, yy, hs, ps, gs, a, m)) if probe else None
                engine.backward(ntp, hs, ps, alpha=0, beta=1, accumulation=2, target_gradients=targets, observer=observer)
            if probe:
                probe.before_step()
            engine.finish(accumulation=2, fixed_weight=1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if probe:
                before = copy.deepcopy(teacher.state_dict())
                grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
                state = copy.deepcopy(optimizer.state_dict())
                probe.after_step()
                metrics = validation_metrics(model, [(x, x + 1)], teacher=teacher)
                assert "val_cosine/mean" in metrics and "val_ema_cosine/mean" in metrics
                assert_tree_equal(teacher.state_dict(), before)
                assert_tree_equal([p.grad for p in model.parameters()], grads)
                assert_tree_equal(optimizer.state_dict(), state)
            assert_tree_equal((torch.get_rng_state(), torch.cuda.get_rng_state()), rng_before)
            teacher.update(model)
            assert teacher.updates == step + 1
        assert_tree_equal(pairs[0][0].state_dict(), pairs[1][0].state_dict())
        assert_tree_equal(pairs[0][1].state_dict(), pairs[1][1].state_dict())
        assert_tree_equal(pairs[0][2].state_dict(), pairs[1][2].state_dict())
        if step == 3:
            model, optimizer, teacher, _ = pairs[1]
            stream = io.BytesIO()
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), teacher=teacher.state_dict()), stream)
            stream.seek(0); state = torch.load(stream, weights_only=False)
            pairs[1] = setup()
            pairs[1][0].load_state_dict(state["model"])
            pairs[1][1].load_state_dict(state["optimizer"])
            pairs[1][2].load_state_dict(state["teacher"])
    return dict(steps=8, exact_model_optimizer_teacher_resume=True, probes_preserve_trajectory=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True); args = parser.parse_args()
    import wandb
    run = wandb.init(project="nanochat", entity="yibozhong657-none", job_type="correctness",
                     name="gradformer-llama2-ema-target-check", config=vars(args))
    try:
        result = dict(identity=identity_checks(), gradients=gradient_checks(),
                      resume_joint=resume_and_probe(False), resume_pure=resume_and_probe(True), passed=True)
        Path(args.output).write_text(json.dumps(result, indent=2)); run.summary.update(result)
        print(json.dumps(result), flush=True)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
