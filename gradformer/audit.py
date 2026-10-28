"""Read-only checkpoint gradient audit plus an isolated frozen-target fitting probe."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import ModelArgs, Transformer
from gradformer.core import hidden_step, prediction_loss
from gradformer.train import batch


def cosine(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-8).mean().item()


def gradient_stats(a, b):
    aa = sum(g.float().square().sum() for g in a)
    bb = sum(g.float().square().sum() for g in b)
    ab = sum((g.float() * h.float()).sum() for g, h in zip(a, b))
    return dict(ntp_norm=aa.sqrt().item(), auxiliary_norm=bb.sqrt().item(),
                auxiliary_to_ntp_ratio=(bb.sqrt() / aa.sqrt().clamp_min(1e-20)).item(),
                cosine_with_ntp=(ab / (aa * bb).sqrt().clamp_min(1e-20)).item())


def load_model(folder):
    config = json.loads((folder / "config.json").read_text())
    args = dict(config["model"])
    if args["hidden_update"] == "baseline":
        args.update(hidden_update="sgd", hidden_step_size=1.)
    model = Transformer(ModelArgs(**args)).cuda().eval()
    weights = folder / "model.pt"
    if not weights.exists():
        latest = json.loads((folder / "latest.json").read_text())
        weights = folder / latest["directory"] / "model.pt"
    missing, unexpected = model.load_state_dict(torch.load(weights, map_location="cuda", weights_only=True), strict=False)
    assert not unexpected
    assert missing == (["hidden_step_raw"] if config["update"] == "baseline" else [])
    return model, config


def audit(model, x, y, mse_weight=.1, cosine_weight=.01):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ntp, inputs, predictions = model(x, y, capture_gradients=True, return_loss=True)
    targets = torch.autograd.grad(ntp, inputs, retain_graph=True)
    mse, _ = prediction_loss(predictions, targets, alpha=1., beta=0.)
    direction, _ = prediction_loss(predictions, targets, alpha=0., beta=1.)
    groups = [list(block.parameters()) for block in model.layers]
    parameters = [p for group in groups for p in group] + [model.hidden_step_raw]
    gradients = {}
    # Weights are effective coefficients after the outer auxiliary multiplier.
    for key, loss in (("ntp", ntp), ("mse", mse_weight * mse), ("cosine", cosine_weight * direction)):
        values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        gradients[key] = [g if g is not None else torch.zeros_like(p) for g, p in zip(values, parameters)]
    rows, state, offset = [], None, 0
    with torch.no_grad():
        for layer, (h, pred, target, group) in enumerate(zip(inputs, predictions, targets, groups)):
            updated, state = hidden_step(h, pred, model.hidden_step_raw[layer], state, layer, model.params)
            delta = updated - h
            eta = F.softplus(model.hidden_step_raw[layer]).item()
            row = dict(layer=layer, eta=eta, prediction_cosine=cosine(pred, target),
                       actual_descent_direction_cosine=cosine(-delta, target),
                       hidden_norm=h.float().norm(dim=-1).mean().item(),
                       prediction_norm=pred.float().norm(dim=-1).mean().item(),
                       target_norm=target.float().norm(dim=-1).mean().item(),
                       update_norm=delta.float().norm(dim=-1).mean().item(),
                       relative_update_norm=(delta.float().norm(dim=-1) / h.float().norm(dim=-1).clamp_min(1e-12)).mean().item())
            if state is not None:
                root = (state[1] / (1 - model.params.hidden_beta2 ** (layer + 1))).sqrt()
                row["epsilon_dominates_fraction"] = (root < model.params.hidden_eps).float().mean().item()
            end = offset + len(group)
            for key in ("mse", "cosine"):
                row[key + "_parameter_gradient"] = gradient_stats(gradients["ntp"][offset:end], gradients[key][offset:end])
            combined = [a + b for a, b in zip(gradients["mse"][offset:end], gradients["cosine"][offset:end])]
            row["total_auxiliary_parameter_gradient"] = gradient_stats(gradients["ntp"][offset:end], combined)
            row["rho_gradients"] = {k: v[-1][layer].item() for k, v in gradients.items()}
            rows.append(row)
            offset = end
    return dict(ntp=ntp.item(), raw_mse=mse.item(), raw_cosine_loss=direction.item(), layers=rows)


def frozen_probe(model, x, y, steps, run):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ntp, inputs, predictions = model(x, y, capture_gradients=True, return_loss=True)
    targets = torch.autograd.grad(ntp, inputs)
    inputs = [h.detach() for h in inputs]
    targets = [g.detach() for g in targets]
    del ntp, predictions
    results = []
    for layer in (0, len(model.layers) // 2, len(model.layers) - 1):
        block = copy.deepcopy(model.layers[layer]).train()
        optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3, weight_decay=0., fused=True)
        h, target = inputs[layer], targets[layer]
        freqs = (model.freqs_cos[:x.shape[1]], model.freqs_sin[:x.shape[1]])
        initial = None
        for step in range(steps + 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = -block(h, *freqs, return_delta=True)
            loss = 1 - F.cosine_similarity(prediction.float(), target.float(), dim=-1, eps=1e-8).mean()
            score = 1 - loss.detach().item()
            if initial is None:
                initial = score
            if step % 20 == 0 or step == steps:
                run.log({f"probe/layer_{layer}/cosine": score, "probe_step": step})
            if step == steps:
                break
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite frozen-target fitting loss")
            loss.backward()
            optimizer.step()
        results.append(dict(layer=layer, initial_cosine=initial, final_cosine=score, steps=steps))
        del block, optimizer
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--probe-steps", type=int, default=200)
    parser.add_argument("--variants", nargs="+", default=["baseline", "adam-no-aux", "adam-mse-cos"])
    parser.add_argument("--use-run-weights", action="store_true",
                        help="Use aux_weight*alpha and aux_weight*beta from each checkpoint config")
    parser.add_argument("--name", default="gradformer-llama2-gradient-and-hidden-step-audit")
    args = parser.parse_args()
    torch.manual_seed(1337)
    import wandb
    run = wandb.init(project="nanochat", entity="yibozhong657-none", job_type="diagnostic",
                     name=args.name,
                     config={**vars(args), "gpu": torch.cuda.get_device_name(),
                             "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "audit_aux_weights": "checkpoint config" if args.use_run_weights else {"mse": .1, "cosine": .01},
                             "baseline_control": "original baseline represented as hidden SGD eta=1",
                             "frozen_probe": "isolated block copies; frozen inputs/targets; cosine only; no NTP updates"})
    val = np.memmap(Path(args.data) / "val.bin", dtype=np.uint16, mode="r")
    train = np.memmap(Path(args.data) / "train.bin", dtype=np.uint16, mode="r")
    result = {}
    for variant in args.variants:
        model, config = load_model(Path(args.checkpoints) / variant)
        weights = (config["aux_weight"] * config["alpha"], config["aux_weight"] * config["beta"]) if args.use_run_weights else (.1, .01)
        rng = np.random.default_rng(987654)
        records = []
        for _ in range(args.batches):
            x, y = batch(val, rng, 16, 256, "cuda")
            records.append(audit(model, x, y, *weights))
        result[variant] = dict(source=config["source"], configuration=config, batches=records,
                               effective_mse_weight=weights[0], effective_cosine_weight=weights[1])
        summary = dict(variant=variant, mse_weight=weights[0], cosine_weight=weights[1],
                       mean_ntp=float(np.mean([row["ntp"] for row in records])))
        for term in ("mse", "cosine"):
            summary[term + "_to_ntp_per_layer"] = [float(np.mean([
                row["layers"][layer][term + "_parameter_gradient"]["auxiliary_to_ntp_ratio"]
                for row in records])) for layer in range(len(model.layers))]
        print(json.dumps({"gradient_audit": summary}), flush=True)
        if variant == "adam-mse-cos" and args.probe_steps > 0:
            x, y = batch(train, np.random.default_rng(444), 4, 32, "cuda")
            result[variant]["frozen_target_probe"] = frozen_probe(model, x, y, args.probe_steps, run)
        del model
        torch.cuda.empty_cache()
    Path(args.output).write_text(json.dumps(result, indent=2))
    run.summary["results"] = result
    run.finish()


if __name__ == "__main__":
    main()
