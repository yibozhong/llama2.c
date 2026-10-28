"""Strict local CE plus detached local input-gradient supervision.

The target differentiates CE(head(updated_hidden), y) with respect to the
current block's input, INCLUDING the current block and its hidden update.
It is not the direct-head gradient at the unmodified input or the final NTP
gradient. Forward values, shared readout policy and layer boundaries match
local_ce.py. No label or gradient enters the forward hidden update.
"""
import torch

from gradformer.core import hidden_step, prediction_loss
from gradformer.local_ce import amp_context, readout_loss


def local_graphs(model, x, y, *, implementation="cce", amp=True):
    if model.params.hidden_update != "sgd" or model.params.dropout != 0:
        raise ValueError("This experiment uses dropout-free hidden SGD")
    h = model.tok_embeddings(x)
    cos, sin = model.freqs_cos[:x.size(1)], model.freqs_sin[:x.size(1)]
    for layer, block in enumerate(model.layers):
        h.requires_grad_(True)
        with amp_context(model, amp):
            predicted = -block(h, cos, sin, return_delta=True)
            updated, _ = hidden_step(h, predicted, model.hidden_step_raw[layer], None, layer, model.params)
            ce = readout_loss(model, updated, y, implementation,
                              train_readout=layer == len(model.layers) - 1)
        yield layer, h, predicted, updated, ce
        h = updated.detach()


def _gradient_stats(ce, auxiliary):
    # Values are from one microbatch, weighted exactly as sent to backward.
    ce_sq = sum(g.float().square().sum() for g in ce.values())
    aux_sq = sum(g.float().square().sum() for g in auxiliary.values()) if auxiliary else ce_sq * 0
    dot = sum((g.float() * auxiliary[k].float()).sum()
              for k, g in ce.items() if k in auxiliary)
    if not isinstance(dot, torch.Tensor):
        dot = ce_sq * 0
    cn, an = ce_sq.sqrt(), aux_sq.sqrt()
    return torch.stack((cn, an, an / cn.clamp_min(1e-30), dot / (cn * an).clamp_min(1e-30)))


def backward(model, x, y, *, alpha=0., beta=1., accumulation=1,
             implementation="cce", amp=True, diagnostics=False):
    """One CE backward + one block auxiliary backward; targets reuse the CE pass.

    Every local objective is (CE + alpha*MSE + beta*(1-cosine))/depth.
    Target scale is the unweighted mean-token CE, independent of accumulation.
    Parameter gradients from CE and auxiliary are summed before local clipping.
    """
    active = alpha != 0 or beta != 0
    scale = len(model.layers) * accumulation
    losses, statistics, signals = [], [], []
    for layer, h, predicted, updated, ce in local_graphs(model, x, y, implementation=implementation, amp=amp):
        target = []
        handle = h.register_hook(lambda g: target.append(g.detach().clone() * scale))
        phase = [0]
        parameter_grads = [{}, {}]
        handles = []
        if diagnostics:
            for name, parameter in model.layers[layer].named_parameters():
                def capture(g, key=name):
                    parameter_grads[phase[0]][key] = g.detach().clone()
                handles.append(parameter.register_hook(capture))
        try:
            (ce / scale).backward(retain_graph=active)
            handle.remove()
            assert len(target) == 1 and not target[0].requires_grad
            if active:
                auxiliary, stats = prediction_loss([predicted], target, y != -1, alpha=alpha, beta=beta)
                phase[0] = 1
                (auxiliary / scale).backward()
            else:
                with torch.no_grad():
                    auxiliary, stats = prediction_loss([predicted], target, y != -1, alpha=0., beta=0.)
            if diagnostics:
                signals.append(_gradient_stats(*parameter_grads).detach())
        finally:
            handle.remove()
            for item in handles:
                item.remove()
        losses.append(torch.stack((ce.detach(), auxiliary.detach())))
        statistics.append(stats[0].detach())
    return torch.stack(losses), torch.stack(statistics), torch.stack(signals) if signals else None


def gradient_validation(model, batches, *, implementation="cce", amp=True):
    """Fixed-batch local alignment; no parameter .grad writes or RNG draws."""
    was_training = model.training
    model.eval()
    records = []
    try:
        with torch.enable_grad():
            for x, y in batches:
                rows = []
                for _, h, predicted, _, ce in local_graphs(model, x, y, implementation=implementation, amp=amp):
                    target, = torch.autograd.grad(ce, h, create_graph=False)
                    with torch.no_grad():
                        _, stats = prediction_loss([predicted], [target], y != -1, alpha=0., beta=1.)
                    rows.append(stats[0])
                records.append(torch.stack(rows))
        average = torch.stack(records).mean(0)
        names = ("mse", "cosine", "prediction_norm", "target_norm", "directional_fraction")
        result = {f"val_gradient/{name}": average[:, i].mean().item() for i, name in enumerate(names)}
        for layer, row in enumerate(average):
            result.update({f"val_gradient/layer_{layer}/{name}": row[i].item() for i, name in enumerate(names)})
        return result
    finally:
        model.train(was_training)
