"""Strict block-local full-vocabulary CE with shared readout and hidden updates.

Forward values match the corresponding hidden-Adam/SGD model. Only the training
graph changes: hidden and any incoming moments stop at each block boundary. The
objective is the mean of the per-layer, all-valid-token mean CEs. Shared tied
embedding/head and final RMSNorm can receive all readout gradients or only the
final layer's readout gradient. The tied embedding additionally receives the
first block's input-path gradient in either mode.
"""
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from gradformer.core import hidden_step


def amp_context(model, enabled=True):
    return (torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
            if enabled and next(model.parameters()).is_cuda else nullcontext())


def readout_loss(model, hidden, targets, implementation="cce", reduction="mean", train_readout=True):
    if train_readout:
        features = model.norm(hidden)
        classifier = model.output.weight
    else:
        # Match native RMSNorm forward exactly, freezing readout parameters
        # without removing its derivative with respect to the hidden state.
        features = model.norm._norm(hidden.float()).type_as(hidden) * model.norm.weight.detach()
        classifier = model.output.weight.detach()
    if implementation == "cce":
        from cut_cross_entropy import linear_cross_entropy
        # Full vocabulary, no gradient filtering or token subsampling. CCE
        # accumulates gradients in fp32 (or compensated sums where necessary).
        return linear_cross_entropy(features, classifier, targets,
                                    ignore_index=-1, shift=0, reduction=reduction,
                                    impl="cce", filter_eps=None, filter_e_grad=False,
                                    filter_c_grad=False, accum_e_fp32=True, accum_c_fp32=True)
    if implementation != "dense":
        raise ValueError(implementation)
    logits = model.output(features) if train_readout else F.linear(features, classifier)
    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                           ignore_index=-1, reduction=reduction)
    return loss.view_as(targets) if reduction == "none" else loss


def local_losses(model, tokens, targets, implementation="cce", amp=True, readout_training="all"):
    """Yield one fresh local graph at a time; caller may free it immediately.

    No parameter changes are allowed while iterating. Calling backwards on an
    earlier yield cannot influence a later graph via shared nonleaf tensors.
    """
    if model.params.hidden_update not in ("adam", "sgd") or model.params.dropout != 0:
        raise ValueError("Strict local pilot requires hidden Adam/SGD without dropout")
    if readout_training not in ("all", "final"):
        raise ValueError(readout_training)
    h, state = model.tok_embeddings(tokens), None
    cos, sin = model.freqs_cos[:tokens.size(1)], model.freqs_sin[:tokens.size(1)]
    for layer, block in enumerate(model.layers):
        with amp_context(model, amp):
            predicted = -block(h, cos, sin, return_delta=True)
            updated, next_state = hidden_step(h, predicted, model.hidden_step_raw[layer],
                                               state, layer, model.params)
            loss = readout_loss(model, updated, targets, implementation,
                                train_readout=readout_training == "all" or layer == len(model.layers) - 1)
        # Cut BOTH paths, while preserving the actual state values for inference.
        h = updated.detach()
        state = tuple(t.detach() for t in next_state) if next_state is not None else None
        yield layer, loss
        del loss, updated, next_state, predicted


def local_backward(model, x, y, *, accumulation=1, implementation="cce", amp=True, readout_training="all"):
    values = []
    for _, loss in local_losses(model, x, y, implementation, amp, readout_training):
        (loss / (len(model.layers) * accumulation)).backward()
        values.append(loss.detach())
    return torch.stack(values)


def end_to_end_backward(model, x, y, *, accumulation=1, implementation="cce", amp=True):
    """Matched control: final CE, connected hidden path, same 1/depth coefficient.

    Matches local training's final-head loss scale and uses the same independent
    gradient clipping in the caller. No intermediate classifiers are evaluated.
    """
    with amp_context(model, amp):
        h, state = model.tok_embeddings(x), None
        cos, sin = model.freqs_cos[:x.size(1)], model.freqs_sin[:x.size(1)]
        for layer, block in enumerate(model.layers):
            predicted = -block(h, cos, sin, return_delta=True)
            h, state = hidden_step(h, predicted, model.hidden_step_raw[layer], state, layer, model.params)
        loss = readout_loss(model, h, y, implementation)
    (loss / (len(model.layers) * accumulation)).backward()
    return loss.detach().reshape(1)


@torch.no_grad()
def clip_local_gradients(model, max_norm=1.):
    """Independent clipping: no other block's gradient changes this block's scale."""
    norms = []
    for i, block in enumerate(model.layers):
        gradients = [p.grad for p in block.parameters() if p.grad is not None]
        if model.hidden_step_raw.grad is not None:
            gradients.append(model.hidden_step_raw.grad[i:i + 1])
        norm = torch.stack([g.float().norm() for g in gradients]).norm()
        if not torch.isfinite(norm):
            raise RuntimeError(f"Nonfinite local gradient norm at block {i}")
        scale = (max_norm / (norm + 1e-6)).clamp(max=1.)
        for g in gradients:
            g.mul_(scale)
        norms.append(norm)
    shared = [model.output.weight, model.norm.weight]
    shared_norm = torch.nn.utils.clip_grad_norm_(shared, max_norm, error_if_nonfinite=True)
    return torch.stack(norms), shared_norm


@torch.no_grad()
def validation_metrics(model, batches, implementation="cce", amp=True):
    """All-token, fixed-head CE before/after every actual hidden update."""
    was_training = model.training
    model.eval()
    ces, fractions, magnitudes = [], [], []
    try:
        for x, y in batches:
            with amp_context(model, amp):
                h, state = model.tok_embeddings(x), None
                cos, sin = model.freqs_cos[:x.size(1)], model.freqs_sin[:x.size(1)]
                previous = readout_loss(model, h, y, implementation, reduction="none")
                valid = y != -1
                row, decreased, sizes = [previous[valid].mean()], [], []
                for i, block in enumerate(model.layers):
                    incoming = h
                    predicted = -block(h, cos, sin, return_delta=True)
                    h, state = hidden_step(h, predicted, model.hidden_step_raw[i], state, i, model.params)
                    hidden_rms = incoming.float().square().mean().sqrt()
                    update_rms = (h - incoming).float().square().mean().sqrt()
                    sizes.append(torch.stack((predicted.float().square().mean().sqrt(), update_rms,
                                              update_rms / hidden_rms.clamp_min(1e-30))))
                    current = readout_loss(model, h, y, implementation, reduction="none")
                    row.append(current[valid].mean())
                    decreased.append((current[valid] < previous[valid]).float().mean())
                    previous = current
                ces.append(torch.stack(row)); fractions.append(torch.stack(decreased)); magnitudes.append(torch.stack(sizes))
        ce, frac = torch.stack(ces).mean(0), torch.stack(fractions).mean(0)
        sizes = torch.stack(magnitudes).mean(0)
        result = {"val_ce": ce[-1].item(), "val_local_ce/mean": ce[1:].mean().item(),
                  "val_local_ce/input": ce[0].item()}
        for i in range(len(model.layers)):
            result.update({f"val_local_ce/layer_{i}": ce[i + 1].item(),
                           f"val_local_ce/layer_{i}/delta": (ce[i + 1] - ce[i]).item(),
                           f"val_local_ce/layer_{i}/token_improved_fraction": frac[i].item()})
            for j, name in enumerate(("prediction_rms", "update_rms", "relative_update_rms")):
                result[f"val_hidden/layer_{i}/{name}"] = sizes[i, j].item()
        return result
    finally:
        model.train(was_training)
