"""Cosine feedback for blocks, direct final CE for the readout only.

The CE VJP requests readout aliases, never the tied embedding leaf. This gives
exactly the direct-head gradient of CE(head(stopgrad(hidden)), y) without a
second vocabulary projection. Targets still differentiate the connected model.
"""
import torch
import torch.nn.functional as F

from gradformer.core import hidden_step, prediction_loss


def forward_trace(model, x, y):
    h = model.dropout(model.tok_embeddings(x))
    inputs, predictions, state = [], [], None
    cos, sin = model.freqs_cos[:x.shape[1]], model.freqs_sin[:x.shape[1]]
    for i, block in enumerate(model.layers):
        pred = -block(h, cos, sin, return_delta=True)
        inputs.append(h); predictions.append(pred)
        eta = model.hidden_step_raw[i]
        incoming = state
        h, state = hidden_step(h, pred, eta, state, i, model.params)
    # Distinct graph nodes isolate the direct readout path despite weight tying.
    head = model.output.weight.view_as(model.output.weight)
    norm = model.norm.weight.view_as(model.norm.weight)
    normalized = model.norm._norm(h.float()).type_as(h) * norm
    logits = F.linear(normalized, head)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-1)
    return dict(loss=loss, inputs=inputs, predictions=predictions, hidden=h,
                head=head, norm=norm, last_eta=eta, last_incoming=incoming)


def backward(model, x, y, *, mode="cosine-head", train_last_eta=True,
             accumulation=1, observer=None):
    if mode not in ("cosine-head", "cosine-only", "ntp-only") or accumulation < 1:
        raise ValueError("Invalid cosine/readout training mode")
    trace = forward_trace(model, x, y)
    ntp = trace["loss"]
    if mode == "ntp-only":
        (ntp / accumulation).backward()
        return ntp.detach(), ntp.new_zeros(()), None
    nodes = list(trace["inputs"])
    if mode == "cosine-head":
        nodes += [trace["head"], trace["norm"]]
        if train_last_eta:
            nodes += [trace["last_eta"]]
    values = torch.autograd.grad(ntp, nodes, retain_graph=True, create_graph=False)
    depth = len(trace["inputs"])
    targets = values[:depth]
    aux, metrics = prediction_loss(trace["predictions"], targets, alpha=0., beta=1.)
    if observer:
        observer(trace, targets, values[depth:])
    (aux / accumulation).backward()
    if mode == "cosine-head":
        with torch.no_grad():
            for p, g in zip((model.output.weight, model.norm.weight), values[depth:depth+2]):
                delta = g / accumulation
                if p.grad is None:
                    p.grad = delta.clone()
                else:
                    p.grad.add_(delta)
            if train_last_eta:
                if model.hidden_step_raw.grad is None:
                    model.hidden_step_raw.grad = torch.zeros_like(model.hidden_step_raw)
                model.hidden_step_raw.grad[-1].add_(values[-1] / accumulation)
    return ntp.detach(), aux.detach(), metrics.detach()


@torch.no_grad()
def clip_gradients(model, limit=1.):
    # Prevent a CE readout/last-eta gradient from rescaling cosine block updates.
    block_grads = [p.grad for b in model.layers for p in b.parameters() if p.grad is not None]
    if model.hidden_step_raw.grad is not None:
        block_grads.append(model.hidden_step_raw.grad[:-1])
    shared_grads = [p.grad for p in (model.output.weight, model.norm.weight) if p.grad is not None]
    last_grads = ([] if model.hidden_step_raw.grad is None else [model.hidden_step_raw.grad[-1:]])
    result = {}
    for name, grads in (("blocks", block_grads), ("readout", shared_grads), ("last_eta", last_grads)):
        norm = sum((g.float().square().sum() for g in grads), model.hidden_step_raw.new_zeros(())).sqrt()
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite " + name + " gradient")
        scale = (limit / (norm + 1e-6)).clamp(max=1.)
        for g in grads:
            g.mul_(scale)
        result["gradient_norm/" + name] = norm.item()
    return result
