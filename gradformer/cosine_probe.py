"""Read-only, full optimizer-batch probes for detached cosine supervision.

The fixed-target measurement freezes target vectors and their validity mask.
It measures the actual AdamW update, including clipping, moments and decay.
Diagnostic forwards use the training precision and preserve torch RNG states.
"""
from contextlib import contextmanager
import time

import torch

from gradformer.core import prediction_loss
from gradformer.runtime import loss_forward


def directions(pred, target):
    p, g = pred.float(), target.detach().float()
    pn, gn = p.norm(dim=-1), g.norm(dim=-1)
    cosine = (p * g).sum(-1) / (pn.clamp_min(1e-8) * gn.clamp_min(1e-8))
    return cosine.detach(), gn > 1e-8


@contextmanager
def diagnostic_context(model):
    device = next(model.parameters()).device
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            yield


class CosineProbe:
    def __init__(self, model, balancer, refresh=False, teacher=None):
        self.model, self.balancer, self.refresh = model, balancer, refresh
        self.teacher = teacher
        self.batches, self.losses = [], []
        self.seconds = 0.

    def observe(self, x, y, inputs, predictions, targets, aux, metrics):
        begin = time.perf_counter()
        self.batches.append((x.detach().clone(), y.detach().clone(),
                             [g.detach().clone() for g in targets],
                             [directions(p, g)[0] for p, g in zip(predictions, targets)] if self.refresh else None,
                             [h.detach().clone() for h in inputs] if self.refresh else None))
        self.losses.append(aux)
        self.seconds += time.perf_counter() - begin

    @torch.no_grad()
    def fixed_loss(self):
        losses, cosines = [], []
        with diagnostic_context(self.model):
            for x, y, targets, _, _ in self.batches:
                _, _, predictions = loss_forward(self.model, "llama2", x, y, True)
                loss, metrics = prediction_loss(predictions, targets, alpha=0., beta=1.)
                losses.append(loss)
                cosines.append(metrics[:, 1])
        return torch.stack(losses).mean().item(), torch.stack(cosines).mean(0).tolist()

    def before_step(self):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        self.parameters_before = [p.detach().clone() for p in self.balancer.parameters]
        self.raw_gradients = [g.detach().clone() if g is not None else None
                              for g in self.balancer.aux_gradients]
        self.before, self.before_layers = self.fixed_loss()
        self.noop_error = abs(self.before - torch.stack(self.losses).mean().item())
        torch.cuda.synchronize()
        self.seconds += time.perf_counter() - begin

    def after_step(self):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        after, after_layers = self.fixed_loss()
        delta = after - self.before
        tolerance = max(1e-6, 5 * self.noop_error)
        result = {"probe/fixed_loss_before": self.before, "probe/fixed_loss_after": after,
                  "probe/fixed_loss_delta": delta, "probe/noop_error": self.noop_error,
                  "probe/decreased": int(delta < -tolerance),
                  "probe/flat": int(abs(delta) <= tolerance), "probe/increased": int(delta > tolerance)}
        with torch.no_grad():
            differences = [p - old for p, old in zip(self.balancer.parameters, self.parameters_before)]
            groups = [("all", range(len(differences)))] + [(f"layer_{i}", ids) for i, ids in enumerate(self.balancer.groups)]
            for name, indices in groups:
                indices = list(indices)
                zero = differences[0].new_zeros(())
                dot = sum(((self.raw_gradients[i] * differences[i]).sum() for i in indices
                           if self.raw_gradients[i] is not None), zero)
                gn = sum((self.raw_gradients[i].square().sum() for i in indices
                          if self.raw_gradients[i] is not None), zero).sqrt()
                dn = sum((differences[i].square().sum() for i in indices), zero).sqrt()
                result[f"probe/{name}/raw_cosine_grad_dot_update"] = dot.item()
                result[f"probe/{name}/grad_update_cosine"] = (dot / (gn * dn).clamp_min(1e-30)).item()
            for i, (a, b) in enumerate(zip(self.before_layers, after_layers)):
                result[f"probe/layer_{i}/fixed_cosine_before"] = a
                result[f"probe/layer_{i}/fixed_cosine_after"] = b
        if self.refresh:
            result.update(self.fresh_metrics())
        torch.cuda.synchronize()
        result["probe/seconds"] = self.seconds + time.perf_counter() - begin
        if not all(torch.isfinite(torch.tensor(v)).item() for v in result.values()):
            raise RuntimeError("Nonfinite cosine probe result")
        return result

    def fresh_metrics(self):
        # autograd.grad only targets activations, so it does not populate p.grad.
        rows = []
        with diagnostic_context(self.model):
            for x, y, old_targets, old_cosines, old_inputs in self.batches:
                ntp, inputs, predictions = loss_forward(self.model, "llama2", x, y, True)
                fresh = (torch.autograd.grad(ntp, inputs) if self.teacher is None
                         else self.teacher.targets(inputs, predictions, y))
                layer_rows = []
                for p, old, new, old_cos, h0, h1 in zip(predictions, old_targets, fresh, old_cosines, old_inputs, inputs):
                    fixed_cos, old_valid = directions(p, old)
                    fresh_cos, new_valid = directions(p, new)
                    target_cos, _ = directions(old, new)
                    common = old_valid & new_valid
                    count = common.sum().clamp_min(1)
                    def average(v):
                        return (v * common).sum() / count
                    layer_rows.append(torch.stack((average(old_cos), average(fixed_cos), average(fresh_cos),
                                                   average(target_cos), common.float().mean(),
                                                   new.detach().float().norm() / old.float().norm().clamp_min(1e-30),
                                                   (h1.detach().float() - h0.float()).norm() / h0.float().norm().clamp_min(1e-30))))
                rows.append(torch.stack(layer_rows).detach())
        result = {}
        names = ("before_common", "after_fixed_common", "after_fresh_common", "target_cosine",
                 "common_valid_fraction", "target_norm_ratio", "hidden_relative_change")
        for i, row in enumerate(torch.stack(rows).mean(0).tolist()):
            result.update({f"probe/layer_{i}/{name}": value for name, value in zip(names, row)})
        return result


def validation_metrics(model, batches, teacher=None):
    """Self-consistency on fixed validation inputs, using each model's own targets."""
    cosines, fractions = None, []
    teacher_cosines, teacher_fractions, agreements, teacher_losses = None, [], None, []
    with diagnostic_context(model):
        for x, y in batches:
            ntp, inputs, predictions = loss_forward(model, "llama2", x, y, True)
            targets = torch.autograd.grad(ntp, inputs)
            teacher_targets = teacher.targets(inputs, predictions, y) if teacher is not None else None
            if cosines is None:
                cosines = [[] for _ in predictions]
                if teacher is not None:
                    teacher_cosines = [[] for _ in predictions]
                    agreements = [[] for _ in predictions]
            valid_row = []
            teacher_valid_row = []
            for i, (p, g) in enumerate(zip(predictions, targets)):
                cosine, valid = directions(p, g)
                cosines[i].append(cosine[valid])
                valid_row.append(valid.float().mean())
                if teacher_targets is not None:
                    tc, tv = directions(p, teacher_targets[i])
                    teacher_cosines[i].append(tc[tv])
                    teacher_valid_row.append(tv.float().mean())
                    agreement, _ = directions(g, teacher_targets[i])
                    agreements[i].append(agreement[valid & tv])
            fractions.append(torch.stack(valid_row))
            if teacher is not None:
                teacher_fractions.append(torch.stack(teacher_valid_row))
                with torch.no_grad():
                    tl, _, _ = loss_forward(teacher.model, "llama2", x, y, False)
                teacher_losses.append(tl.detach())
    result, means = {}, []
    for i, values in enumerate(cosines):
        values = torch.cat(values)
        if not values.numel():
            raise RuntimeError("No valid validation cosine targets")
        mean = values.mean().item()
        means.append(mean)
        result[f"val_cosine/layer_{i}/mean"] = mean
        for name, v in zip(("p10", "p50", "p90"), values.quantile(values.new_tensor([.1, .5, .9])).tolist()):
            result[f"val_cosine/layer_{i}/{name}"] = v
        result[f"val_cosine/layer_{i}/valid_fraction"] = torch.stack(fractions).mean(0)[i].item()
    result["val_cosine/mean"] = sum(means) / len(means)
    result["val_cosine/early_layers_mean"] = sum(means[:-1]) / max(1, len(means) - 1)
    if teacher is not None:
        means = []
        for i, values in enumerate(teacher_cosines):
            values = torch.cat(values)
            common = torch.cat(agreements[i])
            if not values.numel() or not common.numel():
                raise RuntimeError("No valid EMA validation targets")
            mean = values.mean().item()
            means.append(mean)
            result[f"val_ema_cosine/layer_{i}/mean"] = mean
            for name, v in zip(("p10", "p50", "p90"), values.quantile(values.new_tensor([.1, .5, .9])).tolist()):
                result[f"val_ema_cosine/layer_{i}/{name}"] = v
            result[f"val_ema_cosine/layer_{i}/valid_fraction"] = torch.stack(teacher_fractions).mean(0)[i].item()
            result[f"val_ema/layer_{i}/target_agreement"] = common.mean().item()
        result["val_ema_cosine/mean"] = sum(means) / len(means)
        result["val_ema_cosine/early_layers_mean"] = sum(means[:-1]) / max(1, len(means) - 1)
        result["val_ema/ce"] = torch.stack(teacher_losses).mean().item()
    return result
