"""EMA suffix gradients at the student's current augmented hidden state.

Each suffix INCLUDES the layer being supervised. This preserves the original
input-gradient definition; it is not the proposed skip-layer/counterfactual
target. Incoming hidden-Adam moments are held constant when differentiating
with respect to that layer's hidden input, as in the original computation graph.
"""
import copy
import math

import torch

from gradformer.core import hidden_step


class EMATeacher:
    def __init__(self, student, decay=.99):
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay must be finite and in [0, 1)")
        if student.params.hidden_update not in ("sgd", "adam") or student.params.dropout != 0:
            raise ValueError("EMA pilot requires llama2 predicted updates without dropout")
        # Avoid copying the transient nonleaf last_loss cached by llama2 forward.
        cache = student.last_loss
        try:
            student.last_loss = None
            self.model = copy.deepcopy(student)
        finally:
            student.last_loss = cache
        self.model.requires_grad_(False)
        self.model.zero_grad(set_to_none=True)
        self.model.eval()
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, student):
        teachers, students = dict(self.model.named_parameters()), dict(student.named_parameters())
        if teachers.keys() != students.keys():
            raise ValueError("Teacher/student parameter structures differ")
        for name, target in teachers.items():
            # named_parameters deduplicates the tied embedding/output matrix.
            if self.decay == 0:
                target.copy_(students[name])
            else:
                target.mul_(self.decay).add_(students[name], alpha=1 - self.decay)
        for name, target in self.model.named_buffers():
            target.copy_(dict(student.named_buffers())[name])
        self.updates += 1

    def suffix_loss(self, z, layer, y, incoming_state):
        model = self.model
        h, state = z, incoming_state
        cos, sin = model.freqs_cos[:z.shape[1]], model.freqs_sin[:z.shape[1]]
        for i in range(layer, len(model.layers)):
            predicted = -model.layers[i](h, cos, sin, return_delta=True)
            h, state = hidden_step(h, predicted, model.hidden_step_raw[i], state, i, model.params)
        return model._training_loss(model.norm(h), y)

    @torch.enable_grad()
    def targets(self, inputs, predictions, y):
        if len(inputs) != len(self.model.layers) or len(predictions) != len(inputs):
            raise ValueError("Need every student layer input and prediction")
        targets, state = [], None
        for layer, (h, predicted) in enumerate(zip(inputs, predictions)):
            z = h.detach().requires_grad_(True)
            loss = self.suffix_loss(z, layer, y, state)
            target, = torch.autograd.grad(loss, z, create_graph=False)
            targets.append(target.detach())
            # Reconstruct the STUDENT incoming moments for the next layer.
            # This uses the same hidden_step implementation and detached raw
            # predictions, and does not connect target gradients to the student.
            with torch.no_grad():
                _, state = hidden_step(h.detach(), predicted.detach(),
                                       self.model.hidden_step_raw[layer], state, layer, self.model.params)
        return targets

    def state_dict(self):
        return dict(model=self.model.state_dict(), decay=self.decay, updates=self.updates)

    def load_state_dict(self, state):
        if state["decay"] != self.decay:
            raise ValueError("Cannot change EMA decay on resume")
        self.model.load_state_dict(state["model"])
        self.updates = state["updates"]
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def metrics(self, student):
        pairs = list(zip(self.model.parameters(), student.parameters()))
        distance = sum((a - b).square().sum() for a, b in pairs).sqrt()
        norm = sum(b.square().sum() for _, b in pairs).sqrt().clamp_min(1e-30)
        return {"ema/decay": self.decay, "ema/updates": self.updates,
                "ema/relative_parameter_distance": (distance / norm).item()}
