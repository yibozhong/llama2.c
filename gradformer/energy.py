"""Causal conditional energies with exact, differentiable hidden gradients.

Each token's candidate z and causal context c are independent arguments of
E(z; c). The update uses the partial derivative in z, evaluated at z=h.
Context is held fixed ONLY for that partial derivative: final CE differentiates
through both arguments, including the attention that produced c.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from model import Attention, ModelArgs, RMSNorm, Transformer
from gradformer.core import hidden_step


class ConditionalEnergy(nn.Module):
    """E_t = sum_j a_j * SiLU((Wz RMS(z_t) + Wc c_t)_j).

    Signed a permits nonconvex energies. No sum/mean over tokens enters each
    token's gradient. The wider scalar MLP matches a standard SwiGLU block's
    parameter count: 2 * D * (4D) versus 3 * D * (8D/3).
    """
    def __init__(self, width, hidden_width, eps):
        super().__init__()
        self.z_proj = nn.Linear(width, hidden_width, bias=False)
        self.c_proj = nn.Linear(width, hidden_width, bias=False)
        self.amplitude = nn.Parameter(torch.empty(hidden_width))
        self.norm_weight = nn.Parameter(torch.ones(width))
        self.eps = eps
        nn.init.normal_(self.amplitude, std=.02)

    def normalized_candidate(self, z):
        dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
        z = z.to(dtype)
        inv = torch.rsqrt(z.square().mean(-1, keepdim=True) + self.eps)
        return z, inv, z * inv * self.norm_weight

    def energy(self, z, context):
        _, _, n = self.normalized_candidate(z)
        u = self.z_proj(n) + self.c_proj(context)
        dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
        return (F.silu(u.to(dtype)) * self.amplitude).sum(-1)

    def gradient(self, z, context):
        """Exact analytic dE/dz, not a learned vector or a surrogate backward.

        Ordinary differentiable ops expose all mixed second derivatives to the
        final CE backward. Only the scalar MLP needs those derivatives; causal
        attention is outside the inner derivative and uses its usual backward.
        """
        z, inv, n = self.normalized_candidate(z)
        u = (self.z_proj(n) + self.c_proj(context)).to(z.dtype)
        sig = torch.sigmoid(u)
        slope = self.amplitude * sig * (1 + u * (1 - sig))
        q = F.linear(slope, self.z_proj.weight.T).to(z.dtype) * self.norm_weight
        return inv * (q - z * inv.square() * (q * z).mean(-1, keepdim=True))

    def reference_gradient(self, z, context, create_graph=True):
        # A distinct child node isolates the partial derivative without
        # detaching context (or z) from the outer training graph.
        with torch.enable_grad():
            candidate = z.view_as(z)
            if not candidate.requires_grad:
                candidate.requires_grad_(True)
            energy = self.energy(candidate, context)
            return torch.autograd.grad(energy.sum(), candidate, create_graph=create_graph)[0]


class EnergyBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.context_norm = RMSNorm(args.dim, args.norm_eps)
        self.attention = Attention(args)
        self.condition_norm = RMSNorm(args.dim, args.norm_eps)
        self.potential = ConditionalEnergy(args.dim, 4 * args.dim, args.norm_eps)
        self.reference = False

    def forward(self, hidden, cos, sin):
        context = self.condition_norm(self.attention(self.context_norm(hidden), cos, sin))
        if self.reference:
            return self.potential.reference_gradient(hidden, context, create_graph=self.training)
        return self.potential.gradient(hidden, context)


class EnergyTransformer(Transformer):
    def __init__(self, args):
        if args.hidden_update not in ("sgd", "adam") or args.dropout != 0:
            raise ValueError("Energy pilot requires SGD/Adam and zero dropout")
        super().__init__(args)
        self.layers = nn.ModuleList([EnergyBlock(args) for _ in range(args.n_layers)])
        self.layers.apply(self._init_weights)
        for block in self.layers:
            nn.init.normal_(block.attention.wo.weight, std=.02/math.sqrt(2 * args.n_layers))

    def forward(self, tokens, targets=None, capture_gradients=False, return_loss=False):
        if capture_gradients:
            raise ValueError("This model uses actual energy gradients, not prediction supervision")
        h, _ = forward_hidden(self, tokens)
        h = self.norm(h)
        if return_loss:
            if targets is None:
                raise ValueError("Targets required for training loss")
            return self._training_loss(h, targets)
        logits = self.output(h if targets is not None else h[:, [-1], :])
        if targets is not None:
            self.last_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                             targets.reshape(-1), ignore_index=-1)
        return logits


def make_model(kind, *, depth=6, width=288, heads=6, vocab=32000, sequence=256,
               step_size=.01, device="cuda"):
    args = ModelArgs(n_layers=depth, dim=width, n_heads=heads, n_kv_heads=heads,
                     vocab_size=vocab, max_seq_len=sequence, multiple_of=32,
                     dropout=0., hidden_update="baseline" if kind == "baseline" else kind,
                     hidden_step_size=step_size, hidden_beta1=.9, hidden_beta2=.999, hidden_eps=.01)
    return (Transformer(args) if kind == "baseline" else EnergyTransformer(args)).to(device)


def forward_hidden(model, tokens, diagnostics=False):
    h, state, stats = model.tok_embeddings(tokens), None, []
    cos, sin = model.freqs_cos[:tokens.size(1)], model.freqs_sin[:tokens.size(1)]
    for i, block in enumerate(model.layers):
        if model.params.hidden_update == "baseline":
            new_h = block(h, cos, sin)
            g = h - new_h
        else:
            g = block(h, cos, sin)
            new_h, state = hidden_step(h, g, model.hidden_step_raw[i], state, i, model.params)
        if diagnostics:
            with torch.no_grad():
                hn = h.float().square().mean().sqrt()
                delta = (new_h-h).float().square().mean().sqrt()
                stats.append(torch.stack((hn, g.float().square().mean().sqrt(), delta,
                                          delta/hn.clamp_min(1e-12))))
        h = new_h
    return h, torch.stack(stats) if diagnostics else None


def enable_compile(model):
    # Compile the explicit gradient expression, not an autograd.grad call.
    # AOTAutograd can then use a normal first backward of this forward graph.
    for block in model.layers:
        block.forward = torch.compile(block.forward, fullgraph=True, dynamic=False)
