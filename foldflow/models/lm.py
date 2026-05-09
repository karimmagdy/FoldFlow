"""FoldFlow Language Model — Energy-gated attention with Langevin refinement.

Also includes a minimal Transformer++ baseline for fair comparison.
Both models use the same architecture skeleton (embedding → blocks → LM head)
with identical hyperparameters (d_model, n_layers, vocab_size, etc.) so the
only difference is the FoldFlow-specific components.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class LMConfig:
    """Shared config for all LM variants."""
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    vocab_size: int = 50257
    max_seq_len: int = 256
    dropout: float = 0.1
    energy_steps: int = 3  # FoldFlow-only: Langevin refinement steps at inference
    use_energy_gate: bool = True
    use_chaperone: bool = True
    use_langevin: bool = True
    # 2026-05-04 (W4 / Tier B B2): selectable attention head-gating variant.
    # "energy"        -> EnergyAttention with causal cumulative mean (post-W1 fix)
    # "per-token"     -> PerTokenCausalGate (no pooling; current-position only)
    # "talking-heads" -> TalkingHeadsAttention (Shazeer et al. 2020)
    attention_variant: str = "energy"


# ---------------------------------------------------------------------------
# Transformer++ Baseline (fair comparison)
# ---------------------------------------------------------------------------

class TransformerPPBlock(nn.Module):
    """Standard pre-norm Transformer block with RoPE-style attention."""

    def __init__(self, config: LMConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model, config.n_heads,
            dropout=config.dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = x + self.attn(h, h, h, attn_mask=mask, is_causal=False)[0]
        h = h + self.ffn(self.norm2(h))
        return h


class TransformerPPLM(nn.Module):
    """Transformer++ baseline for WikiText-2 comparison."""

    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [TransformerPPBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # tie weights

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device)
        x = self.drop(self.embedding(input_ids) + self.pos_embedding(pos))

        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), 1).bool()

        for layer in self.layers:
            x = layer(x, mask)

        logits = self.lm_head(self.norm(x))
        loss = self._compute_loss(logits, labels) if labels is not None else None
        return logits, loss

    @staticmethod
    def _compute_loss(logits, labels):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )


# ---------------------------------------------------------------------------
# FoldFlow LM
# ---------------------------------------------------------------------------

class EnergyAttention(nn.Module):
    """Attention with per-head energy gating (CAUSALLY SAFE, post-W1 fix).

    History note (2026-05-04): the original implementation gated per head
    using ``x.mean(dim=1)`` -- a non-causal global average over the whole
    sequence -- which during teacher-forced training let the gate at every
    position depend on future tokens. The reviewer of the SR submission
    flagged this as W1 and the present implementation fixes it: the gate
    at position ``t`` uses only the CAUSAL CUMULATIVE MEAN of past tokens
    ``[0, ..., t]``, so future-token information cannot reach the gate.
    See the new "Causal energy gating (W1 fix)" subsection in the SR
    Methods for the formal statement.
    """

    def __init__(self, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Energy gate: context-dependent per-head modulation
        self.energy_gate = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, config.n_heads),
            nn.Sigmoid(),
        )
        self.use_energy_gate = config.use_energy_gate

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)

        scores = torch.einsum("blhd,bmhd->bhlm", q, k) * self.scale
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # Energy-modulated: each head gets a context-dependent gate.
        # CAUSAL: at position t, use the cumulative mean of x[:, 0..t, :]
        # so the gate cannot depend on future tokens.
        if self.use_energy_gate:
            cum_sum = x.cumsum(dim=1)                                          # (B, L, D)
            counts = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)
            causal_ctx = cum_sum / counts                                      # (B, L, D)
            energy_weights = self.energy_gate(causal_ctx)                      # (B, L, n_heads)
            # Reshape for broadcast against (B, n_heads, L_q, L_k):
            #   gate at query position t multiplies attn[:, :, t, :]
            gate = energy_weights.permute(0, 2, 1).unsqueeze(-1)               # (B, n_heads, L, 1)
            attn = attn * gate
        attn = self.dropout(attn)

        out = torch.einsum("bhlm,bmhd->blhd", attn, v).reshape(B, L, D)
        return self.out(out)


# ---------------------------------------------------------------------------
# 2026-05-04 (Tier B B2): two additional head-gating baselines per reviewer Q4
# ---------------------------------------------------------------------------

class PerTokenCausalGate(nn.Module):
    """Per-token head gating with NO pooling.

    Gate at position ``t`` is computed from x[:, t, :] alone -- no
    sequence pooling at all. This is the simplest causally-safe gate and
    serves as a baseline against EnergyAttention's cumulative-mean variant:
    any benefit of EnergyAttention over this baseline is attributable to
    the multi-token cumulative-mean structure rather than to per-head
    gating per se.
    """

    def __init__(self, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Same gate architecture as EnergyAttention but applied per-token
        # (no pooling) so it is trivially causal.
        self.head_gate = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, config.n_heads),
            nn.Sigmoid(),
        )
        # Honour the use_energy_gate switch so that --disable-energy-gate
        # also disables this variant for sanity-check ablations.
        self.use_energy_gate = config.use_energy_gate

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)

        scores = torch.einsum("blhd,bmhd->bhlm", q, k) * self.scale
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if self.use_energy_gate:
            gate = self.head_gate(x).permute(0, 2, 1).unsqueeze(-1)            # (B, n_heads, L, 1)
            attn = attn * gate
        attn = self.dropout(attn)

        out = torch.einsum("bhlm,bmhd->blhd", attn, v).reshape(B, L, D)
        return self.out(out)


class TalkingHeadsAttention(nn.Module):
    """Talking-Heads attention (Shazeer et al., 2020).

    Adds two learned linear projections across the head dimension --
    one before softmax (pre-talking) and one after softmax (post-talking) --
    that mix the per-head attention scores. This is the most-cited
    head-mixing baseline in the head-gating literature and the reviewer
    named it explicitly as a control comparison for energy-gated attention.

    Implementation: standard scaled-dot-product attention with two
    extra (n_heads, n_heads) linear projections on the head axis.
    """

    def __init__(self, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Pre- and post-softmax head-mixing projections (the talking step).
        self.pre_talking = nn.Linear(config.n_heads, config.n_heads, bias=False)
        self.post_talking = nn.Linear(config.n_heads, config.n_heads, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)

        scores = torch.einsum("blhd,bmhd->bhlm", q, k) * self.scale
        # Pre-softmax talking: mix scores across heads at each (l, m).
        # scores: (B, H, L, L) -> permute to (B, L, L, H) -> linear -> back
        scores_p = scores.permute(0, 2, 3, 1)
        scores_p = self.pre_talking(scores_p)
        scores = scores_p.permute(0, 3, 1, 2)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # Post-softmax talking: same mixing on the post-softmax weights.
        attn_p = attn.permute(0, 2, 3, 1)
        attn_p = self.post_talking(attn_p)
        attn = attn_p.permute(0, 3, 1, 2)
        attn = self.dropout(attn)

        out = torch.einsum("bhlm,bmhd->blhd", attn, v).reshape(B, L, D)
        return self.out(out)


# Attention-variant dispatch (Tier B B2).
_ATTN_VARIANTS = {
    "energy": EnergyAttention,
    "per-token": PerTokenCausalGate,
    "talking-heads": TalkingHeadsAttention,
}


def make_attention(config: LMConfig) -> nn.Module:
    """Factory: pick the attention class by config.attention_variant."""
    variant = getattr(config, "attention_variant", "energy")
    if variant not in _ATTN_VARIANTS:
        raise ValueError(
            f"Unknown attention_variant {variant!r}; "
            f"choose from {list(_ATTN_VARIANTS)}"
        )
    return _ATTN_VARIANTS[variant](config)


class FoldFlowLMBlock(nn.Module):
    """FoldFlow LM block: energy-gated attention + chaperone correction."""

    def __init__(self, config: LMConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.attn = make_attention(config)  # 2026-05-04: was EnergyAttention(config); now dispatch by config.attention_variant
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout),
        )
        # Chaperone: corrective signal based on before/after states
        self.chaperone = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
        )
        self.chap_scale = nn.Parameter(torch.tensor(0.1))
        self.use_chaperone = config.use_chaperone

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.norm1(x), mask)
        h_new = h + self.ffn(self.norm2(h))
        if not self.use_chaperone:
            return h_new
        correction = self.chaperone(torch.cat([h, h_new], dim=-1))
        return h_new + self.chap_scale * correction


class FoldFlowLM(nn.Module):
    """FoldFlow Language Model with energy-gated attention + Langevin refinement."""

    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [FoldFlowLMBlock(config) for _ in range(config.n_layers)]
        )

        # Energy head for Langevin refinement at inference
        self.energy_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )

        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.use_langevin = config.use_langevin

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device)
        x = self.drop(self.embedding(input_ids) + self.pos_embedding(pos))

        mask = torch.triu(torch.ones(L, L, device=x.device), 1).bool()

        for layer in self.layers:
            x = layer(x, mask)

        # Langevin refinement at inference (gradient-based energy minimization)
        if not self.training and self.use_langevin and self.config.energy_steps > 0:
            x_ref = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                for _ in range(self.config.energy_steps):
                    energy = self.energy_head(x_ref)
                    grad = torch.autograd.grad(
                        energy.sum(), x_ref, retain_graph=False
                    )[0]
                    x_ref = (x_ref.detach() - 0.01 * grad.detach()).requires_grad_(True)
            x = x_ref.detach()

        logits = self.lm_head(self.norm(x))
        loss = self._compute_loss(logits, labels) if labels is not None else None
        return logits, loss

    @staticmethod
    def _compute_loss(logits, labels):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
