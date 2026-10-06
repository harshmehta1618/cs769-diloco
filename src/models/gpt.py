"""
GPT-style Causal Language Model.

Architecture:
  - Token + learned positional embeddings
  - N transformer blocks (causal self-attention + MLP)
  - Weight-tied language-model head
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    seq_len: int = 512
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 0            # defaults to 4 * d_model if 0
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def n_params(self) -> int:
        """Approximate parameter count (excluding embedding tie)."""
        # Embeddings
        p = self.vocab_size * self.d_model + self.seq_len * self.d_model
        # Each transformer block
        attn = 4 * self.d_model * self.d_model + (4 * self.d_model if self.bias else 0)
        ff   = 2 * self.d_model * self.d_ff   + (self.d_model + self.d_ff if self.bias else 0)
        ln   = 2 * 2 * self.d_model           # two LayerNorms per block
        p   += self.n_layers * (attn + ff + ln)
        # Final LayerNorm
        p   += 2 * self.d_model
        return p


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional flash attention."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model,     bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # Causal mask (registered as buffer so it moves with the model)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).view(
                1, 1, cfg.seq_len, cfg.seq_len
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.cfg.d_model, dim=2)

        n_h, d_h = self.cfg.n_heads, self.cfg.d_head
        q = q.view(B, T, n_h, d_h).transpose(1, 2)
        k = k.view(B, T, n_h, d_h).transpose(1, 2)
        v = v.view(B, T, n_h, d_h).transpose(1, 2)

        # Scaled dot-product (or flash-attention when available)
        try:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.cfg.dropout if self.training else 0.0
            )
        except Exception:
            scale = 1.0 / math.sqrt(d_h)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = torch.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network (GELU activation)."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc   = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        self.act  = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.proj(self.act(self.fc(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp  = MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    GPT-style causal language model.

    Usage::

        cfg   = GPTConfig(n_layers=6, d_model=384, n_heads=6)
        model = GPT(cfg)
        logits, loss = model(input_ids, targets)
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len,    cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying (saves ~vocab_size * d_model params)
        self.tok_emb.weight = self.lm_head.weight

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """GPT-style init: N(0, 0.02) for embeddings, scaled for residuals."""
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        # Scale residual projections by 1/√(2 * n_layers)
        scale = (2 * self.cfg.n_layers) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(block.mlp.proj.weight,    mean=0.0, std=0.02 * scale)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        targets: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        B, T = input_ids.shape
        assert T <= self.cfg.seq_len, (
            f"Sequence length {T} exceeds model maximum {self.cfg.seq_len}"
        )
        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0)          # (1, T)

        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                   # (B, T, V)

        loss: Optional[Tensor] = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """Greedy / top-k sampling generation for quick sanity checks."""
        for _ in range(max_new_tokens):
            ctx = input_ids if input_ids.size(1) <= self.cfg.seq_len else input_ids[:, -self.cfg.seq_len :]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad if trainable_only else True)
        )


# ---------------------------------------------------------------------------
# Preset configs (matching the paper's scale table)
# ---------------------------------------------------------------------------

PRESET_CONFIGS: dict[str, GPTConfig] = {
    # Smoke / debug
    "debug": GPTConfig(n_layers=2, d_model=64,   n_heads=2, seq_len=128),
    # Pilot ~35-60M
    "60m":   GPTConfig(n_layers=8, d_model=512,  n_heads=8, seq_len=1024),
    # Small ~100-160M
    "160m":  GPTConfig(n_layers=12, d_model=768,  n_heads=12, seq_len=1024),
    # Medium ~300-410M
    "410m":  GPTConfig(n_layers=24, d_model=1024, n_heads=16, seq_len=1024),
    # Large confirmation ~1B
    "1b":    GPTConfig(n_layers=24, d_model=2048, n_heads=16, seq_len=2048),
}


def build_model(name_or_cfg: str | GPTConfig) -> GPT:
    """Construct a GPT model from a preset name or a GPTConfig."""
    if isinstance(name_or_cfg, str):
        cfg = PRESET_CONFIGS[name_or_cfg]
    else:
        cfg = name_or_cfg
    return GPT(cfg)
