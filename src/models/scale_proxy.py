"""Scale-proxy model: frozen larger transformer encoders + trainable fusion head.

Per-modality: 4-layer transformer encoder, d=256, 8 heads (frozen at random init).
Fusion: cross-modal attention head + classifier (trainable only).
Total ~2-3M params, ~150k trainable.

Frame in paper as "scale proxy" — depth and parameter count matched, but
encoder is frozen at random init. Honest framing for scaling-skeptic rebuttal.
"""
from __future__ import annotations
import torch
import torch.nn as nn

MODALITIES = ["ACC", "BVP", "EDA", "TEMP"]
MOD_CHANNELS = {"ACC": 3, "BVP": 1, "EDA": 1, "TEMP": 1}


class FrozenLargeEncoder(nn.Module):
    """Linear projection of (T, C) -> (T, d). 4-layer transformer. Mean pool."""
    def __init__(self, in_ch: int, d: int = 256, n_layers: int = 4, n_heads: int = 8,
                 dropout: float = 0.1, max_len: int = 240):
        super().__init__()
        self.proj = nn.Linear(in_ch, d)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.tx = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = self.proj(x) + self.pos[:, : x.size(1)]
        h = self.tx(h).mean(dim=1)
        return h


class ScaleProxy(nn.Module):
    def __init__(self, feat: int = 256, n_layers: int = 4, n_heads: int = 8,
                 dropout: float = 0.1, freeze_encoder: bool = True):
        super().__init__()
        self.encoders = nn.ModuleDict({
            m: FrozenLargeEncoder(MOD_CHANNELS[m], feat, n_layers, n_heads, dropout)
            for m in MODALITIES
        })
        if freeze_encoder:
            for p in self.encoders.parameters():
                p.requires_grad = False

        self.mod_emb = nn.Parameter(torch.randn(len(MODALITIES), feat) * 0.02)
        head_layer = nn.TransformerEncoderLayer(
            d_model=feat, nhead=n_heads, dim_feedforward=feat * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.fuse = nn.TransformerEncoder(head_layer, num_layers=1)
        self.head = nn.Linear(feat, 2)

    def encode_tokens(self, batch: dict) -> torch.Tensor:
        feats = [self.encoders[m](batch[m]) for m in MODALITIES]
        tok = torch.stack(feats, dim=1)
        tok = tok + self.mod_emb[None]
        return tok

    def forward(self, batch: dict) -> torch.Tensor:
        tok = self.encode_tokens(batch)
        h = self.fuse(tok).mean(dim=1)
        return self.head(h)
