"""Scale-proxy model: frozen larger transformer encoders + trainable fusion head.

Per-modality: 3-layer transformer encoder, d=128, 4 heads (frozen at random init).
Fusion: 1-layer self-attention head + classifier (trainable only).
Total ~1.8M params, ~130k trainable.

Frame in paper as "scale proxy" — depth and parameter count matched, but
encoder is frozen at random init.

Frozen encoders mean we can pre-compute per-modality token features once per
(subject, seed) pair and skip the encoder entirely during head training.
This makes the per-fold training cost dominated by the small fusion head.

Use `precompute_subject_features(model, subject_dataset)` to obtain a
`(n_windows, M, feat)` token tensor. Then `model.head_forward(tokens)` trains
only the fusion head + classifier on cached tokens.
"""
from __future__ import annotations
import torch
import torch.nn as nn

MODALITIES = ["ACC", "BVP", "EDA", "TEMP"]
MOD_CHANNELS = {"ACC": 3, "BVP": 1, "EDA": 1, "TEMP": 1}


class FrozenLargeEncoder(nn.Module):
    def __init__(self, in_ch: int, d: int = 128, n_layers: int = 3, n_heads: int = 4,
                 dropout: float = 0.0, max_len: int = 240):
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
        with torch.no_grad():
            h = self.proj(x) + self.pos[:, : x.size(1)]
            h = self.tx(h).mean(dim=1)
        return h.detach()


class ScaleProxy(nn.Module):
    def __init__(self, feat: int = 128, n_layers: int = 3, n_heads: int = 4,
                 dropout: float = 0.1, freeze_encoder: bool = True):
        super().__init__()
        self.feat = feat
        self.encoders = nn.ModuleDict({
            m: FrozenLargeEncoder(MOD_CHANNELS[m], feat, n_layers, n_heads, dropout)
            for m in MODALITIES
        })
        if freeze_encoder:
            for p in self.encoders.parameters():
                p.requires_grad = False
            self.encoders.eval()

        self.mod_emb = nn.Parameter(torch.randn(len(MODALITIES), feat) * 0.02)
        head_layer = nn.TransformerEncoderLayer(
            d_model=feat, nhead=n_heads, dim_feedforward=feat * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.fuse = nn.TransformerEncoder(head_layer, num_layers=1)
        self.head = nn.Linear(feat, 2)

    def encode_tokens(self, batch: dict) -> torch.Tensor:
        feats = [self.encoders[m](batch[m]) for m in MODALITIES]
        tok = torch.stack(feats, dim=1)  # (B, M, F)
        return tok

    def head_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, M, F) — already encoded. Trainable fusion head + classifier."""
        h = tokens + self.mod_emb[None]
        h = self.fuse(h).mean(dim=1)
        return self.head(h)

    def forward(self, batch: dict) -> torch.Tensor:
        tok = self.encode_tokens(batch)
        return self.head_forward(tok)
