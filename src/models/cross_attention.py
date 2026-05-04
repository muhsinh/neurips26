"""Cross-modal self-attention transformer.

Per-modality CNN encoder -> token. 4 tokens + learnable modality embeddings.
2 layers of MHSA, 4 heads, 64-dim. Mean pool tokens -> classifier.
Total params: ~150k.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .late_fusion_mlp import ModEncoder, MODALITIES

FEAT = 64


class CrossAttention(nn.Module):
    def __init__(self, feat: int = FEAT, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.encoders = nn.ModuleDict({
            m: ModEncoder(3 if m == "ACC" else 1, feat) for m in MODALITIES
        })
        self.mod_emb = nn.Parameter(torch.randn(len(MODALITIES), feat) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=feat, nhead=n_heads, dim_feedforward=feat * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(feat, 2)
        self.feat = feat

    def encode_tokens(self, batch: dict) -> torch.Tensor:
        feats = [self.encoders[m](batch[m]) for m in MODALITIES]  # list of (B, F)
        tok = torch.stack(feats, dim=1)  # (B, M, F)
        tok = tok + self.mod_emb[None]
        return tok

    def forward(self, batch: dict) -> torch.Tensor:
        tok = self.encode_tokens(batch)
        h = self.transformer(tok).mean(dim=1)
        return self.head(h)
