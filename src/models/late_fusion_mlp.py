"""Late-fusion MLP. Per-modality 1D CNN encoder, concatenate, classify.

Total params: ~50k.
Input dict: {ACC: (B, 240, 3), BVP/EDA/TEMP: (B, 240, 1)}.
"""
from __future__ import annotations
import torch
import torch.nn as nn

MODALITIES = ["ACC", "BVP", "EDA", "TEMP"]
MOD_CHANNELS = {"ACC": 3, "BVP": 1, "EDA": 1, "TEMP": 1}
FEAT = 64


class ModEncoder(nn.Module):
    def __init__(self, in_ch: int, out_dim: int = FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, out_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        h = self.net(x).squeeze(-1)
        return h


class LateFusionMLP(nn.Module):
    def __init__(self, hidden: int = FEAT, fusion_hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.encoders = nn.ModuleDict({
            m: ModEncoder(MOD_CHANNELS[m], hidden) for m in MODALITIES
        })
        self.fuse = nn.Sequential(
            nn.Linear(hidden * len(MODALITIES), fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 2),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        feats = [self.encoders[m](batch[m]) for m in MODALITIES]
        h = torch.cat(feats, dim=1)
        return self.fuse(h)
