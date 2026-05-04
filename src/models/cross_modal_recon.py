"""Cross-attention transformer with cross-modal reconstruction auxiliary loss.

Same architecture as cross_attention.py. During training, with prob 0.5 per batch,
one modality token is zeroed at the encoder output before fusion. The transformer
must regress the masked modality's feature vector from the surviving tokens.
Total loss = CE + 0.3 * MSE_reconstruction.

Why: instantiates "neuro-inspired generative cross-modal prior" — each modality
must be predictable from the others, breaking shortcut reliance on a single one.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .late_fusion_mlp import ModEncoder, MODALITIES

FEAT = 64
RECON_WEIGHT = 0.3
RECON_PROB = 0.5


class CrossModalRecon(nn.Module):
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
        # reconstruction head: project token to its predicted modality feat
        self.recon_head = nn.Linear(feat, feat)
        self.feat = feat

    def encode_tokens(self, batch: dict) -> torch.Tensor:
        feats = [self.encoders[m](batch[m]) for m in MODALITIES]
        tok = torch.stack(feats, dim=1)
        tok = tok + self.mod_emb[None]
        return tok

    def forward(self, batch: dict) -> torch.Tensor:
        tok = self.encode_tokens(batch)
        h = self.transformer(tok).mean(dim=1)
        return self.head(h)

    def predict_logits(self, batch: dict) -> torch.Tensor:
        return self.forward(batch)

    def training_step(self, batch: dict) -> torch.Tensor:
        tok = self.encode_tokens(batch)  # (B, M, F)
        B, M, F_dim = tok.shape

        # Random per-batch decision: with RECON_PROB mask one modality
        if torch.rand(()).item() < RECON_PROB:
            mask_idx = torch.randint(0, M, (1,)).item()
            target = tok[:, mask_idx, :].detach().clone()
            tok_in = tok.clone()
            tok_in[:, mask_idx, :] = 0.0
            out = self.transformer(tok_in)
            recon = self.recon_head(out[:, mask_idx, :])
            recon_loss = F.mse_loss(recon, target)
            cls_h = out.mean(dim=1)
            cls_logits = self.head(cls_h)
            cls_loss = F.cross_entropy(cls_logits, batch["label"])
            return cls_loss + RECON_WEIGHT * recon_loss

        # Pure classification step
        out = self.transformer(tok)
        cls_h = out.mean(dim=1)
        cls_logits = self.head(cls_h)
        return F.cross_entropy(cls_logits, batch["label"])
