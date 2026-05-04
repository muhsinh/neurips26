"""Common utilities: device, seeding, checkpoint paths."""
from __future__ import annotations
import os
import random
import subprocess
from pathlib import Path
import numpy as np
import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return "uncommitted"


def count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def majority_baseline_f1(labels) -> dict:
    """F1 if you always predict majority class. Reference for collapse."""
    from sklearn.metrics import f1_score, accuracy_score
    labels = np.asarray(labels)
    if len(labels) == 0:
        return {"f1": 0.0, "acc": 0.0, "pred_class": -1}
    counts = np.bincount(labels, minlength=2)
    pred_class = int(np.argmax(counts))
    preds = np.full_like(labels, pred_class)
    return {
        "f1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "acc": float(accuracy_score(labels, preds)),
        "pred_class": pred_class,
    }
