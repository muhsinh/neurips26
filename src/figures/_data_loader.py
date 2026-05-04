"""Helper: load aggregated experiment results, with a synthetic-fallback for scaffolding.

Each loader returns either (real_data_dict, "real") or (placeholder_dict, "placeholder").
Placeholder data lets us lock figure layouts before real numbers arrive.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
            "S10", "S11", "S13", "S14", "S15", "S16", "S17"]
MODELS = ["late_fusion_mlp", "cross_attention", "scale_proxy"]
DROPOUT_CONDS = [
    "all_clean", "drop_ACC", "drop_BVP", "drop_EDA", "drop_TEMP",
    "drop_ACC_BVP", "drop_ACC_EDA", "drop_ACC_TEMP",
    "drop_BVP_EDA", "drop_BVP_TEMP", "drop_EDA_TEMP",
]
NOISE_SIGMAS = [0.1, 0.5, 1.0, 2.0]
REALISTIC = ["acc_motion", "eda_drift", "bvp_dropout"]


def _synthetic_per_subject(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for m_idx, m in enumerate(MODELS):
        center = [0.78, 0.74, 0.70][m_idx]
        out[m] = {}
        for s in SUBJECTS:
            mean = float(np.clip(rng.normal(center, 0.18), 0.0, 1.0))
            if rng.random() < 0.20:
                mean = float(rng.uniform(0.0, 0.30))
            std = float(rng.uniform(0.02, 0.10))
            out[m][s] = {"f1_mean": mean, "f1_std": std,
                         "f1_seeds": [mean - std, mean, mean + std]}
    return out


def _synthetic_dropout(per_subj: dict) -> dict:
    rng = np.random.default_rng(1)
    out = {}
    for m in MODELS:
        clean = float(np.mean([per_subj[m][s]["f1_mean"] for s in SUBJECTS]))
        out[m] = {}
        for c in DROPOUT_CONDS:
            if c == "all_clean":
                f1 = clean
            elif "EDA" in c:
                f1 = max(0.05, clean - rng.uniform(0.30, 0.50))
            elif "BVP" in c:
                f1 = max(0.10, clean - rng.uniform(0.10, 0.25))
            else:
                f1 = max(0.20, clean - rng.uniform(0.02, 0.10))
            out[m][c] = {"f1_mean": f1, "f1_std": float(rng.uniform(0.03, 0.08)),
                         "n_evals": 45}
    return out


def _synthetic_degradation(per_subj: dict) -> dict:
    rng = np.random.default_rng(2)
    out = {}
    for m in MODELS:
        clean = float(np.mean([per_subj[m][s]["f1_mean"] for s in SUBJECTS]))
        out[m] = {"gaussian": {}, "realistic": {}}
        for sigma in NOISE_SIGMAS:
            decay = 1.0 / (1.0 + sigma * 0.8)
            f1 = clean * decay + rng.normal(0, 0.02)
            out[m]["gaussian"][str(sigma)] = {
                "f1_mean": float(np.clip(f1, 0.0, 1.0)),
                "f1_std": float(rng.uniform(0.04, 0.10)),
                "n_evals": 45,
            }
        for cname in REALISTIC:
            f1 = clean - rng.uniform(0.10, 0.25)
            out[m]["realistic"][cname] = {
                "f1_mean": float(np.clip(f1, 0.0, 1.0)),
                "f1_std": float(rng.uniform(0.04, 0.10)),
                "n_evals": 45,
            }
    return out


def load_exp1(agg_dir: Path | str = "results/agg") -> tuple[dict, dict, str]:
    p1 = Path(agg_dir) / "exp1_per_subject.json"
    p2 = Path(agg_dir) / "exp1_overall.json"
    if p1.exists() and p2.exists():
        per_subj = json.load(open(p1))
        overall = json.load(open(p2))
        # Real data exists if at least one subject in one model has a non-None mean
        any_real = any(per_subj[m][s]["f1_mean"] is not None
                       for m in per_subj for s in per_subj[m])
        if any_real:
            return per_subj, overall, "real"
    per_subj = _synthetic_per_subject()
    overall = {m: {
        "f1_mean_over_subjects": float(np.mean([per_subj[m][s]["f1_mean"] for s in SUBJECTS])),
        "f1_std_over_subjects": float(np.std([per_subj[m][s]["f1_mean"] for s in SUBJECTS])),
        "n_collapsed_below_0.3": int(sum(1 for s in SUBJECTS if per_subj[m][s]["f1_mean"] < 0.3)),
        "n_subjects": 15,
    } for m in MODELS}
    return per_subj, overall, "placeholder"


def load_exp2(per_subj: dict, agg_dir: Path | str = "results/agg") -> tuple[dict, str]:
    p = Path(agg_dir) / "exp2_dropout.json"
    if p.exists():
        d = json.load(open(p))
        if any(d.get(m) for m in MODELS):
            return d, "real"
    return _synthetic_dropout(per_subj), "placeholder"


def load_exp3(per_subj: dict, agg_dir: Path | str = "results/agg") -> tuple[dict, str]:
    p = Path(agg_dir) / "exp3_degradation.json"
    if p.exists():
        d = json.load(open(p))
        if any(d.get(m) for m in MODELS):
            return d, "real"
    return _synthetic_degradation(per_subj), "placeholder"
