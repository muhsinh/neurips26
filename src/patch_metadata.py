"""Retroactively patch aggregate result JSONs (Exp 2/3/5/6/7) with provenance
metadata: seed, architecture (where applicable), config, git_commit.

Per-run JSONs (Exp 1, Exp 4) already contain this metadata.

Adds a `_metadata` block at top level so existing consumers don't break.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

import yaml


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "uncommitted"


def patch(path: Path, metadata: dict) -> None:
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, list):
        d = {"_metadata": metadata, "results": d}
    elif isinstance(d, dict):
        d["_metadata"] = metadata
    else:
        print(f"  SKIP {path}: unexpected top-level type {type(d).__name__}")
        return
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"  patched {path}")


def main() -> None:
    cfg = yaml.safe_load(open("configs/base.yaml"))
    commit = git_commit()
    base = {
        "git_commit": commit,
        "config_hyperparameters": cfg,
        "seeds_used": cfg.get("seeds", [42, 1337, 2024]),
    }

    # Exp 2 dropout: per-architecture aggregate
    for f in sorted(Path("results/exp2_dropout").glob("*.json")):
        meta = dict(base)
        meta["experiment"] = "exp2_dropout"
        meta["architecture"] = f.stem
        meta["selection_rule"] = "zero_mask_test_time_dropout"
        patch(f, meta)

    # Exp 3 degradation
    for f in sorted(Path("results/exp3_degradation").glob("*.json")):
        meta = dict(base)
        meta["experiment"] = "exp3_degradation"
        meta["architecture"] = f.stem
        meta["corruption_protocol"] = "gaussian_noise + realistic_per_modality"
        patch(f, meta)

    # Exp 5 modality consistency
    for f in sorted(Path("results/exp5_modality_consistency").glob("*.json")):
        meta = dict(base)
        meta["experiment"] = "exp5_modality_consistency"
        meta["analysis_seed"] = 42
        meta["bootstrap_n"] = 1000
        meta["permutation_n"] = 10000
        meta["architectures_compared"] = [
            "late_fusion_mlp", "cross_attention", "scale_proxy",
        ]
        patch(f, meta)

    # Exp 6 bootstrap CIs
    for f in sorted(Path("results/exp6_bootstrap_ci").glob("*.json")):
        meta = dict(base)
        meta["experiment"] = "exp6_bootstrap_ci"
        meta["analysis_seed_formula"] = "42 + training_seed"
        meta["bootstrap_n"] = 1000
        meta["architectures_evaluated"] = [
            "late_fusion_mlp", "cross_attention", "scale_proxy",
        ]
        patch(f, meta)

    # Exp 7 ERM tuning
    for f in sorted(Path("results/exp7_erm_tuning").glob("*.json")):
        meta = dict(base)
        meta["experiment"] = "exp7_erm_tuning"
        meta["architecture"] = "cross_attention"
        meta["seed"] = 42
        meta["hp_grid"] = {
            "lr": [3e-4, 1e-3, 3e-3],
            "weight_decay": [1e-5, 1e-4],
            "dropout": [0.1, 0.3],
        }
        meta["selection_rules"] = ["train_loss", "val_f1"]
        meta["validation_protocol"] = (
            "held-out subject, numerically-next in WESAD order, wraparound"
        )
        patch(f, meta)


if __name__ == "__main__":
    main()
