"""Exp 2: Modality dropout robustness.

For each (model, fold, seed) checkpoint: evaluate on held-out subject under
11 dropout conditions (clean + each single drop + each pair drop). Modality dropout
implementation is zero-mask (D04).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, accuracy_score

from src.dataset import WindowDataset, load_all_subjects, MODALITIES, collate
from src.utils import get_device, seed_all
from src.train import build_model, to_device

DROPOUT_CONDITIONS = [
    ("all_clean", []),
    ("drop_ACC", ["ACC"]),
    ("drop_BVP", ["BVP"]),
    ("drop_EDA", ["EDA"]),
    ("drop_TEMP", ["TEMP"]),
    ("drop_ACC_BVP", ["ACC", "BVP"]),
    ("drop_ACC_EDA", ["ACC", "EDA"]),
    ("drop_ACC_TEMP", ["ACC", "TEMP"]),
    ("drop_BVP_EDA", ["BVP", "EDA"]),
    ("drop_BVP_TEMP", ["BVP", "TEMP"]),
    ("drop_EDA_TEMP", ["EDA", "TEMP"]),
]


def apply_dropout(batch: dict, drop_mods: list[str]) -> dict:
    out = {}
    for m in MODALITIES:
        x = batch[m]
        if m in drop_mods:
            out[m] = torch.zeros_like(x)
        else:
            out[m] = x
    out["label"] = batch["label"]
    return out


@torch.no_grad()
def eval_one(model, loader, device, drop_mods: list[str]) -> dict:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = to_device(batch, device)
        batch_d = apply_dropout(batch, drop_mods)
        logits = model(batch_d) if not hasattr(model, "predict_logits") else model.predict_logits(batch_d)
        preds = logits.argmax(dim=1).cpu().numpy()
        ys.append(batch["label"].cpu().numpy())
        ps.append(preds)
    y = np.concatenate(ys); p = np.concatenate(ps)
    return {
        "f1_stress": float(f1_score(y, p, pos_label=1, zero_division=0)),
        "acc": float(accuracy_score(y, p)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--exp1_dir", default="results/exp1_loso")
    p.add_argument("--out_dir", default="results/exp2_dropout")
    p.add_argument("--models", default="late_fusion_mlp,cross_attention,scale_proxy")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = get_device(prefer_mps=False)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_subj = load_all_subjects(cfg["dataset"]["processed_dir"])

    models = args.models.split(",")
    aggregated = {m: {cond: {} for cond, _ in DROPOUT_CONDITIONS} for m in models}

    exp1_dir = Path(args.exp1_dir)
    for ckpt in sorted(exp1_dir.glob("*.pt")):
        tag = ckpt.stem
        try:
            model_name, subj, seed_part = tag.split("__")
        except ValueError:
            print(f"skip malformed name: {tag}"); continue
        if model_name not in models:
            continue
        seed = int(seed_part.replace("seed", ""))
        seed_all(seed)

        m = build_model(model_name, cfg).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        m.load_state_dict(state)

        test_ds = WindowDataset([all_subj[subj]])
        loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg["train"]["batch_size"],
                                             shuffle=False, collate_fn=collate)

        for cond_name, drop_mods in DROPOUT_CONDITIONS:
            r = eval_one(m, loader, device, drop_mods)
            aggregated[model_name][cond_name].setdefault(subj, {})[seed] = r
        print(f"  evaluated {tag}")

    for model_name in models:
        out_path = out_dir / f"{model_name}.json"
        with open(out_path, "w") as f:
            json.dump(aggregated[model_name], f, indent=2)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
