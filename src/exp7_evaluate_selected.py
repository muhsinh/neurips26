"""Exp 7 follow-up: take the per-fold selected checkpoints (under each rule)
and run dropout (Exp 2) + degradation (Exp 3) evaluations on them.

For each selection rule (train-loss vs val-F1):
  - Aggregate selected ckpts into a synthetic "model name" so existing
    eval scripts can iterate.
  - Run eval_dropout style + eval_degradation style.

Output structure mimics the cross_attention.json results so the aggregator +
exp4_decision-style comparison works trivially.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, accuracy_score

from src.dataset import (WindowDataset, load_all_subjects, MODALITIES,
                         collate)
from src.models.cross_attention import CrossAttention
from src.utils import get_device, seed_all
from src.train import to_device
from src.eval_dropout import DROPOUT_CONDITIONS, apply_dropout
from src.eval_degradation import (NOISE_SIGMAS, REALISTIC, add_gaussian)

SEED = 42


def build_xattn(cfg: dict, dropout_override: float | None = None) -> CrossAttention:
    c = cfg["models"]["cross_attention"]
    do = dropout_override if dropout_override is not None else c["dropout"]
    return CrossAttention(c["feature_dim"], c["n_heads"], c["n_layers"], do)


@torch.no_grad()
def eval_dropout_one(model, loader, device, drop_mods) -> dict:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = to_device(batch, device)
        batch_d = apply_dropout(batch, drop_mods)
        logits = model(batch_d)
        ys.append(batch["label"].cpu().numpy())
        ps.append(logits.argmax(dim=1).cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return {
        "f1_stress": float(f1_score(y, p, pos_label=1, zero_division=0)),
        "acc": float(accuracy_score(y, p)),
    }


@torch.no_grad()
def eval_corruption_one(model, loader, device, corruption_fn) -> dict:
    model.eval()
    g = torch.Generator(device=device).manual_seed(0)
    ys, ps = [], []
    for batch in loader:
        batch = to_device(batch, device)
        batch_c = corruption_fn(batch, g)
        logits = model(batch_c)
        ys.append(batch["label"].cpu().numpy())
        ps.append(logits.argmax(dim=1).cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return {
        "f1_stress": float(f1_score(y, p, pos_label=1, zero_division=0)),
        "acc": float(accuracy_score(y, p)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--all_folds", default="results/exp7_erm_tuning/all_folds.json")
    p.add_argument("--out_dir", default="results/exp7_erm_tuning")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.out_dir)
    device = get_device(prefer_mps=False)
    folds_data = json.load(open(args.all_folds))["folds"]
    all_subj = load_all_subjects(cfg["dataset"]["processed_dir"])

    selection_rules = ["sel_train_loss_tag", "sel_val_f1_tag"]
    rule_names = {"sel_train_loss_tag": "train_loss",
                  "sel_val_f1_tag": "val_f1"}

    for rule in selection_rules:
        agg_dropout = {cond: {} for cond, _ in DROPOUT_CONDITIONS}
        agg_degradation = {"gaussian": {s: {} for s in NOISE_SIGMAS},
                           "realistic": {n: {} for n in REALISTIC.keys()}}
        loso_summary = []

        for fold in folds_data:
            test_subj = fold["test_subject"]
            sel_tag = fold[rule]
            sel_cfg = next(c for c in fold["configs"] if c["tag"] == sel_tag)
            ckpt = sel_cfg["ckpt"]
            do = sel_cfg["dropout"]

            seed_all(SEED)
            model = build_xattn(cfg, dropout_override=do).to(device)
            state = torch.load(ckpt, map_location=device, weights_only=True)
            model.load_state_dict(state)

            test_ds = WindowDataset([all_subj[test_subj]])
            loader = torch.utils.data.DataLoader(
                test_ds, batch_size=cfg["train"]["batch_size"],
                shuffle=False, collate_fn=collate)

            # Recompute clean test F1 (matches the saved value)
            ys = np.array(sel_cfg["test_y_true"])
            ps = np.array(sel_cfg["test_y_pred"])
            clean_f1 = float(f1_score(ys, ps, pos_label=1, zero_division=0))
            loso_summary.append({
                "test_subject": test_subj,
                "lr": sel_cfg["lr"], "wd": sel_cfg["wd"],
                "dropout": sel_cfg["dropout"],
                "test_f1": clean_f1,
            })

            for cond_name, drop_mods in DROPOUT_CONDITIONS:
                r = eval_dropout_one(model, loader, device, drop_mods)
                agg_dropout[cond_name].setdefault(test_subj, {})["42"] = r
            for sigma in NOISE_SIGMAS:
                r = eval_corruption_one(
                    model, loader, device,
                    lambda b, g, s=sigma: add_gaussian(b, s, g))
                agg_degradation["gaussian"][sigma].setdefault(test_subj, {})["42"] = r
            for cname, fn in REALISTIC.items():
                r = eval_corruption_one(model, loader, device, fn)
                agg_degradation["realistic"][cname].setdefault(test_subj, {})["42"] = r

        rule_name = rule_names[rule]
        with open(out_dir / f"dropout_{rule_name}.json", "w") as f:
            json.dump(agg_dropout, f, indent=2)
        with open(out_dir / f"degradation_{rule_name}.json", "w") as f:
            json.dump(agg_degradation, f, indent=2)
        with open(out_dir / f"loso_summary_{rule_name}.json", "w") as f:
            json.dump(loso_summary, f, indent=2)

        # Print headline numbers
        print(f"\n=== Selection rule: {rule_name} ===")
        clean_f1s = [l["test_f1"] for l in loso_summary]
        print(f"  LOSO clean F1: mean={np.mean(clean_f1s):.3f} ± {np.std(clean_f1s):.3f}")
        print(f"  collapsed (<0.3): {sum(1 for f in clean_f1s if f < 0.3)} / {len(clean_f1s)}")

        # Per-condition mean F1 across folds (with single seed)
        worst_drop = None
        for cond, _ in DROPOUT_CONDITIONS:
            f1s = [r["f1_stress"]
                   for s, sd_map in agg_dropout[cond].items()
                   for r in sd_map.values()]
            mean = float(np.mean(f1s))
            if cond == "all_clean":
                clean_drop = mean
            else:
                if worst_drop is None or mean < worst_drop[1]:
                    worst_drop = (cond, mean)
        print(f"  dropout: clean F1={clean_drop:.3f}, worst-drop {worst_drop[0]} F1={worst_drop[1]:.3f}, "
              f"Δ={(clean_drop - worst_drop[1])*100:.1f}pp")

        for sigma in NOISE_SIGMAS:
            f1s = [r["f1_stress"]
                   for s, sd_map in agg_degradation["gaussian"][sigma].items()
                   for r in sd_map.values()]
            print(f"  σ={sigma}: F1={np.mean(f1s):.3f}")


if __name__ == "__main__":
    main()
