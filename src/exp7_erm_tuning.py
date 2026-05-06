"""Exp 7: Gulrajani-style ERM tuning sweep on cross-attention transformer.

12-config HP grid: 3 LRs × 2 weight decays × 2 dropouts.
For each LOSO fold, hold out one *additional* subject as validation.
Train all 12 configs on the remaining 13 subjects.
Report two selection rules:
  A) train-loss selection (matches original Exp 1 protocol)
  B) held-out-subject validation F1 selection (Gulrajani-proper)

Then evaluate the selected checkpoints on dropout (Exp 2 conditions) and
noise (Exp 3 conditions).

Validation subject choice: numerically-next subject in WESAD ordering, wrapping.
The numerical order is (S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13, S14, S15, S16, S17).
For test S2 → val S3, ..., for test S17 → val S2.
"""
from __future__ import annotations
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.dataset import (WindowDataset, load_all_subjects, MODALITIES,
                         collate)
from src.models.cross_attention import CrossAttention
from src.utils import get_device, seed_all, git_commit, count_params, majority_baseline_f1
from src.train import to_device

SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
            "S10", "S11", "S13", "S14", "S15", "S16", "S17"]
SEED = 42  # single seed for Tier-1 robustness check

LRS = [3e-4, 1e-3, 3e-3]
WDS = [1e-5, 1e-4]
DROPOUTS = [0.1, 0.3]
HP_GRID = list(itertools.product(LRS, WDS, DROPOUTS))


def val_subject_for(test_subj: str) -> str:
    idx = SUBJECTS.index(test_subj)
    return SUBJECTS[(idx + 1) % len(SUBJECTS)]


def build_xattn(feat: int, n_heads: int, n_layers: int, dropout: float) -> CrossAttention:
    return CrossAttention(feat, n_heads, n_layers, dropout)


def train_one_run(model: nn.Module, train_loader, opt, device,
                  max_epochs: int, patience: int) -> list:
    history = []
    best = float("inf"); counter = 0
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = to_device(batch, device)
            opt.zero_grad()
            logits = model(batch)
            loss = nn.functional.cross_entropy(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        mean = float(np.mean(losses))
        history.append({"epoch": epoch, "loss_mean": mean})
        if mean < best - 1e-4:
            best = mean; counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    return history


@torch.no_grad()
def eval_loader(model, loader, device) -> dict:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch)
        ps.append(logits.argmax(dim=1).cpu().numpy())
        ys.append(batch["label"].cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return {
        "f1_stress": float(f1_score(y, p, pos_label=1, zero_division=0)),
        "acc": float(accuracy_score(y, p)),
        "n": int(len(y)),
        "y_true": y.tolist(),
        "y_pred": p.tolist(),
        "confusion": confusion_matrix(y, p, labels=[0, 1]).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--out_dir", default="results/exp7_erm_tuning")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"; raw_dir.mkdir(exist_ok=True)
    sel_dir = out_dir / "selected"; sel_dir.mkdir(exist_ok=True)

    device = get_device(prefer_mps=False)
    all_subj = load_all_subjects(cfg["dataset"]["processed_dir"])

    feat = cfg["models"]["cross_attention"]["feature_dim"]
    n_heads = cfg["models"]["cross_attention"]["n_heads"]
    n_layers = cfg["models"]["cross_attention"]["n_layers"]
    bs = cfg["train"]["batch_size"]
    max_epochs = cfg["train"]["max_epochs"]
    patience = cfg["train"]["patience"]

    summary = {"folds": []}
    t_start = time.time()
    n_folds = len(SUBJECTS)
    n_configs = len(HP_GRID)
    total_runs = n_folds * n_configs

    runs_done = 0
    for fold_i, test_subj in enumerate(SUBJECTS):
        val_subj = val_subject_for(test_subj)
        train_subjects_ids = [s for s in SUBJECTS if s not in (test_subj, val_subj)]
        train_subjects = [all_subj[s] for s in train_subjects_ids]
        val_ds = WindowDataset([all_subj[val_subj]])
        test_ds = WindowDataset([all_subj[test_subj]])
        train_ds = WindowDataset(train_subjects)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=bs, shuffle=True, collate_fn=collate, num_workers=0)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=bs, shuffle=False, collate_fn=collate)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=bs, shuffle=False, collate_fn=collate)

        fold_log = {
            "test_subject": test_subj,
            "val_subject": val_subj,
            "configs": [],
        }

        for cfg_i, (lr, wd, do) in enumerate(HP_GRID):
            tag = f"fold{fold_i:02d}_test{test_subj}_val{val_subj}_lr{lr:.0e}_wd{wd:.0e}_do{do}"
            seed_all(SEED)
            model = build_xattn(feat, n_heads, n_layers, do).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            t0 = time.time()
            history = train_one_run(model, train_loader, opt, device, max_epochs, patience)
            train_time = time.time() - t0
            final_train_loss = history[-1]["loss_mean"]

            val_metrics = eval_loader(model, val_loader, device)
            test_metrics = eval_loader(model, test_loader, device)

            ckpt_path = raw_dir / f"{tag}.pt"
            torch.save(model.state_dict(), ckpt_path)

            fold_log["configs"].append({
                "lr": lr, "wd": wd, "dropout": do,
                "tag": tag,
                "train_loss": final_train_loss,
                "val_f1": val_metrics["f1_stress"],
                "test_f1": test_metrics["f1_stress"],
                "test_acc": test_metrics["acc"],
                "epochs": len(history),
                "ckpt": str(ckpt_path),
                "test_y_true": test_metrics["y_true"],
                "test_y_pred": test_metrics["y_pred"],
            })
            runs_done += 1
            elapsed = time.time() - t_start
            eta = elapsed / runs_done * (total_runs - runs_done)
            print(f"  [{runs_done}/{total_runs}] fold {fold_i+1}/{n_folds} "
                  f"cfg {cfg_i+1}/{n_configs} test={test_subj} val={val_subj} "
                  f"lr={lr:.0e} wd={wd:.0e} do={do}: "
                  f"train_loss={final_train_loss:.3f} val_f1={val_metrics['f1_stress']:.3f} "
                  f"test_f1={test_metrics['f1_stress']:.3f} t={train_time:.1f}s "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

        # Selection per fold:
        # A: minimum train_loss
        # B: maximum val_f1 (ties broken by minimum train_loss)
        sel_train_loss = min(fold_log["configs"], key=lambda c: c["train_loss"])
        sel_val_f1 = max(fold_log["configs"],
                         key=lambda c: (c["val_f1"], -c["train_loss"]))
        fold_log["sel_train_loss_tag"] = sel_train_loss["tag"]
        fold_log["sel_val_f1_tag"] = sel_val_f1["tag"]
        summary["folds"].append(fold_log)

    with open(out_dir / "all_folds.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. {total_runs} runs in {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()
