"""Train one (model, fold, seed) on WESAD LOSO. Save metrics + state_dict.

Usage:
  python -m src.train --model late_fusion_mlp --test_subject S2 --seed 42 --config configs/base.yaml
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.dataset import WindowDataset, load_all_subjects, loso_split, collate, MODALITIES
from src.utils import get_device, seed_all, git_commit, count_params, majority_baseline_f1


def build_model(name: str, cfg: dict) -> nn.Module:
    if name == "late_fusion_mlp":
        from src.models.late_fusion_mlp import LateFusionMLP
        c = cfg["models"]["late_fusion_mlp"]
        return LateFusionMLP(c["hidden_per_mod"], c["fusion_hidden"], c["dropout"])
    if name == "cross_attention":
        from src.models.cross_attention import CrossAttention
        c = cfg["models"]["cross_attention"]
        return CrossAttention(c["feature_dim"], c["n_heads"], c["n_layers"], c["dropout"])
    if name == "scale_proxy":
        from src.models.scale_proxy import ScaleProxy
        c = cfg["models"]["scale_proxy"]
        return ScaleProxy(c["feature_dim"], c["n_layers"], c["n_heads"],
                          c["dropout"], c["freeze_encoder"])
    if name == "cross_modal_recon":
        from src.models.cross_modal_recon import CrossModalRecon
        c = cfg["models"]["cross_attention"]
        return CrossModalRecon(c["feature_dim"], c["n_heads"], c["n_layers"], c["dropout"])
    raise ValueError(f"unknown model {name}")


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for m in MODALITIES:
        out[m] = batch[m].to(device, non_blocking=True)
    out["label"] = batch["label"].to(device, non_blocking=True)
    return out


def train_one(model, loader, opt, device, epoch_idx, recon_loss=False) -> dict:
    model.train()
    losses = []
    for batch in loader:
        batch = to_device(batch, device)
        opt.zero_grad()
        if recon_loss:
            loss = model.training_step(batch)
        else:
            logits = model(batch)
            loss = nn.functional.cross_entropy(logits, batch["label"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.item()))
    return {"epoch": epoch_idx, "loss_mean": float(np.mean(losses))}


@torch.no_grad()
def eval_model(model, loader, device) -> dict:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch) if not hasattr(model, "predict_logits") else model.predict_logits(batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        ys.append(batch["label"].cpu().numpy())
        ps.append(preds)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return {
        "f1_stress": float(f1_score(y, p, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y, p, average="macro", zero_division=0)),
        "acc": float(accuracy_score(y, p)),
        "n": int(len(y)),
        "n_stress": int((y == 1).sum()),
        "confusion": confusion_matrix(y, p, labels=[0, 1]).tolist(),
        "y_true": y.tolist(),
        "y_pred": p.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--model", required=True,
                   choices=["late_fusion_mlp", "cross_attention", "scale_proxy", "cross_modal_recon"])
    p.add_argument("--test_subject", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--save_state", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed_all(args.seed)
    device = get_device(prefer_mps=False)  # CPU is fine for these sizes; MPS adds variance

    all_subj = load_all_subjects(cfg["dataset"]["processed_dir"])
    if args.test_subject not in all_subj:
        raise SystemExit(f"test_subject {args.test_subject} not in processed dir")

    train_subj, test_subj = loso_split(all_subj, args.test_subject)
    train_ds = WindowDataset(train_subj)
    test_ds = WindowDataset(test_subj)

    bs = cfg["train"]["batch_size"]
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True,
                                               collate_fn=collate, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False,
                                              collate_fn=collate, num_workers=0)

    model = build_model(args.model, cfg).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
    )

    best_loss = float("inf")
    patience_counter = 0
    epoch_hist = []
    t0 = time.time()
    recon = (args.model == "cross_modal_recon")
    for epoch in range(cfg["train"]["max_epochs"]):
        info = train_one(model, train_loader, opt, device, epoch, recon_loss=recon)
        epoch_hist.append(info)
        if info["loss_mean"] < best_loss - 1e-4:
            best_loss = info["loss_mean"]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["patience"]:
                break

    train_time = time.time() - t0
    test_metrics = eval_model(model, test_loader, device)
    baseline = majority_baseline_f1(test_subj[0].labels)

    out_dir = Path(args.out_dir or "results/exp1_loso")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}__{args.test_subject}__seed{args.seed}"
    payload = {
        "model": args.model,
        "test_subject": args.test_subject,
        "seed": args.seed,
        "git": git_commit(),
        "params_total": count_params(model),
        "params_trainable": count_params(model, trainable_only=True),
        "epochs_trained": len(epoch_hist),
        "train_loss_history": epoch_hist,
        "train_time_sec": float(train_time),
        "test_metrics": test_metrics,
        "majority_baseline": baseline,
        "config": cfg,
    }
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(payload, f, indent=2)

    if args.save_state:
        ckpt_path = out_dir / f"{tag}.pt"
        torch.save(model.state_dict(), ckpt_path)

    print(f"[{tag}] f1_stress={test_metrics['f1_stress']:.3f} "
          f"acc={test_metrics['acc']:.3f} maj_f1={baseline['f1']:.3f} "
          f"epochs={len(epoch_hist)} time={train_time:.1f}s")


if __name__ == "__main__":
    main()
