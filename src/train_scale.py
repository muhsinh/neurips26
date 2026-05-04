"""Specialized fast trainer for scale_proxy: pre-compute frozen-encoder tokens
once per (subject, seed) pair, then train the fusion head on cached tokens.

Per-fold runtime drops from ~2 hours to ~5 seconds, since the bottleneck (the
1.8M-param frozen encoder) runs ~519 forwards total instead of ~30k.
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

from src.dataset import (WindowDataset, load_all_subjects, MODALITIES,
                         collate, SubjectWindows)
from src.models.scale_proxy import ScaleProxy
from src.utils import get_device, seed_all, git_commit, count_params, majority_baseline_f1


SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
            "S10", "S11", "S13", "S14", "S15", "S16", "S17"]


def precompute_subject_tokens(model: ScaleProxy, subject: SubjectWindows,
                              device: torch.device, batch_size: int = 64):
    """Run encoder once on subject's windows. Return (n_windows, M, feat)."""
    model.eval()
    ds = WindowDataset([subject])
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         collate_fn=collate)
    out_tok = []
    out_lab = []
    with torch.no_grad():
        for batch in loader:
            d = {m: batch[m].to(device) for m in MODALITIES}
            tok = model.encode_tokens(d)
            out_tok.append(tok.cpu())
            out_lab.append(batch["label"])
    return torch.cat(out_tok, dim=0), torch.cat(out_lab, dim=0)


def train_head_on_cached(model: ScaleProxy, train_tokens: torch.Tensor,
                          train_labels: torch.Tensor, cfg: dict,
                          device: torch.device) -> list:
    """Train the fusion head on cached tokens. Early stop on training loss plateau."""
    model.train()
    n = len(train_labels)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
    )
    bs = cfg["train"]["batch_size"]
    history = []
    best = float("inf")
    counter = 0
    for epoch in range(cfg["train"]["max_epochs"]):
        perm = torch.randperm(n)
        epoch_losses = []
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            tok = train_tokens[idx].to(device)
            y = train_labels[idx].to(device)
            opt.zero_grad()
            logits = model.head_forward(tok)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_losses.append(float(loss.item()))
        mean_loss = float(np.mean(epoch_losses))
        history.append({"epoch": epoch, "loss_mean": mean_loss})
        if mean_loss < best - 1e-4:
            best = mean_loss; counter = 0
        else:
            counter += 1
            if counter >= cfg["train"]["patience"]:
                break
    return history


@torch.no_grad()
def eval_head_on_cached(model: ScaleProxy, tokens: torch.Tensor,
                        labels: torch.Tensor, device: torch.device) -> dict:
    model.eval()
    bs = 64
    ys, ps = [], []
    for i in range(0, len(labels), bs):
        tok = tokens[i:i + bs].to(device)
        logits = model.head_forward(tok)
        ps.append(logits.argmax(dim=1).cpu().numpy())
        ys.append(labels[i:i + bs].numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
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
    p.add_argument("--seeds", default="42,1337,2024")
    p.add_argument("--subjects", default=",".join(SUBJECTS))
    p.add_argument("--out_dir", default="results/exp1_loso")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seeds = [int(s) for s in args.seeds.split(",")]
    subjects = args.subjects.split(",")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(prefer_mps=False)
    all_subj = load_all_subjects(cfg["dataset"]["processed_dir"])

    t_start = time.time()
    runs_done = 0

    for seed in seeds:
        seed_all(seed)
        # Build encoder once per seed
        model = ScaleProxy(
            cfg["models"]["scale_proxy"]["feature_dim"],
            cfg["models"]["scale_proxy"]["n_layers"],
            cfg["models"]["scale_proxy"]["n_heads"],
            cfg["models"]["scale_proxy"]["dropout"],
            freeze_encoder=True,
        ).to(device)

        # Precompute tokens for every subject under this seed's encoder
        t0 = time.time()
        cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for sid in subjects:
            cache[sid] = precompute_subject_tokens(model, all_subj[sid], device,
                                                   batch_size=cfg["train"]["batch_size"])
        print(f"[seed {seed}] tokens precomputed in {time.time()-t0:.1f}s "
              f"({sum(t.shape[0] for t, _ in cache.values())} total windows)")

        for test_subject in subjects:
            tag = f"scale_proxy__{test_subject}__seed{seed}"
            json_path = out_dir / f"{tag}.json"
            ckpt_path = out_dir / f"{tag}.pt"
            if json_path.exists() and ckpt_path.exists():
                print(f"  {tag} -- skip"); continue

            seed_all(seed)  # re-seed for head init
            model = ScaleProxy(
                cfg["models"]["scale_proxy"]["feature_dim"],
                cfg["models"]["scale_proxy"]["n_layers"],
                cfg["models"]["scale_proxy"]["n_heads"],
                cfg["models"]["scale_proxy"]["dropout"],
                freeze_encoder=True,
            ).to(device)
            # IMPORTANT: rebuild causes new random encoder. Restore cached encoder weights
            # by RE-USING the same encoder we used for precompute. Simpler: keep tokens
            # but reuse the previous model object (just re-init the head).
            # To avoid that complexity, the head is re-initialised by seed_all + random
            # head linear layers; encoder weights vary by seed but tokens were also
            # produced under the same seed, so consistent.

            train_tokens = torch.cat([cache[s][0] for s in subjects if s != test_subject], dim=0)
            train_labels = torch.cat([cache[s][1] for s in subjects if s != test_subject], dim=0)
            test_tokens, test_labels = cache[test_subject]

            t0 = time.time()
            history = train_head_on_cached(model, train_tokens, train_labels, cfg, device)
            train_time = time.time() - t0
            test_metrics = eval_head_on_cached(model, test_tokens, test_labels, device)
            baseline = majority_baseline_f1(test_labels.numpy())

            payload = {
                "model": "scale_proxy",
                "test_subject": test_subject,
                "seed": seed,
                "git": git_commit(),
                "params_total": count_params(model),
                "params_trainable": count_params(model, trainable_only=True),
                "epochs_trained": len(history),
                "train_loss_history": history,
                "train_time_sec": float(train_time),
                "test_metrics": test_metrics,
                "majority_baseline": baseline,
                "config": cfg,
            }
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [{tag}] f1_stress={test_metrics['f1_stress']:.3f} "
                  f"acc={test_metrics['acc']:.3f} epochs={len(history)} "
                  f"time={train_time:.1f}s")
            runs_done += 1

    print(f"\nDone. {runs_done} scale_proxy runs in {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()
