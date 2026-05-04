"""Exp 3: Sensor degradation robustness.

Per checkpoint: evaluate held-out subject under
- Gaussian noise on all modalities at sigma in {0.1, 0.5, 1.0, 2.0}
- Realistic per-modality corruptions:
  * ACC motion: 1Hz sine added (after standardisation, amp=1.0)
  * EDA drift: linear ramp (slope ramping by 1 std over the window)
  * BVP dropout: random 25% of samples zeroed
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, accuracy_score

from src.dataset import WindowDataset, load_all_subjects, MODALITIES, collate
from src.utils import get_device, seed_all
from src.train import build_model, to_device

NOISE_SIGMAS = [0.1, 0.5, 1.0, 2.0]


def add_gaussian(batch: dict, sigma: float, generator: torch.Generator) -> dict:
    out = {}
    for m in MODALITIES:
        x = batch[m]
        noise = torch.randn(x.shape, generator=generator, device=x.device) * sigma
        out[m] = x + noise
    out["label"] = batch["label"]
    return out


def acc_motion_artifact(batch: dict, generator: torch.Generator) -> dict:
    out = {m: batch[m].clone() for m in MODALITIES}
    x = out["ACC"]  # (B, T, 3)
    B, T, C = x.shape
    t = torch.linspace(0, T / 4.0, T, device=x.device)  # 4 Hz, T samples
    sine = torch.sin(2 * math.pi * 1.0 * t).view(1, T, 1)
    phase_shift = torch.rand((B, 1, C), generator=generator, device=x.device) * 2 * math.pi
    sine_b = torch.sin(2 * math.pi * 1.0 * t.view(1, T, 1) + phase_shift)
    out["ACC"] = x + sine_b * 1.0
    out["label"] = batch["label"]
    return out


def eda_drift(batch: dict, generator: torch.Generator) -> dict:
    out = {m: batch[m].clone() for m in MODALITIES}
    x = out["EDA"]  # (B, T, 1)
    B, T, _ = x.shape
    ramp = torch.linspace(0, 1.0, T, device=x.device).view(1, T, 1)
    sign = (torch.rand((B, 1, 1), generator=generator, device=x.device) > 0.5).float() * 2 - 1
    out["EDA"] = x + ramp * sign
    out["label"] = batch["label"]
    return out


def bvp_dropout(batch: dict, generator: torch.Generator) -> dict:
    out = {m: batch[m].clone() for m in MODALITIES}
    x = out["BVP"]  # (B, T, 1)
    mask = (torch.rand(x.shape, generator=generator, device=x.device) > 0.25).float()
    out["BVP"] = x * mask
    out["label"] = batch["label"]
    return out


REALISTIC = {
    "acc_motion": acc_motion_artifact,
    "eda_drift": eda_drift,
    "bvp_dropout": bvp_dropout,
}


@torch.no_grad()
def eval_corrupted(model, loader, device, corruption_fn) -> dict:
    model.eval()
    g = torch.Generator(device=device).manual_seed(0)
    ys, ps = [], []
    for batch in loader:
        batch = to_device(batch, device)
        batch_c = corruption_fn(batch, g)
        logits = model(batch_c)
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
    p.add_argument("--out_dir", default="results/exp3_degradation")
    p.add_argument("--models", default="late_fusion_mlp,cross_attention,scale_proxy")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = get_device(prefer_mps=False)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_subj = load_all_subjects(cfg["dataset"]["processed_dir"])

    models = args.models.split(",")
    aggregated = {m: {"gaussian": {s: {} for s in NOISE_SIGMAS},
                      "realistic": {n: {} for n in REALISTIC.keys()}}
                  for m in models}

    exp1_dir = Path(args.exp1_dir)
    for ckpt in sorted(exp1_dir.glob("*.pt")):
        tag = ckpt.stem
        try:
            model_name, subj, seed_part = tag.split("__")
        except ValueError:
            continue
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

        for sigma in NOISE_SIGMAS:
            r = eval_corrupted(m, loader, device, lambda b, g, s=sigma: add_gaussian(b, s, g))
            aggregated[model_name]["gaussian"][sigma].setdefault(subj, {})[seed] = r
        for cname, fn in REALISTIC.items():
            r = eval_corrupted(m, loader, device, fn)
            aggregated[model_name]["realistic"][cname].setdefault(subj, {})[seed] = r
        print(f"  evaluated {tag}")

    for model_name in models:
        out_path = out_dir / f"{model_name}.json"
        with open(out_path, "w") as f:
            json.dump(aggregated[model_name], f, indent=2)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
