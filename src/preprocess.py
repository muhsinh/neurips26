"""WESAD preprocessing.

Stage 1: Load raw .pkl per subject (encoding='latin1').
Stage 2: Resample wrist modalities {ACC, BVP, EDA, TEMP} to 4 Hz.
Stage 3: 60s non-overlap windows. Drop transition windows. Binary label.
Stage 4: Per-subject z-score (computed on the subject's own full recording).
Stage 5: Cache as .npz per subject.

Why per-subject z-score (Decision D02): each subject normalised on own stats. No leakage of labels.
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly

SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
            "S10", "S11", "S13", "S14", "S15", "S16", "S17"]

WRIST_RATES = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
LABEL_RATE = 700
TARGET_RATE = 4
WINDOW_SEC = 60
WINDOW_LEN = TARGET_RATE * WINDOW_SEC

LABEL_STRESS = {2}
LABEL_NON_STRESS = {1, 3}


def load_subject(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def resample_to_target(signal: np.ndarray, src_hz: int, tgt_hz: int) -> np.ndarray:
    """Resample (T, C) to target rate via polyphase filter."""
    if signal.ndim == 1:
        signal = signal[:, None]
    if src_hz == tgt_hz:
        return signal.astype(np.float32)
    from math import gcd
    g = gcd(src_hz, tgt_hz)
    up = tgt_hz // g
    down = src_hz // g
    out = resample_poly(signal, up, down, axis=0)
    return out.astype(np.float32)


def downsample_labels(labels: np.ndarray, src_hz: int, tgt_hz: int) -> np.ndarray:
    """Decimate labels by majority within each output sample."""
    factor = src_hz // tgt_hz
    n = len(labels) // factor
    trimmed = labels[: n * factor].reshape(n, factor)
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        vals, counts = np.unique(trimmed[i], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def make_windows(modalities: dict, labels: np.ndarray) -> tuple[dict, np.ndarray]:
    """Cut into 60s non-overlap windows. Drop windows with mixed/invalid labels.

    Returns dict {mod: (n_windows, WINDOW_LEN, channels)}, label_vec (n_windows,).
    """
    n_total = min(len(labels), min(m.shape[0] for m in modalities.values()))
    n_windows = n_total // WINDOW_LEN
    bin_labels = []
    keep_idx = []
    for i in range(n_windows):
        s, e = i * WINDOW_LEN, (i + 1) * WINDOW_LEN
        win = labels[s:e]
        vals, counts = np.unique(win, return_counts=True)
        majority = vals[np.argmax(counts)]
        purity = counts.max() / counts.sum()
        if purity < 0.95:
            continue
        if int(majority) in LABEL_STRESS:
            bin_labels.append(1)
            keep_idx.append(i)
        elif int(majority) in LABEL_NON_STRESS:
            bin_labels.append(0)
            keep_idx.append(i)
    if not keep_idx:
        return {}, np.array([], dtype=np.int64)
    keep_idx = np.array(keep_idx)
    out = {}
    for mod, sig in modalities.items():
        windows = np.stack([sig[i * WINDOW_LEN:(i + 1) * WINDOW_LEN] for i in keep_idx], axis=0)
        out[mod] = windows.astype(np.float32)
    return out, np.array(bin_labels, dtype=np.int64)


def zscore_per_subject(modalities: dict) -> dict:
    """Standardize each modality across the entire subject's recording."""
    out = {}
    for mod, sig in modalities.items():
        flat = sig.reshape(-1, sig.shape[-1])
        mu = flat.mean(axis=0, keepdims=True)
        sigma = flat.std(axis=0, keepdims=True) + 1e-6
        out[mod] = ((sig - mu[None, :, :]) / sigma[None, :, :]).astype(np.float32)
    return out


def process_subject(pkl_path: Path, out_dir: Path) -> dict:
    raw = load_subject(pkl_path)
    wrist = raw["signal"]["wrist"]
    labels = np.asarray(raw["label"]).flatten().astype(np.int8)

    resampled = {}
    for mod, src_hz in WRIST_RATES.items():
        sig = wrist[mod]
        if sig.ndim == 1:
            sig = sig[:, None]
        resampled[mod] = resample_to_target(sig, src_hz, TARGET_RATE)

    labels_4hz = downsample_labels(labels, LABEL_RATE, TARGET_RATE)

    n = min(min(m.shape[0] for m in resampled.values()), len(labels_4hz))
    resampled = {m: s[:n] for m, s in resampled.items()}
    labels_4hz = labels_4hz[:n]

    windows, bin_labels = make_windows(resampled, labels_4hz)
    if not windows:
        return {"subject": pkl_path.parent.name, "n_windows": 0}

    windows = zscore_per_subject(windows)

    subj = pkl_path.parent.name
    out_path = out_dir / f"{subj}.npz"
    np.savez_compressed(
        out_path,
        labels=bin_labels,
        **{f"X_{m}": v for m, v in windows.items()},
    )
    n_stress = int((bin_labels == 1).sum())
    n_non = int((bin_labels == 0).sum())
    return {
        "subject": subj,
        "n_windows": len(bin_labels),
        "n_stress": n_stress,
        "n_non_stress": n_non,
        "stress_frac": n_stress / max(1, len(bin_labels)),
        "shapes": {m: list(v.shape) for m, v in windows.items()},
        "out": str(out_path),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="data/raw/WESAD")
    p.add_argument("--out_dir", default="data/processed")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for subj in SUBJECTS:
        pkl_path = raw_dir / subj / f"{subj}.pkl"
        if not pkl_path.exists():
            print(f"MISSING: {pkl_path}", file=sys.stderr)
            continue
        info = process_subject(pkl_path, out_dir)
        print(f"{subj}: n_windows={info.get('n_windows')} stress_frac={info.get('stress_frac', 0):.3f}")
        summary.append(info)

    print(f"\nProcessed {len(summary)} subjects.")
    print(f"Total windows: {sum(s.get('n_windows', 0) for s in summary)}")
    return summary


if __name__ == "__main__":
    main()
