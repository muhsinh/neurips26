"""Sanity checks 3.1-3.5 from the brief.

Usage: python -m src.sanity --raw_dir data/raw/WESAD --processed_dir data/processed
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path
import numpy as np

EXPECTED = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
            "S10", "S11", "S13", "S14", "S15", "S16", "S17"]


def check_download(raw_dir: Path) -> dict:
    missing = []
    present = []
    for s in EXPECTED:
        p = raw_dir / s / f"{s}.pkl"
        (present if p.exists() else missing).append(s)
    return {"present": present, "missing": missing}


def check_pickle_load(raw_dir: Path, subj: str = "S2") -> dict:
    p = raw_dir / subj / f"{subj}.pkl"
    if not p.exists():
        return {"ok": False, "reason": f"file not found: {p}"}
    try:
        d = pickle.load(open(p, "rb"), encoding="latin1")
    except UnicodeDecodeError as e:
        return {"ok": False, "reason": f"unicode error (encoding!=latin1): {e}"}
    except Exception as e:
        return {"ok": False, "reason": f"load error: {e}"}
    keys = list(d.get("signal", {}).get("wrist", {}).keys())
    expected_keys = {"ACC", "BVP", "EDA", "TEMP"}
    missing = expected_keys - set(keys)
    chest_len = d["signal"]["chest"]["ECG"].shape[0]
    bvp_len = d["signal"]["wrist"]["BVP"].shape[0]
    eda_len = d["signal"]["wrist"]["EDA"].shape[0]
    return {
        "ok": not missing,
        "wrist_keys": keys,
        "missing_modalities": list(missing),
        "labels_len": int(np.asarray(d["label"]).size),
        "chest_700hz_seconds": chest_len / 700,
        "wrist_BVP_64hz_seconds": bvp_len / 64,
        "wrist_EDA_4hz_seconds": eda_len / 4,
    }


def check_label_distribution(processed_dir: Path) -> dict:
    out = {}
    for npz in sorted(processed_dir.glob("S*.npz")):
        d = np.load(npz)
        labels = d["labels"]
        n_total = len(labels)
        n_stress = int((labels == 1).sum())
        out[npz.stem] = {
            "n_windows": n_total,
            "n_stress": n_stress,
            "n_non_stress": int((labels == 0).sum()),
            "stress_frac": n_stress / max(1, n_total),
        }
    return out


def check_tensor_shapes(processed_dir: Path) -> dict:
    out = {}
    for npz in sorted(processed_dir.glob("S*.npz")):
        d = np.load(npz)
        shapes = {k: list(d[k].shape) for k in d.files}
        out[npz.stem] = shapes
    return out


def smoke_test_loso_lr(processed_dir: Path) -> dict:
    """Sanity 3.5. Logistic regression on flattened features.

    1) Intra-subject 80/20 baseline.
    2) Per-subject LOSO. Show the per-subject collapse signal in raw form.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.preprocessing import StandardScaler

    files = sorted(processed_dir.glob("S*.npz"))
    if not files:
        return {"ok": False, "reason": "no processed npz files"}

    def flatten(npz):
        d = np.load(npz)
        feats = []
        for m in ["ACC", "BVP", "EDA", "TEMP"]:
            x = d[f"X_{m}"]
            feats.append(x.reshape(x.shape[0], -1))
        X = np.concatenate(feats, axis=1)
        y = d["labels"]
        return X, y

    # Intra-subject 80/20 on first available subject
    X0, y0 = flatten(files[0])
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y0))
    cut = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    sc = StandardScaler().fit(X0[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X0[tr]), y0[tr])
    intra_acc = accuracy_score(y0[te], clf.predict(sc.transform(X0[te])))

    # LOSO across all subjects
    Xs, ys, sids = [], [], []
    for f in files:
        X, y = flatten(f)
        Xs.append(X); ys.append(y); sids.append(f.stem)

    loso = {}
    all_train_X = np.concatenate(Xs); all_train_y = np.concatenate(ys)
    sc_all = StandardScaler().fit(all_train_X)
    Xn = [sc_all.transform(x) for x in Xs]

    for i, sid in enumerate(sids):
        train_X = np.concatenate([x for j, x in enumerate(Xn) if j != i])
        train_y = np.concatenate([y for j, y in enumerate(ys) if j != i])
        test_X, test_y = Xn[i], ys[i]
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(train_X, train_y)
        pred = clf.predict(test_X)
        loso[sid] = {
            "f1": float(f1_score(test_y, pred, pos_label=1, zero_division=0)),
            "acc": float(accuracy_score(test_y, pred)),
            "n": int(len(test_y)),
            "stress_frac": float((test_y == 1).mean()) if len(test_y) else 0.0,
        }

    f1s = np.array([v["f1"] for v in loso.values()])
    return {
        "ok": True,
        "intra_subject_acc_S0": float(intra_acc),
        "loso_per_subject": loso,
        "loso_f1_mean": float(f1s.mean()),
        "loso_f1_std": float(f1s.std()),
        "loso_f1_min": float(f1s.min()),
        "loso_f1_max": float(f1s.max()),
        "n_subjects_below_0.3_f1": int((f1s < 0.3).sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="data/raw/WESAD")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--skip", default="", help="comma-separated check names to skip")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    proc_dir = Path(args.processed_dir)
    skip = set(args.skip.split(",")) if args.skip else set()

    print("=== SANITY 3.1: download ===")
    info = check_download(raw_dir)
    print(f"  present: {info['present']}")
    if info["missing"]:
        print(f"  MISSING: {info['missing']}")
        sys.exit(1)
    print(f"  all 15 subjects present")

    print("\n=== SANITY 3.2: pickle load ===")
    info = check_pickle_load(raw_dir, "S2")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if not info["ok"]:
        sys.exit(1)

    if "label" in skip:
        print("\n[skipping label/shape/smoke checks]")
        return

    print("\n=== SANITY 3.3: label distribution per subject ===")
    label_info = check_label_distribution(proc_dir)
    for sid, d in label_info.items():
        print(f"  {sid}: n={d['n_windows']:4d} stress_frac={d['stress_frac']:.3f}")
    if any(d["n_windows"] == 0 for d in label_info.values()):
        print("WARN: some subjects have 0 windows")

    print("\n=== SANITY 3.4: tensor shapes ===")
    shapes = check_tensor_shapes(proc_dir)
    sid0 = next(iter(shapes))
    for k, v in shapes[sid0].items():
        print(f"  {sid0}.{k}: {v}")

    print("\n=== SANITY 3.5: logistic regression LOSO smoke ===")
    smoke = smoke_test_loso_lr(proc_dir)
    print(f"  intra-subject acc on first subject: {smoke['intra_subject_acc_S0']:.3f}")
    print(f"  LOSO F1 mean={smoke['loso_f1_mean']:.3f} std={smoke['loso_f1_std']:.3f}")
    print(f"  LOSO F1 range=[{smoke['loso_f1_min']:.3f}, {smoke['loso_f1_max']:.3f}]")
    print(f"  subjects with F1<0.3 (collapse signal): {smoke['n_subjects_below_0.3_f1']} / {len(smoke['loso_per_subject'])}")
    print("  per-subject F1:")
    for sid, v in smoke["loso_per_subject"].items():
        flag = "  <-- COLLAPSE" if v["f1"] < 0.3 else ""
        print(f"    {sid}: f1={v['f1']:.3f} acc={v['acc']:.3f} n={v['n']} stress={v['stress_frac']:.2f}{flag}")

    print("\nALL SANITY CHECKS PASSED" if smoke["ok"] else "FAILED")


if __name__ == "__main__":
    main()
