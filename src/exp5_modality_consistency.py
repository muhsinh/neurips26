"""Exp5: Cross-architecture modality-importance consistency.

For each architecture, build a length-4 vector of mean F1-loss when each
single modality is the only one dropped (single_mod_drop_F1 - clean_F1).

Compute Spearman ρ + Pearson r between every architecture pair.
Bootstrap 95% CI: resample 15 subjects with replacement 1000 times,
recompute per-architecture F1-loss vectors per bootstrap, recompute ρ.
Permutation null: shuffle modality labels in one architecture's vector
10000 times, recompute ρ, count proportion ≥ observed.

Also runs dual-modality dropout (C(4,2)=6) as fallback statistical-power
boost, since 4-modality Spearman has limited support.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

MODALITIES = ["ACC", "BVP", "EDA", "TEMP"]
MODELS = ["late_fusion_mlp", "cross_attention", "scale_proxy"]
SINGLE_DROPS = [f"drop_{m}" for m in MODALITIES]
PAIR_DROPS = [f"drop_{a}_{b}" for a, b in itertools.combinations(MODALITIES, 2)]


def load_dropout(exp2_dir: Path) -> dict:
    out = {}
    for m in MODELS:
        out[m] = json.load(open(exp2_dir / f"{m}.json"))
    return out


def f1_loss_vec(arch_data: dict, conds: list[str], subjects: list[str]) -> np.ndarray:
    """Return per-condition mean F1-loss across (subject × seed). Length len(conds)."""
    clean_mat = []
    for s in subjects:
        seed_map = arch_data["all_clean"].get(s, {})
        for sd, r in seed_map.items():
            clean_mat.append(r["f1_stress"])
    clean_overall = float(np.mean(clean_mat))

    losses = []
    for c in conds:
        vals = []
        for s in subjects:
            for sd, r in arch_data[c].get(s, {}).items():
                vals.append(clean_overall - r["f1_stress"])
        losses.append(float(np.mean(vals)) if vals else 0.0)
    return np.array(losses)


def f1_loss_vec_subject_subset(arch_data: dict, conds: list[str], subj_subset: list[str]) -> np.ndarray:
    clean_vals = []
    for s in subj_subset:
        for sd, r in arch_data["all_clean"].get(s, {}).items():
            clean_vals.append(r["f1_stress"])
    clean_mu = float(np.mean(clean_vals)) if clean_vals else 0.0
    losses = []
    for c in conds:
        vals = []
        for s in subj_subset:
            for sd, r in arch_data[c].get(s, {}).items():
                vals.append(clean_mu - r["f1_stress"])
        losses.append(float(np.mean(vals)) if vals else 0.0)
    return np.array(losses)


def bootstrap_rho(arch_a: dict, arch_b: dict, conds: list[str], subjects: list[str],
                  n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    rhos = []
    n = len(subjects)
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        sample = [subjects[i] for i in sample_idx]
        va = f1_loss_vec_subject_subset(arch_a, conds, sample)
        vb = f1_loss_vec_subject_subset(arch_b, conds, sample)
        if va.std() < 1e-9 or vb.std() < 1e-9:
            continue
        rho = spearmanr(va, vb).correlation
        if not np.isnan(rho):
            rhos.append(rho)
    if not rhos:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(rhos)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def permutation_null(va: np.ndarray, vb: np.ndarray, n_perm: int = 10000,
                     seed: int = 42) -> float:
    """One-sided permutation p: fraction of permuted ρ ≥ observed."""
    rng = np.random.default_rng(seed)
    if va.std() < 1e-9 or vb.std() < 1e-9:
        return float("nan")
    obs = spearmanr(va, vb).correlation
    n_ge = 0
    for _ in range(n_perm):
        vb_perm = rng.permutation(vb)
        rho = spearmanr(va, vb_perm).correlation
        if not np.isnan(rho) and rho >= obs:
            n_ge += 1
    return (n_ge + 1) / (n_perm + 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp2_dir", default="results/exp2_dropout")
    p.add_argument("--out_dir", default="results/exp5_modality_consistency")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--n_perm", type=int, default=10000)
    args = p.parse_args()

    SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
                "S10", "S11", "S13", "S14", "S15", "S16", "S17"]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data = load_dropout(Path(args.exp2_dir))

    for cond_kind, conds in [("single_modality", SINGLE_DROPS),
                              ("dual_modality", PAIR_DROPS)]:
        print(f"\n=== {cond_kind} ({len(conds)} conditions) ===")
        vecs = {m: f1_loss_vec(data[m], conds, SUBJECTS) for m in MODELS}
        for m in MODELS:
            named = dict(zip(conds, np.round(vecs[m], 3).tolist()))
            print(f"  {m}: {named}")

        result = {"vectors": {m: vecs[m].tolist() for m in MODELS},
                  "conditions": conds, "pairs": {}}

        for a, b in itertools.combinations(MODELS, 2):
            va, vb = vecs[a], vecs[b]
            rho_obs = float(spearmanr(va, vb).correlation) if va.std() > 1e-9 and vb.std() > 1e-9 else float("nan")
            r_obs = float(pearsonr(va, vb).correlation) if va.std() > 1e-9 and vb.std() > 1e-9 else float("nan")
            mean_rho, lo, hi = bootstrap_rho(data[a], data[b], conds, SUBJECTS,
                                             n_boot=args.n_boot)
            p_perm = permutation_null(va, vb, n_perm=args.n_perm)
            print(f"  {a} × {b}: ρ={rho_obs:+.3f} (boot mean {mean_rho:+.3f}, "
                  f"95% CI [{lo:+.3f}, {hi:+.3f}]), r={r_obs:+.3f}, perm p={p_perm:.4f}")
            result["pairs"][f"{a}__x__{b}"] = {
                "spearman_rho_obs": rho_obs,
                "pearson_r_obs": r_obs,
                "bootstrap_rho_mean": mean_rho,
                "bootstrap_rho_ci_lo": lo,
                "bootstrap_rho_ci_hi": hi,
                "permutation_p": p_perm,
                "n_boot": args.n_boot,
                "n_perm": args.n_perm,
            }

        with open(out_dir / f"{cond_kind}.json", "w") as f:
            json.dump(result, f, indent=2)

    print(f"\nsaved {out_dir}/single_modality.json, dual_modality.json")


if __name__ == "__main__":
    main()
