"""Exp6: Per-(subject, architecture, seed) bootstrap CIs on F1.

For each (subject, arch, seed) triple, take the per-window predictions stored
in Exp1 result JSONs (y_true, y_pred). Bootstrap-resample windows 1000 times,
compute F1 on each resample, report bootstrap mean + 95% CI.

Two reported quantities per (subject, architecture):
  1. Within-subject sampling uncertainty: the bootstrap CI averaged across
     seeds. Captures "if we resampled this subject's windows, how much would
     F1 wobble?"
  2. Across-seed training uncertainty: std across the 3 per-seed point F1
     estimates (no bootstrap). Captures "if we retrained with a different
     seed, how much would F1 wobble?"
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
            "S10", "S11", "S13", "S14", "S15", "S16", "S17"]
SEEDS = [42, 1337, 2024]
MODELS = ["late_fusion_mlp", "cross_attention", "scale_proxy"]


def bootstrap_f1(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int,
                 seed: int) -> tuple[float, float, float, list[float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1s.append(float(f1_score(y_true[idx], y_pred[idx], pos_label=1, zero_division=0)))
    arr = np.array(f1s)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), f1s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp1_dir", default="results/exp1_loso")
    p.add_argument("--out_dir", default="results/exp6_bootstrap_ci")
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    exp1 = Path(args.exp1_dir)

    out: dict = {}
    for m in MODELS:
        out[m] = {}
        for s in SUBJECTS:
            out[m][s] = {"per_seed": {}, "summary": {}}
            point_f1s = []
            ci_widths = []
            ci_los = []
            ci_his = []
            for sd in SEEDS:
                jp = exp1 / f"{m}__{s}__seed{sd}.json"
                if not jp.exists():
                    continue
                d = json.load(open(jp))
                y = np.array(d["test_metrics"]["y_true"])
                p_arr = np.array(d["test_metrics"]["y_pred"])
                point_f1 = d["test_metrics"]["f1_stress"]
                mean, lo, hi, _ = bootstrap_f1(y, p_arr, args.n_boot, seed=42 + sd)
                out[m][s]["per_seed"][str(sd)] = {
                    "point_f1": point_f1,
                    "boot_mean": mean,
                    "boot_ci_lo": lo,
                    "boot_ci_hi": hi,
                    "n_windows": len(y),
                    "n_stress": int((y == 1).sum()),
                }
                point_f1s.append(point_f1)
                ci_widths.append(hi - lo)
                ci_los.append(lo); ci_his.append(hi)
            if point_f1s:
                out[m][s]["summary"] = {
                    "across_seed_mean_point_f1": float(np.mean(point_f1s)),
                    "across_seed_std_point_f1": float(np.std(point_f1s)),
                    "mean_within_subject_ci_width": float(np.mean(ci_widths)),
                    "mean_within_subject_ci_lo": float(np.mean(ci_los)),
                    "mean_within_subject_ci_hi": float(np.mean(ci_his)),
                    "all_seeds_ci_excludes_0.3": all(
                        out[m][s]["per_seed"][str(sd)]["boot_ci_lo"] > 0.3
                        or out[m][s]["per_seed"][str(sd)]["boot_ci_hi"] < 0.3
                        for sd in SEEDS if str(sd) in out[m][s]["per_seed"]
                    ),
                    "all_seeds_collapse_confirmed": all(
                        out[m][s]["per_seed"][str(sd)]["boot_ci_hi"] < 0.3
                        for sd in SEEDS if str(sd) in out[m][s]["per_seed"]
                    ),
                }

    with open(out_dir / "per_subject_ci.json", "w") as f:
        json.dump(out, f, indent=2)

    print("=== S17 (universal collapse subject) ===")
    for m in MODELS:
        d = out[m]["S17"]
        print(f"  {m}: across-seed F1 mean={d['summary']['across_seed_mean_point_f1']:.3f} "
              f"std={d['summary']['across_seed_std_point_f1']:.3f}")
        for sd in SEEDS:
            ps = d["per_seed"].get(str(sd), {})
            if ps:
                print(f"    seed {sd}: F1={ps['point_f1']:.3f}, "
                      f"95% CI [{ps['boot_ci_lo']:.3f}, {ps['boot_ci_hi']:.3f}], "
                      f"n={ps['n_windows']}")
        print(f"    collapse confirmed across all seeds (CI hi < 0.3): "
              f"{d['summary']['all_seeds_collapse_confirmed']}")
    print()
    print("=== S14 (partial collapse subject) ===")
    for m in MODELS:
        d = out[m]["S14"]
        print(f"  {m}: across-seed F1 mean={d['summary']['across_seed_mean_point_f1']:.3f} "
              f"std={d['summary']['across_seed_std_point_f1']:.3f}")
        for sd in SEEDS:
            ps = d["per_seed"].get(str(sd), {})
            if ps:
                print(f"    seed {sd}: F1={ps['point_f1']:.3f}, "
                      f"95% CI [{ps['boot_ci_lo']:.3f}, {ps['boot_ci_hi']:.3f}]")
        print(f"    collapse confirmed across all seeds (CI hi < 0.3): "
              f"{d['summary']['all_seeds_collapse_confirmed']}")

    # Distribution of CI widths across all subjects (to decide if Fig2A inset works)
    all_widths = []
    for m in MODELS:
        for s in SUBJECTS:
            for sd in SEEDS:
                ps = out[m][s]["per_seed"].get(str(sd), {})
                if ps:
                    all_widths.append(ps["boot_ci_hi"] - ps["boot_ci_lo"])
    print(f"\nCI width distribution: median={np.median(all_widths):.3f}, "
          f"p25={np.percentile(all_widths, 25):.3f}, p75={np.percentile(all_widths, 75):.3f}, "
          f"max={max(all_widths):.3f}")
    print(f"saved {out_dir}/per_subject_ci.json")


if __name__ == "__main__":
    main()
