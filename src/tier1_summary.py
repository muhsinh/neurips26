"""Tier 1 final summary: aggregate Exp5/6/7 results into reviewer-grade output."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def main():
    print("=" * 72)
    print("TIER 1 SUMMARY")
    print("=" * 72)

    # ---- Exp 5 ----
    print("\n--- Exp 5: cross-architecture modality-importance correlation ---")
    for kind in ("single_modality", "dual_modality"):
        p = Path(f"results/exp5_modality_consistency/{kind}.json")
        if not p.exists():
            print(f"  ({kind}: no data)")
            continue
        d = json.load(open(p))
        print(f"  [{kind}, n_conds={len(d['conditions'])}]")
        for pair_key, info in d["pairs"].items():
            print(f"    {pair_key}: ρ={info['spearman_rho_obs']:+.3f} "
                  f"(boot mean {info['bootstrap_rho_mean']:+.3f}, "
                  f"95% CI [{info['bootstrap_rho_ci_lo']:+.3f}, "
                  f"{info['bootstrap_rho_ci_hi']:+.3f}]), "
                  f"r={info['pearson_r_obs']:+.3f}, perm p={info['permutation_p']:.4f}")

    # ---- Exp 6 ----
    print("\n--- Exp 6: per-(subject, arch, seed) bootstrap CIs ---")
    p = Path("results/exp6_bootstrap_ci/per_subject_ci.json")
    if p.exists():
        d = json.load(open(p))
        for subj in ("S17", "S14"):
            print(f"  {subj}:")
            for m in ("late_fusion_mlp", "cross_attention", "scale_proxy"):
                rec = d[m][subj]
                summary = rec["summary"]
                print(f"    {m:18s}: across-seed mean F1={summary['across_seed_mean_point_f1']:.3f} "
                      f"(std {summary['across_seed_std_point_f1']:.3f})")
                for sd in ("42", "1337", "2024"):
                    if sd in rec["per_seed"]:
                        ps = rec["per_seed"][sd]
                        print(f"      seed {sd}: F1={ps['point_f1']:.3f} "
                              f"95% CI [{ps['boot_ci_lo']:.3f}, {ps['boot_ci_hi']:.3f}]")

    # ---- Exp 7 ----
    print("\n--- Exp 7: Gulrajani-style ERM tuning sweep ---")
    base = json.load(open("results/exp7_erm_tuning/baseline_seed42.json"))
    print(f"  Baseline (X-attn, seed 42): "
          f"LOSO F1={base['loso_f1_mean']:.3f}, "
          f"worst-drop F1={base['exp2_drop_BVP_EDA_f1']:.3f}, "
          f"σ=2 F1={base['exp3_sigma_2.0_f1']:.3f}")

    for rule in ("train_loss", "val_f1"):
        s_path = Path(f"results/exp7_erm_tuning/loso_summary_{rule}.json")
        d_path = Path(f"results/exp7_erm_tuning/dropout_{rule}.json")
        n_path = Path(f"results/exp7_erm_tuning/degradation_{rule}.json")
        if not s_path.exists():
            print(f"  ({rule}: no data)")
            continue
        loso = json.load(open(s_path))
        clean_f1s = [l["test_f1"] for l in loso]
        loso_mean = float(np.mean(clean_f1s))
        coll = sum(1 for f in clean_f1s if f < 0.3)
        # dropout
        dr = json.load(open(d_path))
        cond_means = {}
        for cond, subj_seeds in dr.items():
            f1s = [r["f1_stress"]
                   for s, sd_map in subj_seeds.items()
                   for r in sd_map.values()]
            cond_means[cond] = float(np.mean(f1s))
        clean_drop = cond_means["all_clean"]
        worst_cond = min((c for c in cond_means if c != "all_clean"),
                         key=lambda c: cond_means[c])
        worst_drop = cond_means[worst_cond]
        # noise
        ng = json.load(open(n_path))
        noise_means = {}
        for sigma, subj_seeds in ng["gaussian"].items():
            f1s = [r["f1_stress"]
                   for s, sd_map in subj_seeds.items()
                   for r in sd_map.values()]
            noise_means[sigma] = float(np.mean(f1s))
        sigma2 = noise_means["2.0"]

        loso_delta_pp = (loso_mean - base["loso_f1_mean"]) * 100
        worst_delta_pp = (worst_drop - base["exp2_drop_BVP_EDA_f1"]) * 100
        sigma2_delta_pp = (sigma2 - base["exp3_sigma_2.0_f1"]) * 100

        print(f"  [{rule}]:")
        print(f"    LOSO F1 mean: {loso_mean:.3f} (Δ={loso_delta_pp:+.1f}pp), collapsed={coll}/15")
        print(f"    worst-drop ({worst_cond}) F1: {worst_drop:.3f} (Δ={worst_delta_pp:+.1f}pp)")
        print(f"    σ=2 F1: {sigma2:.3f} (Δ={sigma2_delta_pp:+.1f}pp)")

        # Surfacing flags
        flags = []
        if abs(loso_delta_pp) > 3: flags.append("LOSO F1 > 3pp shift")
        if worst_delta_pp > 3: flags.append("worst-drop recovers > 3pp")
        if sigma2_delta_pp > 3: flags.append(f"σ=2 recovers > 3pp")
        if flags:
            print(f"    ** SURFACING FLAGS: {flags}")
        else:
            print(f"    (no surfacing flags — failure modes persist under tuning)")

        # selection-bias check (val F1 of selected config vs test F1 of selected config)
        all_folds = json.load(open("results/exp7_erm_tuning/all_folds.json"))["folds"]
        sel_key = "sel_train_loss_tag" if rule == "train_loss" else "sel_val_f1_tag"
        gaps = []
        for fold in all_folds:
            sel_cfg = next(c for c in fold["configs"] if c["tag"] == fold[sel_key])
            gaps.append(abs(sel_cfg["val_f1"] - sel_cfg["test_f1"]))
        mean_gap_pp = float(np.mean(gaps)) * 100
        print(f"    val-vs-test gap (mean across folds): {mean_gap_pp:.1f}pp")
        if mean_gap_pp > 10:
            print(f"    ** SELECTION-BIAS FLAG: gap > 10pp")

    # Decision logs
    print("\n--- Decision logs ---")
    for d in sorted(Path("decisions").glob("d1[3-5]_*.md")):
        print(f"  {d}")


if __name__ == "__main__":
    main()
