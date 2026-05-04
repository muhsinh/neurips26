"""Aggregate Exp1/Exp2/Exp3 result JSONs into per-subject and overall stats.

Outputs:
  results/agg/exp1_per_subject.json  -- {model: {subj: {f1_mean, f1_std, seeds: [...]}}}
  results/agg/exp1_overall.json      -- {model: {f1_mean, f1_std, n_subjects}}
  results/agg/exp2_dropout.json      -- {model: {cond: {f1_mean, f1_std}}}
  results/agg/exp3_degradation.json  -- {model: {gaussian: {sigma: ...}, realistic: {...}}}
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def agg_exp1(exp1_dir: Path, models: list[str], subjects: list[str], seeds: list[int]) -> tuple[dict, dict]:
    per_subj = {m: {s: {} for s in subjects} for m in models}
    overall = {}
    for m in models:
        for subj in subjects:
            f1s = []; accs = []; baseline_f1s = []; n_windows = []
            for sd in seeds:
                jp = exp1_dir / f"{m}__{subj}__seed{sd}.json"
                if not jp.exists():
                    continue
                d = json.load(open(jp))
                f1s.append(d["test_metrics"]["f1_stress"])
                accs.append(d["test_metrics"]["acc"])
                baseline_f1s.append(d["majority_baseline"]["f1"])
                n_windows.append(d["test_metrics"]["n"])
            per_subj[m][subj] = {
                "f1_seeds": f1s,
                "acc_seeds": accs,
                "f1_mean": float(np.mean(f1s)) if f1s else None,
                "f1_std": float(np.std(f1s)) if f1s else None,
                "acc_mean": float(np.mean(accs)) if accs else None,
                "majority_f1_mean": float(np.mean(baseline_f1s)) if baseline_f1s else None,
                "n_windows": int(np.mean(n_windows)) if n_windows else 0,
            }
        subj_means = [per_subj[m][s]["f1_mean"] for s in subjects
                      if per_subj[m][s]["f1_mean"] is not None]
        overall[m] = {
            "f1_mean_over_subjects": float(np.mean(subj_means)) if subj_means else None,
            "f1_std_over_subjects": float(np.std(subj_means)) if subj_means else None,
            "f1_min": float(np.min(subj_means)) if subj_means else None,
            "f1_max": float(np.max(subj_means)) if subj_means else None,
            "n_collapsed_below_0.3": int(sum(1 for x in subj_means if x < 0.3)),
            "n_subjects": len(subj_means),
        }
    return per_subj, overall


def agg_exp2(exp2_dir: Path, models: list[str]) -> dict:
    out = {}
    for m in models:
        jp = exp2_dir / f"{m}.json"
        if not jp.exists():
            out[m] = None; continue
        d = json.load(open(jp))
        out[m] = {}
        for cond, subj_seeds in d.items():
            f1s = []
            for subj, seedmap in subj_seeds.items():
                for sd, r in seedmap.items():
                    f1s.append(r["f1_stress"])
            out[m][cond] = {
                "f1_mean": float(np.mean(f1s)) if f1s else None,
                "f1_std": float(np.std(f1s)) if f1s else None,
                "n_evals": len(f1s),
            }
    return out


def agg_exp3(exp3_dir: Path, models: list[str]) -> dict:
    out = {}
    for m in models:
        jp = exp3_dir / f"{m}.json"
        if not jp.exists():
            out[m] = None; continue
        d = json.load(open(jp))
        out[m] = {"gaussian": {}, "realistic": {}}
        for sigma, subj_seeds in d["gaussian"].items():
            f1s = [r["f1_stress"]
                   for subj, seedmap in subj_seeds.items()
                   for sd, r in seedmap.items()]
            out[m]["gaussian"][str(sigma)] = {
                "f1_mean": float(np.mean(f1s)) if f1s else None,
                "f1_std": float(np.std(f1s)) if f1s else None,
                "n_evals": len(f1s),
            }
        for cname, subj_seeds in d["realistic"].items():
            f1s = [r["f1_stress"]
                   for subj, seedmap in subj_seeds.items()
                   for sd, r in seedmap.items()]
            out[m]["realistic"][cname] = {
                "f1_mean": float(np.mean(f1s)) if f1s else None,
                "f1_std": float(np.std(f1s)) if f1s else None,
                "n_evals": len(f1s),
            }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp1_dir", default="results/exp1_loso")
    p.add_argument("--exp2_dir", default="results/exp2_dropout")
    p.add_argument("--exp3_dir", default="results/exp3_degradation")
    p.add_argument("--out_dir", default="results/agg")
    p.add_argument("--models", default="late_fusion_mlp,cross_attention,scale_proxy")
    args = p.parse_args()

    SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
                "S10", "S11", "S13", "S14", "S15", "S16", "S17"]
    SEEDS = [42, 1337, 2024]
    models = args.models.split(",")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    per_subj, overall = agg_exp1(Path(args.exp1_dir), models, SUBJECTS, SEEDS)
    json.dump(per_subj, open(out_dir / "exp1_per_subject.json", "w"), indent=2)
    json.dump(overall, open(out_dir / "exp1_overall.json", "w"), indent=2)

    e2 = agg_exp2(Path(args.exp2_dir), models)
    json.dump(e2, open(out_dir / "exp2_dropout.json", "w"), indent=2)

    e3 = agg_exp3(Path(args.exp3_dir), models)
    json.dump(e3, open(out_dir / "exp3_degradation.json", "w"), indent=2)

    print("=== Exp1 overall ===")
    for m, d in overall.items():
        if not d["n_subjects"]:
            print(f"  {m}: NO RESULTS"); continue
        print(f"  {m}: F1={d['f1_mean_over_subjects']:.3f} ± {d['f1_std_over_subjects']:.3f}, "
              f"min={d['f1_min']:.3f} max={d['f1_max']:.3f}, "
              f"collapsed={d['n_collapsed_below_0.3']}/{d['n_subjects']}")

    print("\n=== Exp2 dropout (selected) ===")
    for m, d in e2.items():
        if d is None: continue
        clean = d.get("all_clean", {}).get("f1_mean")
        worst_drop = min(((c, x["f1_mean"]) for c, x in d.items() if c != "all_clean"),
                        key=lambda kv: kv[1] if kv[1] is not None else 1e9, default=("?", None))
        print(f"  {m}: clean F1={clean:.3f}  worst_drop {worst_drop[0]} -> F1={worst_drop[1]:.3f}")

    print("\n=== Exp3 degradation (Gaussian) ===")
    for m, d in e3.items():
        if d is None: continue
        for sigma_key, x in d["gaussian"].items():
            print(f"  {m}: sigma={sigma_key} F1={x['f1_mean']:.3f}")


if __name__ == "__main__":
    main()
