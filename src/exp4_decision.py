"""Apply D06 inclusion threshold for cross_modal_recon variant.

Threshold: ≥3 pp absolute F1 improvement (mean over seeds) over cross_attention,
on at least one of {Exp2 worst-case dropout, Exp3 high-noise condition},
AND improvement exceeds 1 std of the seed-fold variance.

Outputs: results/exp4_recon/inclusion_decision.json with verdict + numbers.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def load_json(p: Path):
    return json.load(open(p))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp4_e2", default="results/exp2_dropout/cross_modal_recon.json")
    p.add_argument("--exp4_e3", default="results/exp3_degradation/cross_modal_recon.json")
    p.add_argument("--ref_e2", default="results/exp2_dropout/cross_attention.json")
    p.add_argument("--ref_e3", default="results/exp3_degradation/cross_attention.json")
    p.add_argument("--out", default="results/exp4_recon/inclusion_decision.json")
    args = p.parse_args()

    out = {"d06_threshold_pp": 3.0, "checks": [], "include": False}

    # Worst-case dropout comparison
    if Path(args.exp4_e2).exists() and Path(args.ref_e2).exists():
        e2_recon = load_json(args.exp4_e2)
        e2_ref = load_json(args.ref_e2)

        # Compute per-condition mean F1 over (subject × seed)
        def mean_f1_by_cond(d):
            out = {}
            for cond, subj_seeds in d.items():
                f1s = [r["f1_stress"]
                       for subj, seedmap in subj_seeds.items()
                       for sd, r in seedmap.items()]
                out[cond] = (float(np.mean(f1s)) if f1s else None,
                             float(np.std(f1s)) if f1s else None)
            return out

        recon_means = mean_f1_by_cond(e2_recon)
        ref_means = mean_f1_by_cond(e2_ref)

        # Worst-case for ref = condition where ref has lowest F1
        worst_cond = min((c for c in ref_means if c != "all_clean" and ref_means[c][0] is not None),
                        key=lambda c: ref_means[c][0])
        recon_v, recon_std = recon_means[worst_cond]
        ref_v, ref_std = ref_means[worst_cond]
        delta_pp = (recon_v - ref_v) * 100
        threshold_std = max(ref_std, recon_std) * 100
        passes = delta_pp >= 3.0 and delta_pp > threshold_std
        out["checks"].append({
            "test": "worst-case dropout",
            "condition": worst_cond,
            "ref_f1": ref_v, "recon_f1": recon_v,
            "delta_pp": delta_pp, "std_pp": threshold_std,
            "passes": bool(passes),
        })
        if passes:
            out["include"] = True

    # High-noise comparison
    if Path(args.exp4_e3).exists() and Path(args.ref_e3).exists():
        e3_recon = load_json(args.exp4_e3)
        e3_ref = load_json(args.ref_e3)

        high_sigma = "2.0"
        recon_block = e3_recon["gaussian"].get(high_sigma, {})
        ref_block = e3_ref["gaussian"].get(high_sigma, {})
        recon_f1s = [r["f1_stress"]
                     for s, sd in recon_block.items() for sd_k, r in sd.items()]
        ref_f1s = [r["f1_stress"]
                   for s, sd in ref_block.items() for sd_k, r in sd.items()]
        if recon_f1s and ref_f1s:
            recon_v = float(np.mean(recon_f1s))
            ref_v = float(np.mean(ref_f1s))
            recon_std = float(np.std(recon_f1s))
            ref_std = float(np.std(ref_f1s))
            delta_pp = (recon_v - ref_v) * 100
            threshold_std = max(ref_std, recon_std) * 100
            passes = delta_pp >= 3.0 and delta_pp > threshold_std
            out["checks"].append({
                "test": f"high-noise σ={high_sigma}",
                "ref_f1": ref_v, "recon_f1": recon_v,
                "delta_pp": delta_pp, "std_pp": threshold_std,
                "passes": bool(passes),
            })
            if passes:
                out["include"] = True

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("=== Exp4 inclusion decision (D06) ===")
    print(f"  threshold: ≥{out['d06_threshold_pp']} pp AND beyond 1 std")
    for c in out["checks"]:
        verdict = "PASS" if c["passes"] else "FAIL"
        print(f"  {c['test']}: ref F1={c['ref_f1']:.3f} recon F1={c['recon_f1']:.3f} "
              f"Δ={c['delta_pp']:+.2f}pp std={c['std_pp']:.2f}pp -> {verdict}")
    print(f"\n  include in paper: {out['include']}")


if __name__ == "__main__":
    main()
