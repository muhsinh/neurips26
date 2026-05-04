"""Final summary: print headline numbers for all 4 experiments + decision file paths."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agg_dir", default="results/agg")
    args = p.parse_args()
    agg = Path(args.agg_dir)

    print("=" * 70)
    print("WESAD diagnostic study — final summary")
    print("=" * 70)

    if (agg / "exp1_overall.json").exists():
        d = json.load(open(agg / "exp1_overall.json"))
        print("\n--- Experiment 1: per-subject collapse (LOSO) ---")
        for m, v in d.items():
            print(f"  {m:18s}: F1={v['f1_mean_over_subjects']:.3f} ± {v['f1_std_over_subjects']:.3f}  "
                  f"min={v['f1_min']:.3f} max={v['f1_max']:.3f}  "
                  f"collapsed={v['n_collapsed_below_0.3']}/{v['n_subjects']}")

    if (agg / "exp2_dropout.json").exists():
        d = json.load(open(agg / "exp2_dropout.json"))
        print("\n--- Experiment 2: modality shortcut (test-time zero-mask) ---")
        for m, conds in d.items():
            if not conds:
                continue
            clean = conds["all_clean"]["f1_mean"]
            worst = min(((c, x["f1_mean"]) for c, x in conds.items() if c != "all_clean"),
                        key=lambda kv: kv[1])
            drop_pp = (clean - worst[1]) * 100
            print(f"  {m:18s}: clean F1={clean:.3f}, worst-drop ({worst[0]}) "
                  f"F1={worst[1]:.3f}, Δ={drop_pp:+.1f} pp")

    if (agg / "exp3_degradation.json").exists():
        d = json.load(open(agg / "exp3_degradation.json"))
        print("\n--- Experiment 3: sensor degradation (Gaussian noise) ---")
        for m, blocks in d.items():
            if not blocks:
                continue
            clean = blocks["gaussian"]["0.1"]["f1_mean"]
            worst = blocks["gaussian"]["2.0"]["f1_mean"]
            print(f"  {m:18s}: σ=0.1 F1={clean:.3f}, σ=2.0 F1={worst:.3f}, "
                  f"Δ={(clean-worst)*100:+.1f} pp")

        print("\n  Realistic corruptions:")
        for m, blocks in d.items():
            if not blocks:
                continue
            for cname, x in blocks["realistic"].items():
                print(f"    {m:18s}: {cname:14s} F1={x['f1_mean']:.3f}")

    e4 = Path("results/exp4_recon/inclusion_decision.json")
    if e4.exists():
        d = json.load(open(e4))
        print("\n--- Experiment 4: cross-modal recon (D06 inclusion check) ---")
        for c in d["checks"]:
            verdict = "PASS" if c["passes"] else "FAIL"
            print(f"  {c['test']:24s} ref={c['ref_f1']:.3f} recon={c['recon_f1']:.3f} "
                  f"Δ={c['delta_pp']:+.2f}pp std={c['std_pp']:.2f}pp -> {verdict}")
        print(f"\n  decision: include in paper = {d['include']}")

    print("\n--- Decision logs ---")
    for dpath in sorted(Path("decisions").glob("*.md")):
        print(f"  {dpath}")

    print("\n--- Figures ---")
    for fp in sorted(Path("figures").glob("fig*.pdf")):
        print(f"  {fp}")


if __name__ == "__main__":
    main()
