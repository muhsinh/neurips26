"""Run Exp2 (dropout) and Exp3 (degradation) evaluations on Exp4 cross_modal_recon ckpts.

Outputs to results/exp2_dropout/cross_modal_recon.json and
results/exp3_degradation/cross_modal_recon.json so the existing aggregator + decision
script can compare against cross_attention reference.
"""
from __future__ import annotations
import subprocess
import sys


def main():
    rc = subprocess.call([
        sys.executable, "-m", "src.eval_dropout",
        "--models", "cross_modal_recon",
        "--exp1_dir", "results/exp4_recon",
        "--out_dir", "results/exp2_dropout",
    ])
    if rc != 0:
        sys.exit(rc)
    rc = subprocess.call([
        sys.executable, "-m", "src.eval_degradation",
        "--models", "cross_modal_recon",
        "--exp1_dir", "results/exp4_recon",
        "--out_dir", "results/exp3_degradation",
    ])
    sys.exit(rc)


if __name__ == "__main__":
    main()
