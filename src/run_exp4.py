"""Conditional Exp4: cross-modal reconstruction variant.

Train cross_modal_recon under same LOSO protocol with all 3 seeds.
Then re-run dropout + degradation evaluations on these checkpoints.
Apply D06 inclusion threshold (≥3 pp + beyond 1 std seed-fold variance).
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
            "S10", "S11", "S13", "S14", "S15", "S16", "S17"]
SEEDS = [42, 1337, 2024]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/exp4_recon")
    p.add_argument("--config", default="configs/base.yaml")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = [(s, sd) for s in SUBJECTS for sd in SEEDS]
    n = len(combos)
    t_start = time.time()
    for i, (subj, seed) in enumerate(combos, start=1):
        tag = f"cross_modal_recon__{subj}__seed{seed}"
        json_path = out_dir / f"{tag}.json"
        ckpt_path = out_dir / f"{tag}.pt"
        if json_path.exists() and ckpt_path.exists():
            print(f"  [{i}/{n}] {tag} -- skip"); continue
        cmd = [sys.executable, "-m", "src.train",
               "--config", args.config,
               "--model", "cross_modal_recon",
               "--test_subject", subj,
               "--seed", str(seed),
               "--out_dir", str(out_dir),
               "--save_state"]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  [{i}/{n}] {tag} FAILED rc={rc}", file=sys.stderr)
            sys.exit(rc)
        elapsed = time.time() - t_start
        eta = elapsed / i * (n - i)
        print(f"  [{i}/{n}] elapsed={elapsed:.0f}s eta={eta:.0f}s")
    print(f"\nDone. {n} runs in {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()
