"""Run Exp1 grid: 3 archs × 15 LOSO folds × 3 seeds = 135 runs.

Calls train.py for each combination. Saves checkpoints (state_dict) for Exp2/3 reuse.
Skips combinations that already have a result JSON to support resume.
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
MODELS = ["late_fusion_mlp", "cross_attention", "scale_proxy"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/exp1_loso")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--subjects", default=",".join(SUBJECTS))
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = p.parse_args()

    models = args.models.split(",")
    seeds = list(map(int, args.seeds.split(",")))
    subjects = args.subjects.split(",")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = [(m, s, sd) for m in models for s in subjects for sd in seeds]
    n = len(combos)

    t_start = time.time()
    done = 0
    skipped = 0
    for i, (model, subj, seed) in enumerate(combos, start=1):
        tag = f"{model}__{subj}__seed{seed}"
        json_path = out_dir / f"{tag}.json"
        ckpt_path = out_dir / f"{tag}.pt"
        if args.skip_existing and json_path.exists() and ckpt_path.exists():
            print(f"[{i}/{n}] {tag} -- skip (exists)")
            skipped += 1
            continue
        cmd = [sys.executable, "-m", "src.train",
               "--config", args.config,
               "--model", model,
               "--test_subject", subj,
               "--seed", str(seed),
               "--out_dir", str(out_dir),
               "--save_state"]
        t0 = time.time()
        rc = subprocess.call(cmd)
        dt = time.time() - t0
        if rc != 0:
            print(f"[{i}/{n}] {tag} FAILED rc={rc}", file=sys.stderr)
            sys.exit(rc)
        done += 1
        elapsed = time.time() - t_start
        eta = elapsed / done * (n - i - skipped) if done > 0 else 0
        print(f"  [{i}/{n}] done={done} skipped={skipped} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    print(f"\nDone. {done} runs, {skipped} skipped. Total {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()
