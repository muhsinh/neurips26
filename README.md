# NeurIPS 2026 Position Paper — Empirical Section

Diagnostic experiments demonstrating three connected failure modes of multi-sensor
fusion architectures on the WESAD wearable affective computing benchmark:

1. **Per-subject collapse** under leave-one-subject-out evaluation
2. **Modality shortcut** under test-time modality dropout
3. **Sensor degradation brittleness** under realistic noise/corruption

Optionally tests a **cross-modal reconstruction** auxiliary loss as an architectural intervention.

## Setup

```bash
cd project
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch numpy scipy scikit-learn matplotlib pandas pyyaml requests
```

## Reproduce

Each step is a single shell command. No manual edits required.

```bash
# 1. Download WESAD (Kaggle CLI; 2.6 GB, ~5 min). Requires ~/.kaggle/kaggle.json.
kaggle datasets download -d mohamedasem318/wesad-full-dataset -p data/raw/ --force
cd data/raw && unzip -q wesad-full-dataset.zip && cd ../..

# 2. Preprocess (4 Hz wrist modalities, 60s windows, per-subject z-score)
python -m src.preprocess --raw_dir data/raw/WESAD --out_dir data/processed

# 3. Sanity checks 3.1-3.5
python -m src.sanity --raw_dir data/raw/WESAD --processed_dir data/processed

# 4. Exp 1: late_fusion_mlp + cross_attention training (90 LOSO runs, ~12 min CPU)
python -m src.run_exp1

# 5. Exp 1 (continued): scale_proxy fast trainer (45 LOSO runs, ~1 min)
python -m src.train_scale

# 6. Exp 2 (dropout) and Exp 3 (degradation) — inference only on the 135 ckpts
python -m src.eval_dropout
python -m src.eval_degradation

# 7. Exp 4: cross_modal_recon training + dropout/noise eval (45 LOSO runs, ~8 min)
python -m src.run_exp4
python -m src.run_exp4_evals
python -m src.exp4_decision

# 8. Tier-1 reviewer-hardening analyses (no retraining, ~1 min total)
python -m src.exp5_modality_consistency
python -m src.exp6_bootstrap_ci

# 9. Exp 7: Gulrajani-style HP sweep (180 trainings, ~22 min CPU)
python -m src.exp7_erm_tuning
python -m src.exp7_evaluate_selected

# 10. Aggregate + summarise
python -m src.aggregate
python -m src.summary
python -m src.tier1_summary

# 11. Patch aggregate JSONs with provenance metadata (idempotent)
python -m src.patch_metadata

# 12. Render figures
python -m src.figures.fig1_thesis
python -m src.figures.fig2_diagnostic
python -m src.figures.fig3_unification
python -m src.figures.fig4_scale
```

Total wall-clock from cold cache: ~50 min (download + preprocess) + ~45 min
(all training) + ~5 min (analyses + figures) ≈ 1.7 hours.

If checkpoints already exist, every script skips re-running so the figure
re-render path is < 30 seconds.

## Determinism

- Single-fold deterministic re-run verified: identical predictions and metrics
  vs. saved JSON for one (architecture, fold, seed) tuple (Check 3 pass).
- Bootstrap and permutation analyses (Exp 5, Exp 6) produce byte-identical
  output on rerun (Check 4 pass).
- All seeds parameterised: training seeds in `configs/base.yaml`, analysis
  seeds default to 42, exposed via `--seed` flags where applicable.

## Layout

```
data/raw/WESAD/SX/SX.pkl  — UCI raw subject pickles
data/processed/SX.npz     — 4 Hz windowed cache
src/{preprocess,dataset,train,eval_*,aggregate,sanity}.py
src/models/{late_fusion_mlp,cross_attention,scale_proxy,cross_modal_recon}.py
src/figures/{style,fig1_thesis,fig2_diagnostic,fig3_unification,fig4_scale}.py
results/{exp1_loso,exp2_dropout,exp3_degradation,exp4_recon,agg}/
figures/{fig1_thesis,fig2_diagnostic,fig3_unification,fig4_scale}.{pdf,png}
decisions/dXX_*.md         — reviewer-check decision logs
```

## Configuration

`configs/base.yaml` is the single source of truth for hyperparameters, seeds,
modalities, dropout conditions, and noise levels. Every script accepts
`--config configs/base.yaml`.

## Key choices (decision logs in `decisions/`)

| ID | Decision | Choice |
|----|----------|--------|
| D01 | Window length | 60 s (matches Schmidt et al.) |
| D02 | Subject normalization | Per-subject z-score on own data |
| D03 | Headline metric | Binary F1 (stress = positive class) |
| D04 | Modality dropout method | Zero-mask (≈ mean-mask post z-score) |
| D05 | Scale-proxy implementation | Frozen randomly-init large transformer |
| D06 | Exp4 inclusion threshold | ≥3 pp F1 + beyond 1 std seed-fold variance |
| D07-D11 | Figure visual choices | strip + grouped bars + log noise + per-subject means |
| D12 | Exp4 verdict | FAIL D06 → exclude from paper |
| D13 | Exp5 modality consistency | dual-mod ρ ≥ 0.886 all pairs (single-mod underpowered) |
| D14 | Exp6 bootstrap CIs | S17 confirmed in 6 of 9 (arch×seed); S14 softened |
| D15 | Exp7 ERM tuning | position holds + val-vs-test gap finding |

## Reproducibility

- Seeds: `{42, 1337, 2024}`. Every run logs seed, config, git commit, model state.
- Device: CPU (M3 Pro 18 GB RAM). MPS fallback enabled but unused for variance.
- All result JSONs include the full config + git short hash for traceability.
