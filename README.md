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

```bash
# 1. Download WESAD (Kaggle CLI; 2.6 GB, ~5 min)
kaggle datasets download -d mohamedasem318/wesad-full-dataset -p data/raw/ --force
cd data/raw && unzip -q wesad-full-dataset.zip && cd ../..

# 2. Preprocess (4 Hz wrist modalities, 60s windows, per-subject z-score)
python -m src.preprocess --raw_dir data/raw/WESAD --out_dir data/processed

# 3. Sanity checks 3.1-3.5
python -m src.sanity --raw_dir data/raw/WESAD --processed_dir data/processed

# 4. Exp1: 3 archs × 15 LOSO folds × 3 seeds (~60 min CPU)
python -m src.run_exp1

# 5. Exp2 (dropout) and Exp3 (degradation) — inference only on Exp1 checkpoints
python -m src.eval_dropout
python -m src.eval_degradation

# 6. Aggregate
python -m src.aggregate

# 7. Render figures
python -m src.figures.fig1_thesis
python -m src.figures.fig2_diagnostic
python -m src.figures.fig3_unification
python -m src.figures.fig4_scale
```

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

## Reproducibility

- Seeds: `{42, 1337, 2024}`. Every run logs seed, config, git commit, model state.
- Device: CPU (M3 Pro 18 GB RAM). MPS fallback enabled but unused for variance.
- All result JSONs include the full config + git short hash for traceability.
