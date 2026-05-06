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

## Dataset source

WESAD was published by Schmidt et al. (ICMI 2018) and listed on the UCI ML
Repository as dataset id 465. As of May 2026 the UCI page hosts a 261-byte
pointer-zip whose `WESAD.txt` redirects to a now-dead Sciebo share
(`https://uni-siegen.sciebo.de/s/pYjSgfOVs6Ntahr/download` returns HTTP 404,
as does the alternate share token `HGdUkoNlW9Ld0u9`). The official upstream
is therefore not currently downloadable.

Until upstream is restored, the only working public mirror we found is the
Kaggle dataset `mohamedasem318/wesad-full-dataset` (2.6 GB, CC BY 4.0,
2,280+ downloads at time of use). We verified the content matches the
Schmidt et al. specification:

- 15 subject directories: S2-S11, S13-S17 (no S1, no S12 — matches paper).
- Each `SX.pkl` loads with `pickle.load(f, encoding='latin1')` (matches paper
  Python-2 pickle format).
- Per-subject recording length 6079 seconds, with chest signals at 700 Hz,
  wrist BVP at 64 Hz, wrist EDA/TEMP at 4 Hz (matches paper hardware spec).
- Label distribution per subject: stress fraction ≈ 0.27-0.31 across all 15
  subjects (matches expected baseline:stress:amusement ratios from Schmidt et al.).

For verification, the SHA-256 of two canonical pickles in our copy:

```
36ef5e8afc0f91998eefba7c12fc9fa97b7b07198cbec0126917d7abb436ca23  S2/S2.pkl
3315796a75227d54d7b0056736f671484fd2fb85afffa65818fd76aeff2920fa  S17/S17.pkl
```

If you obtain WESAD from a different source (restored UCI, archived Sciebo,
direct from authors), `python -m src.sanity --raw_dir data/raw/WESAD` will
fail loudly if the data does not match the expected structure.

## Reproduce

Each step is a single shell command. No manual edits required.

```bash
# 1. Download WESAD. PRIMARY: Kaggle mirror (2.6 GB, ~5 min, requires ~/.kaggle/kaggle.json).
kaggle datasets download -d mohamedasem318/wesad-full-dataset -p data/raw/ --force
cd data/raw && unzip -q wesad-full-dataset.zip && cd ../..

# Alternative: official UCI source at
#   https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection
# Place the unzipped subject directories at data/raw/WESAD/SX/SX.pkl.
# Note: UCI currently serves only a pointer-zip; the actual data is hosted on
# a third-party share that was offline at time of use (May 2026). Sanity-check
# step 3 below will confirm whichever copy you obtain matches the paper spec.

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
