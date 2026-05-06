# Figure Captions (with real numbers)

## Figure 1 — Thesis schematic

> **Figure 1. Three failure modes of multi-sensor fusion are three faces of one
> domain-generalization problem.** From left to right: *(1)* Per-subject collapse
> — leave-one-subject-out evaluation reveals catastrophic failure on a subset
> of subjects that aggregate accuracy hides; *(2)* Modality shortcut — removing
> a single sensor at test time collapses prediction, evidence that the model
> leaned on one modality rather than fusing all four; *(3)* Sensor degradation
> — realistic test-time noise drives F1 toward chance across architectures.
> All three are instances of distribution shift that current fusion architectures
> do not bridge. The proposed direction is neuro-inspired architectural priors
> (inset: cross-modal generative, where each modality must be predictable from
> the others through a shared latent).

## Figure 2 — Diagnostic results on WESAD

> **Figure 2. Three failure modes of multi-sensor fusion on WESAD.**
> **(A)** Per-subject F1 under leave-one-subject-out cross-validation. Each
> point is one held-out subject; vertical lines show ±1 std across 3 training
> seeds; horizontal bar is the per-architecture mean. Subject S17 collapses
> to F1 ≈ 0.04 across all three architectures, hidden by the aggregate mean
> of 0.72–0.81.
> **(B)** Test-time modality dropout (zero-mask at inference). Each bar is
> the mean F1 across the 3 seeds × 15 LOSO folds (45 evaluations per condition);
> error bars are ±1 standard deviation across that pooled seed-fold sample
> (the per-subject and per-seed contributions are not separated). Removing
> the BVP+EDA pair collapses F1 by 50–53 pp for the late-fusion MLP and
> cross-attention transformer; the scale-proxy is even worse, losing 61 pp
> when EDA+TEMP are removed. EDA is the dominant single-modality shortcut
> across all three architectures (see Figure 3B and Exp 5).
> **(C)** Test-time additive Gaussian noise on all modalities. Lines show
> mean F1 across 3 seeds × 15 folds; shaded bands are ±1 std across the same
> pooled sample. F1 drops by 33–35 pp from σ=0.1 to σ=2.0 for late-fusion
> and cross-attention; scale-proxy is roughly flat — an artefact of
> frozen randomly-initialised encoders saturating at large input magnitudes,
> not a genuine fix to the underlying brittleness (see Figure 4 caption).
> *N = 15 subjects (S2–S11, S13–S17), 3 seeds {42, 1337, 2024}, binary
> stress vs. non-stress, 60 s windows at 4 Hz, per-subject z-score
> normalization, leave-one-subject-out CV (135 trained checkpoints).*

## Figure 3 — Unification

> **Figure 3. The three failure modes share a common subject- and
> modality-level structure.** **(A)** Each point is one subject (mean over
> the three architectures and three seeds): x = clean LOSO F1; y = worst-case
> modality-dropout F1; colour = high-noise (σ=2.0) F1. The pink-shaded
> bottom-left rectangle highlights subjects S14 and S17, which collapse on
> every failure mode — that cluster, not the global trend, is the unification
> claim. Cross-subject correlation between clean F1 and worst-dropout F1
> across all 15 subjects is weak in aggregate (Spearman ρ = +0.17, Pearson
> r = +0.22); the strong cluster signal is intentionally separated from the
> aggregate correlation in the panel.
> **(B)** F1 drop induced by removing each single modality, grouped by
> architecture. EDA dominates across all three: the late-fusion MLP loses
> 0.42 F1, cross-attention 0.39, scale-proxy 0.57. ACC, BVP, and TEMP are
> close to zero impact. The shared shortcut across architectures is direct
> evidence of a structural rather than architecture-specific failure.
> Modality-importance rankings on the dual-modality dropout conditions
> (6 pair-drops) agree at Spearman ρ ≥ 0.886 across every architecture pair
> (95% bootstrap CI lower-bound ≥ 0.60, permutation p ≤ 0.016 — see Exp 5
> in the appendix and decisions/d13). The single-modality Spearman is weaker
> (ρ ∈ [0.20, 0.80]) because three of four modalities have near-zero impact
> and rank-tie unstably; Pearson r exceeds 0.99 across pairs in that case.
> *Same 135 checkpoints, same protocol as Figure 2.*

## Figure 4 — Scale alone does not fix the failure modes

> **Figure 4. Three failure modes persist across a ~37× sweep in trainable
> parameter count.** Architectures compared: the late-fusion MLP
> (~50 k parameters, all trainable), the cross-attention transformer
> (~150 k parameters, all trainable), and a frozen randomly-initialised
> scale proxy (~1.85 M total parameters, ~130 k trainable head — see D05).
> **(A)** Per-subject F1 — the bimodal distribution and per-subject collapse
> persist across all three; the scale-proxy collapses on 2/15 subjects vs.
> 1/15 for the others.
> **(B)** F1 under the worst-case test-time modality dropout per architecture
> (each bar is the mean across 3 seeds × 15 folds for that architecture's
> worst-case dropout condition; error bars are ±1 std clipped to ≤0.20).
> The modality shortcut persists across the parameter sweep; the scale-proxy
> is the worst (F1 ≈ 0.11 vs ≈ 0.28 for the trained-encoder architectures),
> and the gap is within seed-fold variance.
> **(C)** F1 drop from clean (σ=0.1) to high-noise (σ=2.0) for the two
> *trained-encoder* architectures (late-fusion 35 pp, cross-attention 33 pp).
> The scale-proxy is intentionally omitted from this panel: its frozen
> randomly-initialised encoder maps large-amplitude noisy inputs into nearly
> the same feature region as clean inputs (saturation), producing an
> artefactual ≈0 drop (raw σ=2.0 F1 = 0.732 vs clean 0.721, Δ = +0.011) that
> would misrepresent structural robustness if plotted alongside the trained
> encoders. Scale-proxy framing as a "capacity-and-depth match for what
> bigger pretrained models would do" — not a true pretrained encoder — is
> documented in decisions/d05.

## Note on Experiment 4 (cross-modal reconstruction) exclusion

> Per inclusion threshold D06 (≥3 pp absolute F1 improvement AND beyond
> 1 std seed-fold variance), the cross-modal reconstruction variant did not
> reach the bar:
> - Worst-case dropout: Δ = +0.62 pp (well below 3 pp).
> - High-noise σ=2.0: Δ = +3.50 pp (passes 3 pp absolute, fails 1-std variance
>   check; std = 8.99 pp).
>
> The trained checkpoints, evaluation JSONs, and decision logs are retained
> in `results/exp4_recon/` for inspection. The discussion section frames this
> as evidence that naive instantiations of the cross-modal generative principle
> (a single auxiliary loss bolted onto a standard fusion stack) are
> insufficient — the right architectural commitment requires deeper changes
> than this experiment tests.
