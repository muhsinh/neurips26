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
> **(B)** Test-time modality dropout. Removing the BVP+EDA pair collapses F1
> by 50–53 pp for the late-fusion MLP and cross-attention transformer; the
> scale-proxy is even worse, losing 61 pp when EDA+TEMP are removed. EDA is
> the dominant single-modality shortcut across all three architectures.
> **(C)** Test-time additive Gaussian noise on all modalities. F1 drops by
> 33–35 pp from σ=0.1 to σ=2.0 for the late-fusion and cross-attention
> architectures. The scale-proxy is roughly flat — an artifact of frozen
> randomly-initialized encoders saturating at large input magnitudes, not a
> genuine fix to the underlying brittleness (see Figure 4 caption).
> *N = 15 subjects (S2–S11, S13–S17), 3 seeds {42, 1337, 2024}, binary
> stress vs. non-stress, 60 s windows at 4 Hz, per-subject z-score
> normalization, leave-one-subject-out CV (135 trained checkpoints).*

## Figure 3 — Unification

> **Figure 3. The three failure modes share a common subject- and
> modality-level structure.** **(A)** Each point is one subject; x-axis is
> clean LOSO F1 averaged over architectures and seeds, y-axis is worst-case
> modality-dropout F1, color is high-noise F1. Subjects S14 and S17 are in
> the bottom-left "fragile across all three failure modes" cluster (highlighted).
> Spearman ρ = +0.17 (Pearson r = +0.22) reflects the diagonal trend of the
> robust majority; the unification claim is most clearly visible in the bottom-left
> cluster, where collapse subjects fail under every condition.
> **(B)** F1 drop induced by removing each single modality, grouped by
> architecture. EDA dominates across all three: the late-fusion MLP loses
> 0.42 F1, cross-attention 0.39, scale-proxy 0.57. ACC, BVP, and TEMP are
> close to zero impact. The shared shortcut across architectures is direct
> evidence of a structural rather than architecture-specific failure.
> *Same 135 checkpoints, same protocol as Figure 2.*

## Figure 4 — Scale alone does not fix the failure modes

> **Figure 4. Three failure modes persist across a ~37× sweep in trainable
> parameter count.** Comparing late-fusion MLP (~50 k params, ~50 k trainable),
> cross-attention transformer (~150 k params, all trainable), and a frozen
> randomly-initialized scale proxy (~1.85 M total, ~130 k trainable head):
> **(A)** Per-subject F1 — the bimodal distribution and per-subject collapse
> persist across all three; scale-proxy collapses 2/15 vs. 1/15 for the others.
> **(B)** F1 under the worst-case test-time modality dropout — scale-proxy
> is *worse* (F1 ≈ 0.11) than both smaller architectures (≈ 0.28). The shortcut
> is amplified, not eliminated, by parameter count.
> **(C)** F1 drop from clean (σ=0.1) to high-noise (σ=2.0). Late-fusion (35 pp)
> and cross-attention (33 pp) degrade as expected; scale-proxy is flat, but
> this reflects frozen-encoder feature saturation rather than scale-driven
> robustness — the encoder maps large-amplitude noisy inputs into nearly
> the same feature region as clean inputs, an artifact of random
> initialisation rather than a fix to the structural failure.
> *Scale proxy is randomly-initialized then frozen, with only the fusion head
> trained — framed honestly as a capacity-and-depth match for "what bigger
> models would do," not a true pretrained encoder. See decisions/d05.*

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
