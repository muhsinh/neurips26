# Figure Captions

Caption template (per brief §6.8):
> Figure N. **[Bold one-sentence claim]**. [One sentence per panel]. [One sentence on takeaway]. [Methodological note: N subjects, 3 seeds, LOSO CV, etc.]

---

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
> **(A)** Per-subject F1 under leave-one-subject-out cross-validation reveals
> catastrophic collapse on a subset of subjects, hidden by the aggregate mean
> (horizontal bar). Each point is one held-out subject; vertical lines show
> ±1 standard deviation across 3 training seeds.
> **(B)** Test-time modality dropout: removing certain wrist modalities collapses
> F1 across all three architectures, evidence of shortcut learning rather than
> genuine multi-modal fusion.
> **(C)** Test-time additive Gaussian noise on all modalities degrades F1 toward
> chance for every architecture; shaded bands show ±1 std across 3 seeds × 15 folds.
> The three patterns persist across architectures spanning ~150× parameter count,
> suggesting a structural rather than capacity-driven failure.
> *N = 15 subjects, 3 seeds {42, 1337, 2024}, binary stress vs. non-stress
> classification, 60 s windows at 4 Hz, per-subject z-score normalization.*

## Figure 3 — Unification

> **Figure 3. The three failure modes share a common subject- and
> modality-level structure.** **(A)** Each point is one (subject, architecture)
> pair: x = clean F1 under LOSO; y = F1 under the worst-case modality dropout
> for that pair; color = F1 under high-noise conditions. Subjects who collapse
> under one failure mode tend to collapse under the others (Pearson r ≈ [filled
> in from data]). **(B)** F1 drop induced by removing each single modality,
> grouped by architecture. All three architectures rank the modalities in the
> same order, with one modality dominating the contribution to the prediction —
> evidence of a shared shortcut, not architecture-specific heuristics.
> *Same data and protocol as Figure 2.*

## Figure 4 — Scale doesn't fix it

> **Figure 4. The three failure modes persist across a ~150× sweep in parameter
> count.** Three vertically stacked panels compare a small late-fusion MLP
> (~50 k parameters), a medium cross-attention transformer (~150 k), and a
> frozen randomly-initialized scale proxy (~9 M total, ~0.5 M trainable):
> **(A)** Per-subject F1 — the bimodal distribution and per-subject collapse
> persist across all three; **(B)** F1 under the worst-case modality dropout
> — the shortcut is not eliminated by scale; **(C)** F1 under high-amplitude
> Gaussian noise — degradation brittleness persists. The structural failure
> survives a model-size sweep that covers more than two orders of magnitude.
> *Scale proxy is randomly-initialized then frozen, with only the fusion head
> trained — framed honestly as a capacity-and-depth match for "what bigger
> models would do," not a true pretrained encoder. See decisions/d05.*
