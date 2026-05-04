# Decision 02: Subject normalization scheme

**Time:** 2026-05-04 14:18 PT
**Choice faced:** How to normalize per-subject signals.
**Alternatives considered:**
- A: Per-subject z-score on that subject's own full data — pro: WESAD-literature standard, no leakage from training-fold composition; con: held-out subject "sees" its own statistics at test time (but does not see labels — accepted in DG literature).
- B: Global z-score using train-fold subjects only — pro: more "principled" in some DG framings; con: confounds the per-subject collapse story with feature-distribution shift, non-standard for WESAD.
- C: No normalization — pro: surfaces raw inter-subject variance; con: hurts all models uniformly, muddies architecture comparison.
**Choice:** A (per-subject z-score using each subject's own full-recording stats).
**Reviewer-frame reasoning:** Per-subject z-score is the canonical preprocessing for WESAD across the literature. It is also the easier story: "even after standard normalization that gives every subject identical means/stds, models still collapse on certain subjects" — pinning failure on something other than feature scale.
**Logged for follow-up if:** Reviewer pre-empt requires a global-norm comparison — possible side-experiment in Sec 4 (out of scope for this run).
