# Decision 04: Modality dropout — zero-mask vs mean-mask vs learned-mask

**Time:** 2026-05-04 14:25 PT
**Choice faced:** How to remove a modality at test time for Experiment 2.
**Alternatives considered:**
- A: Zero-mask (set all values of that modality to 0). Pro: standard ablation in robustness literature; reproducible; clean "remove the input" signal. Con: zero is in-distribution since we z-scored.
- B: Mean-mask (replace with that modality's training-distribution mean). Pro: arguably more "realistic" sensor failure (degraded but valid baseline). Con: introduces a confound — model may have learned to handle mean as a token; mean-imputation also leaks training-set info.
- C: Learned mask token. Pro: matches modern transformer practice. Con: overkill for ablation; trains a task-specific token that confounds the diagnostic interpretation.
**Choice:** A (zero-mask) for the headline plot. Mean-mask reserved for limitations section if reviewer pressure feels likely.
**Reviewer-frame reasoning:** Zero-mask is the canonical robustness ablation; reviewers questioning the choice can be answered with "this matches X et al." Mean-mask would muddy the modality-shortcut interpretation by mixing "removed" and "replaced" signals.
**Note:** Inputs are per-subject z-scored, so the mean of each modality across the subject is ~0. Zero-mask therefore approximates mean-mask in practice — the "confound" critique of mean-mask is reduced. Worth flagging this in the methods section.

**Logged for follow-up if:** Time permits, run a tiny side-check with mean-masking on cross_attention only and report deltas in the appendix (low priority).
