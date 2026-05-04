# Decision 08: Sample-level vs window-level F1 in Panel A

**Time:** 2026-05-04 14:35 PT
**Choice faced:** Granularity of the per-subject F1 metric for Figure 2 Panel A.
**Alternatives considered:**
- A: Window-level F1, computed on the held-out subject's predictions only. Per-subject scalar = F1 of all that subject's windows.
- B: Aggregate windows across all held-out subjects, compute one F1 per LOSO round-robin. Hides per-subject variance.
- C: Sample-level (per-time-step) F1. Higher resolution but unconventional for windowed prediction.
**Choice:** A.
**Reviewer-frame reasoning:** The argument *is* "per-subject collapse." Anything that aggregates across subjects defeats the point of the panel. Window-level F1 per subject is the right unit because each window is one classifier prediction.
**Logged for follow-up if:** None — A stays.
