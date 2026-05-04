# Decision 01: Window length

**Time:** 2026-05-04 14:18 PT
**Choice faced:** Length of non-overlapping window for stress classification.
**Alternatives considered:**
- A: 60s — pro: Schmidt et al. baseline + WESAD-literature standard, enough samples per window for HRV-relevant stats; con: ~30-50 windows per subject is small.
- B: 30s — pro: doubles window count per subject, less context; con: weaker HRV stats, diverges from literature.
- C: 10s — pro: many windows; con: undermines stress-state claim, hard to compare to prior work.
**Choice:** A (60s).
**Reviewer-frame reasoning:** Sticking with WESAD-literature standard makes per-subject collapse claim defensible against "your protocol is non-standard" objection. The point is the failure mode, not benchmark-pushing.
**Logged for follow-up if:** Any subject yields <30 windows post-filtering — escalate to user.
