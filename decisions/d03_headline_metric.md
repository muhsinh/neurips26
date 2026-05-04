# Decision 03: Headline metric

**Time:** 2026-05-04 14:18 PT
**Choice faced:** Headline metric for stress classification on WESAD.
**Alternatives considered:**
- A: Binary F1 (positive class = stress) — pro: handles ~30% class imbalance, comparable to wearable-stress literature.
- B: Accuracy — pro: simple; con: misleads under imbalance.
- C: AUROC — pro: threshold-independent; con: harder to read in figure captions, less common in this literature.
- D: Balanced accuracy — pro: simple under imbalance; con: less common as headline in this domain.
**Choice:** A (binary F1, stress = positive class). Report accuracy alongside in JSON for completeness.
**Reviewer-frame reasoning:** The wearable-stress literature reports F1; reviewers comparing to prior work need F1. The class imbalance (~30% stress) makes accuracy a misleading headline.
**Logged for follow-up if:** None — F1 stays for all panels.
