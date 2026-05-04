# Decision 06: Exp4 (cross-modal recon) inclusion threshold

**Time:** 2026-05-04 14:30 PT
**Choice faced:** What effect size justifies including the cross-modal recon variant in the paper.
**Alternatives considered:**
- A: ≥3 pp absolute F1 improvement averaged over seeds AND beyond 1 std seed-fold variance, on at least one of {worst-case dropout, high-noise condition}. Conservative.
- B: Any positive directional effect, even <1 pp. Liberal.
- C: ≥5 pp + statistical-significance test (e.g., paired bootstrap). Very conservative.
**Choice:** A.
**Reviewer-frame reasoning:** No rebuttal cycle, so a 1-2pp claim invites "is it noise?" objections we can't defend. The 3pp+1std bar is conservative enough that a positive result reads as a real signal. If the variant fails, we cut and write the discussion as "naive instantiations of the principle are insufficient" — honest and defensible.
**Logged for follow-up if:** Effect lands at 2-3pp and review feels close — escalate to user before committing to inclusion or exclusion.
