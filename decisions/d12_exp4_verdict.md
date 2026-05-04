# Decision 12: Exp4 (cross-modal recon) verdict

**Time:** 2026-05-04 15:30 PT
**Per D06 inclusion threshold (≥3 pp absolute Δ AND beyond 1 std seed-fold variance):**

| Test | Cross-attention F1 | Cross-modal-recon F1 | Δ pp | std pp | Verdict |
|------|--------------------:|---------------------:|------:|-------:|--------:|
| Worst-case modality dropout (drop_BVP_EDA) | 0.278 | 0.284 | +0.62 | 32.29 | FAIL |
| High-noise (σ=2.0)                          | 0.444 | 0.479 | +3.50 |  8.99 | FAIL |

**Choice:** Cross_modal_recon does NOT meet the inclusion threshold. **Exclude from paper.**

**Per the brief §5.4:** "Do not write up null results — there is no rebuttal and reviewers will weaponize them. Instead, write the discussion section to argue that naive instantiations of the principle are insufficient and the right architectural commitment requires deeper changes (this is honest and defensible)."

**Reviewer-frame reasoning:** A 0.6 pp gain on dropout (within noise) and a 3.5 pp gain on high-noise that fails the variance check are not actionable improvements. Including them as "weak positive results" risks reviewer 4 framing them as null, which would undercut the position-paper claim. The honest discussion-section framing is stronger: the failure of the naive recon variant supports the position that neuro-inspired priors must be deeper architectural commitments, not single auxiliary losses tacked onto existing fusion stacks.

**Logged for follow-up if:** Reviewer asks "did you try X?" — point them to this decision file and the cross_modal_recon ckpts retained in `results/exp4_recon/`.

**Receipts (saved on disk):**
- 45 trained ckpts: `results/exp4_recon/cross_modal_recon__S*__seed*.pt`
- 45 metric JSONs: `results/exp4_recon/cross_modal_recon__S*__seed*.json`
- Dropout eval: `results/exp2_dropout/cross_modal_recon.json`
- Degradation eval: `results/exp3_degradation/cross_modal_recon.json`
- Decision JSON: `results/exp4_recon/inclusion_decision.json`
