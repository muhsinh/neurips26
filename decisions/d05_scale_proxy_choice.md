# Decision 05: Scale-proxy implementation

**Time:** 2026-05-04 14:25 PT
**Choice faced:** How to instantiate the "scale proxy" architecture so we can rebut
"but bigger pretrained models would fix this."
**Alternatives considered:**
- A: True pretrained physiological encoder (PaPaGei, BIOT, MOMENT-1-large, Chronos, Timer). Pro: most defensible. Con: time-boxed search risk (30 min); some require GPU; some mismatch input sampling rate (4 Hz here is non-standard).
- B: Frozen randomly-init larger transformer per modality (4 layers, 256 hidden, ~2M params per modality) with only the fusion head trained. Pro: depth + parameter count match scaling-skeptic intuition; reproducible; no external download dependency. Con: technically not "pretrained" — must be framed honestly.
- C: Skip third architecture entirely; argue structurally instead. Pro: cleaner story. Con: leaves the most predictable reviewer objection unaddressed.
**Choice:** B (frozen randomly-init larger transformer). Total ~9.2M params, ~0.5M trainable head. Frame in paper as "scale proxy (frozen large random init)" not "pretrained."
**Reviewer-frame reasoning:** A genuinely pretrained model would be marginally better, but the structural argument doesn't depend on it. The proxy gives us the parameter count and depth. Honest framing forecloses "you used random init not pretrained" as a critique by acknowledging it upfront.
**Logged for follow-up if:** Reviewer feedback indicates a true pretrained encoder is required — could swap in an HF-hosted physio model post-submission.
