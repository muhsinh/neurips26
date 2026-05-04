# Decision 07: Figure 2 visual choices — error bars, bands, raw points

**Time:** 2026-05-04 14:35 PT
**Choice faced:** How to encode variance in each Figure 2 panel.
**Alternatives considered:**
- Panel A (per-subject collapse): box plot vs. strip plot of raw points vs. bar+error.
- Panel B (modality dropout): grouped bars with error bars vs. heatmap.
- Panel C (noise sweep): error bars at each point vs. shaded ±1 std band.
**Choices:**
- Panel A: strip plot — each subject is a dot, with thin vertical lines for seed std.
- Panel B: grouped bars with caps (D10 may switch to heatmap if results are clean).
- Panel C: line + shaded ±1 std band.
**Reviewer-frame reasoning:** The per-subject collapse argument *requires* showing the distribution of subjects, not the central tendency. A boxplot would hide the bimodal structure that makes "per-subject collapse" visible. Bands work better than error bars when the curve has 4 points — they communicate the trend without visual noise. Grouped bars are easier to compare precisely than heatmap for a small condition count.
**Logged for follow-up if:** Panel B gets cluttered at 11 conditions — switch to heatmap (see D10).
