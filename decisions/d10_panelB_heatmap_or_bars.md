# Decision 10: Panel B (modality dropout) — heatmap or grouped bars

**Time:** 2026-05-04 14:35 PT (will revisit after Exp2 results land)
**Choice faced:** 11 dropout conditions × 3 architectures — bars or heatmap?
**Alternatives considered:**
- A: Grouped bars (3 archs × 11 conds). Pro: precise comparison; readable at this scale.
- B: Heatmap (3 rows × 11 cols, color = F1 drop). Pro: shows the cross-architecture pattern at a glance — the "all archs shortcut on the same modality" story.
**Choice (preliminary):** A (grouped bars). Switch to B if Exp2 results show a clean and stark pattern where the heatmap reads better at a glance.
**Reviewer-frame reasoning:** Decide after seeing the data. Grouped bars are safer (precise comparison); heatmap is bolder (pattern visibility) but only works if the pattern is stark.
**Logged for follow-up if:** Pattern is clear and uniform — switch to heatmap for stronger visual story.
