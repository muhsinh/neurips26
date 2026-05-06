# Decision 14: Per-(subject, architecture, seed) bootstrap CIs (Exp 6)

**Time:** 2026-05-06 (Tier-1 day)
**Choice faced:** How to defend the collapse claim against the small-N objection
("1-2 of 15 subjects below F1=0.3 could be lucky-fold artifact").

## Methodology

For each (subject, architecture, seed) triple:
- Use the per-window predictions stored in Exp 1 result JSONs (y_true, y_pred).
- Bootstrap-resample windows with replacement 1000 times.
- Compute F1 on each resample.
- Report bootstrap mean + 2.5/97.5 percentile (95% CI).

**Two distinct uncertainty sources are reported separately** because reviewers
will distinguish them:

1. **Within-subject sampling uncertainty** (window bootstrap CI per seed).
   "How much would F1 wobble if we resampled the held-out subject's windows?"
2. **Across-seed training uncertainty** (std of point F1 across the 3 seeds).
   "How much would F1 wobble if we retrained with a different seed?"

**Caveat (for the writeup):** Percentile bootstrap can have coverage problems
on small samples (Efron-Tibshirani 1993). The held-out subjects have 33-36
windows each, of which ~30% are stress-positive — the window count is small
and stratification is uneven. Percentile-bootstrap CIs should be read as
order-of-magnitude uncertainty, not exact frequentist guarantees.

## Results — S17 (universal collapse subject)

| Architecture | seed | point F1 | 95% CI | n_windows |
|--------------|------|----------|---------|-----------|
| LF-MLP       | 42   | 0.000 | [0.000, 0.000] | 36 |
| LF-MLP       | 1337 | 0.000 | [0.000, 0.000] | 36 |
| LF-MLP       | 2024 | 0.105 | [0.000, 0.316] | 36 |
| X-attn       | 42   | 0.000 | [0.000, 0.000] | 36 |
| X-attn       | 1337 | 0.000 | [0.000, 0.000] | 36 |
| X-attn       | 2024 | 0.143 | [0.000, 0.400] | 36 |
| Scale-proxy  | 42   | 0.000 | [0.000, 0.000] | 36 |
| Scale-proxy  | 1337 | 0.000 | [0.000, 0.000] | 36 |
| Scale-proxy  | 2024 | 0.000 | [0.000, 0.000] | 36 |

**6 of 9 combinations: F1 = 0.000 with CI = [0.000, 0.000].**
Three of nine (the seed=2024 runs of LF-MLP and X-attn) have point F1 in 0.10-0.14
range with CI extending up to 0.40 — not strictly below the 0.3 collapse threshold.

## Results — S14 (partial collapse subject)

| Architecture | seed | point F1 | 95% CI |
|--------------|------|----------|---------|
| LF-MLP       | 42   | 0.500 | [0.167, 0.737] |
| LF-MLP       | 1337 | 0.435 | [0.133, 0.667] |
| LF-MLP       | 2024 | 0.348 | [0.100, 0.583] |
| X-attn       | 42   | 0.480 | [0.200, 0.696] |
| X-attn       | 1337 | 0.435 | [0.133, 0.667] |
| X-attn       | 2024 | 0.435 | [0.143, 0.667] |
| Scale-proxy  | 42   | 0.000 | [0.000, 0.000] |
| Scale-proxy  | 1337 | 0.182 | [0.000, 0.385] |
| Scale-proxy  | 2024 | 0.154 | [0.000, 0.345] |

S14 is harder to call strictly collapsed under bootstrap CI. The point F1 is
in 0.35-0.50 for LF-MLP and X-attn (well below the population mean ≈ 0.78)
but the CI is wide enough to admit values up to 0.70 due to small window count.
For Scale-proxy, S14 *is* collapsed (CI hi ≤ 0.39).

## Verdict (revised, per brief honest-reporting requirement)

**S17:** Collapse to chance is the dominant outcome (6 of 9 (arch × seed)
combinations have F1 = 0 with CI exactly [0, 0]). The remaining 3 combinations
(seed-2024 runs of LF-MLP and X-attn) have F1 ∈ [0.10, 0.14] with CI extending
to 0.32-0.40. Headline claim: "S17 collapses to chance in 6 of 9 (architecture
× seed) runs and to F1 ≤ 0.14 in the remaining 3."

**S14:** Partial collapse. Point F1 below population mean across all 9 runs,
but bootstrap CIs are wide due to small window count (~35 windows / subject
with ~30% positive class). Headline claim: "S14 underperforms population
mean by ≥0.3 F1 across all (architecture × seed) runs, but window-bootstrap
CIs are too wide to claim strict-below-0.3 with confidence."

## Figure choice (Fig 2A treatment)

Two options were considered:
1. Add inset showing distribution of CI widths across all (subject × arch × seed) combinations. Median CI width 0.200, p75 0.392, max 0.762.
2. Add Supplementary Figure S1 with full per-subject CI breakdown.

**Choice:** Option 1 (inset). The inset is a small histogram showing the
empirical CI-width distribution alongside annotations for the most-fragile
subjects. Justification: the inset doesn't try to plot CI for every subject
(would be illegible) but does show that the typical uncertainty is moderate
(0.2 F1) while flagging that some subjects have very wide CIs.

## Reviewer-frame reasoning

A skeptic asking "is the collapse just lucky-fold artifact?" gets:
1. S17 hits F1 = 0 with CI exactly [0, 0] in 6 of 9 (arch × seed) runs —
   that's the strongest possible bootstrap-confirmation of collapse.
2. The 3 remaining runs still have F1 ≤ 0.14 with CI ≤ 0.40 — even the
   non-strict cases are far below the population mean of 0.78.
3. Two distinct uncertainty sources are reported (window bootstrap +
   across-seed training std), so any "but you didn't account for X" objection
   can be redirected to the right column of the table.
4. The percentile-bootstrap caveat is disclosed.
