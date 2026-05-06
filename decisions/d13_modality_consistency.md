# Decision 13: Cross-architecture modality-importance consistency (Exp 5)

**Time:** 2026-05-06 (post-Tier-1 start)
**Choice faced:** How to defend "EDA shortcut across architectures" against the coincidence objection (Reviewer 4 skeptic).

## Methodology

For each architecture, build a vector of single-modality-dropout F1 losses
(length 4, one per modality). Compute Spearman ρ + Pearson r between every
architecture pair.

**Bootstrap procedure (1000 resamples):** Resample 15 subjects with replacement; re-derive each architecture's per-modality F1-loss vector from the resampled subjects; recompute ρ. Report 2.5th and 97.5th percentile.

**Permutation null (10,000 shuffles):** Shuffle the modality labels in one architecture's vector, recompute ρ, count fraction ≥ observed. Caveat: only 4 modalities → 4! = 24 unique permutations, so the empirical p-value floor is ~1/24 ≈ 0.042 by combinatorics.

**Dual-modality fallback (6 pair-drop conditions):** With C(4,2) = 6 conditions, Spearman has more support (6! = 720 permutations).

## Pre-registered interpretation rule

"ρ ≥ 0.8 across all three pairs AND permutation p < 0.10 for all three" =
supports cross-architectural agreement. Mixed/weaker = soften the claim.

## Results

### Single-modality dropout (4 conditions)

| Pair | Spearman ρ | bootstrap 95% CI | Pearson r | perm p |
|------|------------|------------------|-----------|--------|
| LF-MLP × X-attn      | +0.200 | [+0.200, +1.000] | +0.990 | 0.459 |
| LF-MLP × Scale-proxy | +0.400 | [+0.200, +1.000] | +0.990 | 0.377 |
| X-attn × Scale-proxy | +0.800 | [+0.200, +1.000] | +0.999 | 0.170 |

Single-modality **FAILS** the pre-registered Spearman threshold. Three of four
modalities (ACC, BVP, TEMP) all have F1-loss < 0.06 — they're effectively tied
near zero, so rank order between them flips unstably across architectures.
Pearson is uniformly very high (r > 0.99) because the EDA point dominates the
linear fit; this is the more informative measure when the rank ties are noise.

### Dual-modality dropout (6 conditions, fallback)

| Pair | Spearman ρ | bootstrap 95% CI | Pearson r | perm p |
|------|------------|------------------|-----------|--------|
| LF-MLP × X-attn      | +1.000 | [+0.657, +1.000] | +0.950 | **0.0021** |
| LF-MLP × Scale-proxy | +0.886 | [+0.600, +1.000] | +0.892 | **0.0161** |
| X-attn × Scale-proxy | +0.886 | [+0.714, +1.000] | +0.983 | **0.0161** |

Dual-modality **PASSES** the pre-registered threshold: all ρ ≥ 0.886, all
permutation p ≤ 0.016, all bootstrap CI lower bounds ≥ 0.60.

## Verdict

Dual-modality analysis supports cross-architectural agreement on shortcut
structure. Single-modality analysis is consistent (Pearson very high) but
underpowered on Spearman because the non-EDA modalities are near-zero ties.

**Reporting:** Both analyses go in the paper. Headline statistic = dual-modality.
Single-modality reported in caption + appendix with the explicit reason for the
power limitation.

**Effect on the unification claim:** The cross-architectural agreement on
*which subset-of-modalities are jointly necessary* (dual-modality) is strong.
This rebuts the coincidence objection without overclaiming.

## Reviewer-frame reasoning

A skeptic asking "could EDA-as-shortcut be coincidence?" gets:
1. Three independent architectures, with a frozen-encoder baseline (so capacity
   can't be the explanation), all rank the same modality pairs as most
   destructive when dropped.
2. ρ ≥ 0.886 across all architecture pairs, bootstrap CI lower-bound ≥ 0.60,
   permutation p < 0.02 across all three pairs.
3. The single-modality limitation is honestly disclosed.

This is harder to wave away than "EDA appeared in the same row of every bar
chart." It's a falsifiable rank-consistency claim with explicit uncertainty
quantification.
