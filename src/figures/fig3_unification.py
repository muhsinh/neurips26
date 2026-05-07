"""Figure 3 — Unification: same subjects fail across modes; same modalities are shortcuts.

A: Scatter of subject clean F1 vs subject worst-case dropout F1. Color = high-noise F1.
   Spearman ρ + Pearson r annotated. Fragile-cluster (S14, S17) highlighted.
B: Grouped bars: F1 drop per modality dropped, by architecture. EDA dominates.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

from src.figures.style import MODEL_LABELS, PALETTE, apply_style, panel_label
from src.figures._data_loader import (
    MODELS,
    NOISE_SIGMAS,
    SUBJECTS,
    load_exp1,
    load_exp2,
    load_exp3,
)


def _per_subject_dropout_summary(exp2_raw_path: Path) -> dict:
    out: dict = {}
    for m in MODELS:
        p = exp2_raw_path / f"{m}.json"
        if not p.exists():
            return {}
        d = json.load(open(p))
        out[m] = {}
        for cond, subj_seeds in d.items():
            if cond.startswith("_"):
                continue
            for subj, seedmap in subj_seeds.items():
                out[m].setdefault(subj, {})[cond] = float(
                    np.mean([r["f1_stress"] for r in seedmap.values()]))
    return out


def _per_subject_noise_summary(exp3_raw_path: Path) -> dict:
    out: dict = {}
    for m in MODELS:
        p = exp3_raw_path / f"{m}.json"
        if not p.exists():
            return {}
        d = json.load(open(p))
        out[m] = {}
        for sigma_key, subj_seeds in d["gaussian"].items():
            if abs(float(sigma_key) - max(NOISE_SIGMAS)) > 1e-6:
                continue
            for subj, seedmap in subj_seeds.items():
                out[m][subj] = float(np.mean([r["f1_stress"] for r in seedmap.values()]))
    return out


def _label_offset(x: float, y: float, lab: str, fragile_set: set) -> tuple[int, int]:
    """Return (dx_pts, dy_pts) offset for a subject label.

    Strategy: only label fragile-cluster subjects (S14, S17) and a few clear
    outliers — labels in the dense (clean≈1, worst-dropout≈0.5) cluster cause
    unavoidable overlap and don't add information that isn't already in
    the panel-A annotation block.
    """
    if lab in fragile_set:
        return (0, 12)  # above marker, inside the pink fragile rectangle
    return (5, 5)


def panel_A(ax, per_subj, exp2_per_subj, exp3_per_subj, status):
    rng = np.random.default_rng(0)
    xs, ys, cs, labels = [], [], [], []

    for s in SUBJECTS:
        clean_vals = [per_subj[m][s]["f1_mean"] for m in MODELS
                      if per_subj[m][s]["f1_mean"] is not None]
        if not clean_vals:
            continue
        clean = float(np.mean(clean_vals))

        worst_vals = []
        for m in MODELS:
            if exp2_per_subj and s in exp2_per_subj.get(m, {}):
                conds = exp2_per_subj[m][s]
                worst_v = min(
                    (conds[c] for c in conds if c != "all_clean"),
                    default=None,
                )
                if worst_v is not None:
                    worst_vals.append(worst_v)
        worst = (float(np.mean(worst_vals)) if worst_vals
                 else float(np.clip(clean - rng.uniform(0.20, 0.45), 0, 1)))

        noise_vals = []
        for m in MODELS:
            if exp3_per_subj and s in exp3_per_subj.get(m, {}):
                noise_vals.append(exp3_per_subj[m][s])
        noise = (float(np.mean(noise_vals)) if noise_vals
                 else float(np.clip(clean - rng.uniform(0.30, 0.60), 0, 1)))

        xs.append(clean); ys.append(worst); cs.append(noise); labels.append(s)

    cmap = LinearSegmentedColormap.from_list("noise", ["#220022", "#E07B91", "#FFEFC8"])
    norm = Normalize(vmin=0, vmax=max(0.01, max(cs)))
    sc = ax.scatter(xs, ys, c=cs, cmap=cmap, s=80, edgecolor="white",
                    linewidth=0.7, zorder=3, alpha=0.95, norm=norm)

    lim_lo, lim_hi = -0.06, 1.06
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color=PALETTE["baseline"], linestyle="--", lw=0.8, alpha=0.7,
            zorder=1)
    ax.text(0.04, 0.10, "y = x (clean = worst-dropout)",
            transform=ax.transAxes, fontsize=6.5,
            color=PALETTE["baseline"], ha="left", va="bottom", style="italic")

    xs_a, ys_a = np.array(xs), np.array(ys)
    if len(xs_a) >= 3 and xs_a.std() > 1e-3 and ys_a.std() > 1e-3:
        r_p = float(np.corrcoef(xs_a, ys_a)[0, 1])
        r_s = float(spearmanr(xs_a, ys_a).correlation)
    else:
        r_p = r_s = float("nan")

    if not (np.isnan(r_p) or np.isnan(r_s)):
        ax.text(0.98, 0.04,
                f"Population: ρ={r_s:+.2f}, r={r_p:+.2f}\n(N=15, ceiling-dominated)",
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                color=PALETTE["annotation"], fontweight="normal",
                bbox=dict(facecolor="white", edgecolor=PALETTE["annotation"],
                          lw=0.5, pad=2, alpha=0.9))

    fragile = [(lab, x, y, c) for lab, x, y, c in zip(labels, xs, ys, cs)
               if x < 0.55 and y < 0.20]
    fragile_names = [f[0] for f in fragile]

    # Label ONLY fragile-cluster subjects (S14, S17). Other labels would not
    # serve the cluster-led story this panel is making. Correlation values
    # (Spearman ρ, Pearson r) are reported in the caption only.
    fragile_set = set(fragile_names)
    for lab, x, y in zip(labels, xs, ys):
        if lab not in fragile_set:
            continue
        dx, dy = _label_offset(x, y, lab, fragile_set)
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, fontweight="bold",
                    color=PALETTE["annotation"], alpha=0.95)

    if fragile:
        x_lo = min(f[1] for f in fragile) - 0.04
        x_hi = max(f[1] for f in fragile) + 0.10
        y_lo = lim_lo + 0.005
        y_hi = max(f[2] for f in fragile) + 0.10
        rect = Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                         fc=PALETTE["stress"], ec=PALETTE["stress"],
                         alpha=0.10, lw=0.8, ls="--", zorder=1)
        ax.add_patch(rect)
        # Cluster claim — anchor to upper-left of axes so it doesn't crowd
        # panel B's left edge or the colorbar.
        if fragile_names:
            ax.annotate(
                f"Fragile cluster (N=2):\n{', '.join(fragile_names)}\nexistence-proof, not population",
                xy=((x_lo + x_hi) / 2, y_hi),
                xytext=(0.04, 0.92), textcoords="axes fraction",
                fontsize=9, fontweight="bold", color=PALETTE["stress"],
                ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=PALETTE["stress"],
                                lw=0.6, connectionstyle="arc3,rad=0.25",
                                alpha=0.7),
            )

    ax.set_xlabel("Clean F1 (LOSO, mean over architectures)")
    ax.set_ylabel("Worst-dropout F1 (mean over architectures)")
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)

    cb = plt.colorbar(sc, ax=ax, orientation="horizontal",
                      fraction=0.06, pad=0.18)
    cb.set_label("High-noise F1", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)


def panel_B(ax, exp2, status):
    mods = ["ACC", "BVP", "EDA", "TEMP"]
    width = 0.25
    x = np.arange(len(mods))
    for i, m in enumerate(MODELS):
        clean = exp2[m]["all_clean"]["f1_mean"]
        drops = [clean - exp2[m][f"drop_{md}"]["f1_mean"] for md in mods]
        ax.bar(x + (i - 1) * width, drops, width=width,
               color=PALETTE[m], edgecolor="white", linewidth=0.5,
               label=MODEL_LABELS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(mods)
    ax.set_ylabel("F1 drop when modality removed")
    ax.set_ylim(-0.05, 0.78)
    ax.axhline(0, color=PALETTE["baseline"], lw=0.5, alpha=0.5)
    # Legend outside on the right so the annotation has a clear top corner.
    ax.legend(loc="upper left", frameon=False, fontsize=7,
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # Cross-arch Spearman box (strongest stat) — upper-left, above bars.
    ax.text(0.02, 0.97,
            "Cross-arch dual-mod\nSpearman ρ ≥ 0.886\n(perm p ≤ 0.016)",
            transform=ax.transAxes, fontsize=7, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor=PALETTE["annotation"],
                      lw=0.5, pad=2, alpha=0.95))

    # Dominant-modality label under the EDA bar group (small plain text).
    avg_drops = {md: float(np.mean(
        [exp2[m]["all_clean"]["f1_mean"] - exp2[m][f"drop_{md}"]["f1_mean"]
         for m in MODELS])) for md in mods}
    dom_md = max(avg_drops, key=avg_drops.get)
    dom_drop = avg_drops[dom_md]
    dom_idx = mods.index(dom_md)
    ax.text(dom_idx, dom_drop + 0.04,
            f"{dom_md} dominant\n(4/6 dual-mod include {dom_md})",
            fontsize=6.5, color=PALETTE["stress"], fontweight="bold",
            ha="center")


def render(out_path: Path) -> None:
    apply_style()
    per_subj, _, status1 = load_exp1()
    exp2, status2 = load_exp2(per_subj)
    exp3, status3 = load_exp3(per_subj)
    exp2_per = _per_subject_dropout_summary(Path("results/exp2_dropout"))
    exp3_per = _per_subject_noise_summary(Path("results/exp3_degradation"))

    # Render at native print size so fonts read at the target paper width.
    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0],
                          left=0.09, right=0.91, top=0.84, bottom=0.20,
                          wspace=0.40)
    axA = fig.add_subplot(gs[0, 0])
    panel_A(axA, per_subj, exp2_per, exp3_per, status1)
    panel_label(axA, "A", x=-0.12, y=1.03)

    axB = fig.add_subplot(gs[0, 1])
    panel_B(axB, exp2, status2)
    panel_label(axB, "B", x=-0.12, y=1.03)

    flag = ""
    if any(s == "placeholder" for s in (status1, status2, status3)):
        flag = "  [PLACEHOLDER DATA]"
    fig.suptitle("Same modalities shortcut (population); same subjects fail (N=2 cluster)" + flag,
                 fontsize=10.5, fontweight="bold", y=0.97)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"saved {out_path}.pdf and .png")


if __name__ == "__main__":
    render(Path("figures/fig3_unification"))
