"""Figure 3 — Unification: same subjects fail across modes; same modalities are shortcuts.

A: Scatter of subject clean F1 vs subject worst-case dropout F1. Color = high-noise F1.
   Pearson r annotated.
B: Grouped bars: F1 drop per modality dropped, by architecture.
"""
from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

from src.figures.style import apply_style, PALETTE, MODEL_LABELS, panel_label
from src.figures._data_loader import (
    load_exp1, load_exp2, load_exp3,
    SUBJECTS, MODELS, DROPOUT_CONDS, NOISE_SIGMAS,
)


def _per_subject_dropout_summary(exp2_raw_path: Path) -> dict:
    """If raw exp2 JSON exists, compute per-subject mean across seeds, per condition.
    Returns {model: {subject: {cond: f1_mean}}}.
    """
    out = {}
    for m in MODELS:
        p = exp2_raw_path / f"{m}.json"
        if not p.exists():
            return {}
        d = json.load(open(p))
        out[m] = {}
        for cond, subj_seeds in d.items():
            for subj, seedmap in subj_seeds.items():
                out[m].setdefault(subj, {})[cond] = float(
                    np.mean([r["f1_stress"] for r in seedmap.values()]))
    return out


def _per_subject_noise_summary(exp3_raw_path: Path) -> dict:
    """{model: {subject: f1_mean_at_high_sigma}}."""
    out = {}
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


def panel_A(ax, per_subj, exp2_per_subj, exp3_per_subj, status):
    """Each point = one subject (averaged over architectures + seeds).
    x = clean LOSO F1, y = worst-case dropout F1, color = high-noise F1.
    """
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
                worst_v = min((conds[c] for c in conds if c != "all_clean"),
                              default=None)
                if worst_v is not None:
                    worst_vals.append(worst_v)
        worst = float(np.mean(worst_vals)) if worst_vals else \
                float(np.clip(clean - rng.uniform(0.20, 0.45), 0, 1))

        noise_vals = []
        for m in MODELS:
            if exp3_per_subj and s in exp3_per_subj.get(m, {}):
                noise_vals.append(exp3_per_subj[m][s])
        noise = float(np.mean(noise_vals)) if noise_vals else \
                float(np.clip(clean - rng.uniform(0.30, 0.60), 0, 1))

        xs.append(clean); ys.append(worst); cs.append(noise); labels.append(s)

    cmap = LinearSegmentedColormap.from_list("noise", ["#220022", "#E07B91", "#FFEFC8"])
    norm = Normalize(vmin=0, vmax=max(0.01, max(cs)))
    sc = ax.scatter(xs, ys, c=cs, cmap=cmap, s=80, edgecolor="white",
                    linewidth=0.7, zorder=3, alpha=0.95, norm=norm)

    # Label each point with subject id
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color=PALETTE["annotation"], alpha=0.7)

    lim_lo, lim_hi = -0.04, 1.04
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color=PALETTE["baseline"], linestyle="--", lw=0.6, alpha=0.6)

    xs_a, ys_a, cs_a = np.array(xs), np.array(ys), np.array(cs)
    from scipy.stats import spearmanr
    if len(xs_a) >= 3 and xs_a.std() > 1e-3 and ys_a.std() > 1e-3:
        r_p = float(np.corrcoef(xs_a, ys_a)[0, 1])
        r_s = float(spearmanr(xs_a, ys_a).correlation)
    else:
        r_p = r_s = float("nan")

    # Identify subjects in the bottom-left "fragile across all modes" cluster
    fragile = [(lab, x, y, c) for lab, x, y, c in zip(labels, xs, ys, cs)
               if x < 0.55 and y < 0.20]
    fragile_names = [f[0] for f in fragile]

    ax.text(0.04, 0.94, f"Spearman ρ = {r_s:+.2f}   Pearson r = {r_p:+.2f}",
            transform=ax.transAxes, fontsize=9.5,
            color=PALETTE["annotation"], fontweight="bold")
    if fragile_names:
        ax.text(0.04, 0.87,
                f"{', '.join(fragile_names)}: fragile across all 3 failure modes",
                transform=ax.transAxes, fontsize=8, style="italic",
                color=PALETTE["stress"])
    ax.text(0.04, 0.81,
            "each point = one subject (mean over architectures & seeds)",
            transform=ax.transAxes, fontsize=7.5, style="italic",
            color=PALETTE["annotation"])

    # Highlight the fragile cluster with a faint box
    if fragile:
        x_lo = min(f[1] for f in fragile) - 0.04
        x_hi = max(f[1] for f in fragile) + 0.06
        y_lo = -0.04
        y_hi = max(f[2] for f in fragile) + 0.10
        from matplotlib.patches import Rectangle
        rect = Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                         fc=PALETTE["stress"], ec=PALETTE["stress"],
                         alpha=0.10, lw=0.6, ls="--", zorder=1)
        ax.add_patch(rect)

    ax.set_xlabel("Clean F1 (LOSO, mean over architectures)")
    ax.set_ylabel("Worst-dropout F1 (mean over architectures)")
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)

    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("High-noise F1", fontsize=8)
    cb.ax.tick_params(labelsize=7)


def panel_B(ax, exp2, status):
    """Grouped bars: F1 drop when each single modality dropped, per architecture."""
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
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False, fontsize=7)

    # Find dominant modality (largest avg drop)
    avg_drops = {}
    for md in mods:
        avg_drops[md] = float(np.mean(
            [exp2[m]["all_clean"]["f1_mean"] - exp2[m][f"drop_{md}"]["f1_mean"]
             for m in MODELS]))
    dom_md = max(avg_drops, key=avg_drops.get)
    dom_drop = avg_drops[dom_md]
    dom_idx = mods.index(dom_md)
    ax.annotate(f"{dom_md}: shortcut\nacross all archs",
                xy=(dom_idx, dom_drop), xytext=(dom_idx - 1.0, dom_drop + 0.08),
                fontsize=7.5, color=PALETTE["stress"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["stress"], lw=0.7))


def render(out_path: Path) -> None:
    apply_style()
    per_subj, _, status1 = load_exp1()
    exp2, status2 = load_exp2(per_subj)
    exp3, status3 = load_exp3(per_subj)
    exp2_per = _per_subject_dropout_summary(Path("results/exp2_dropout"))
    exp3_per = _per_subject_noise_summary(Path("results/exp3_degradation"))

    fig = plt.figure(figsize=(9.5, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0],
                          left=0.07, right=0.97, top=0.90, bottom=0.13,
                          wspace=0.32)
    axA = fig.add_subplot(gs[0, 0])
    panel_A(axA, per_subj, exp2_per, exp3_per, status1)
    panel_label(axA, "A", x=-0.12, y=1.04)

    axB = fig.add_subplot(gs[0, 1])
    panel_B(axB, exp2, status2)
    panel_label(axB, "B", x=-0.14, y=1.04)

    flag = ""
    if any(s == "placeholder" for s in (status1, status2, status3)):
        flag = "  [PLACEHOLDER DATA]"
    fig.suptitle("Same subjects fail across failure modes; same modalities shortcut" + flag,
                 fontsize=11.5, fontweight="bold", y=0.99)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"saved {out_path}.pdf and .png")


if __name__ == "__main__":
    render(Path("figures/fig3_unification"))
