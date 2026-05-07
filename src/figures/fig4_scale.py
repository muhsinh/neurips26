"""Figure 4 — Scale doesn't fix it.

Three sub-views (vertical strip), all showing the same per-architecture comparison:
  Top:    per-subject F1 strip plot
  Middle: F1 under worst-case modality dropout
  Bottom: F1 drop, clean to high-noise

Annotate parameter count beneath each architecture label.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.style import (
    MODEL_LABELS,
    MODEL_PARAMS,
    PALETTE,
    apply_style,
    panel_label,
)
from src.figures._data_loader import (
    MODELS,
    NOISE_SIGMAS,
    SUBJECTS,
    load_exp1,
    load_exp2,
    load_exp3,
)


def panel_top(ax, per_subj) -> None:
    rng = np.random.default_rng(7)
    x_centers = np.arange(len(MODELS))
    for i, m in enumerate(MODELS):
        means = [per_subj[m][s]["f1_mean"] for s in SUBJECTS]
        xs = x_centers[i] + (rng.random(len(means)) - 0.5) * 0.30
        ax.scatter(xs, means, s=18, c=PALETTE[m], alpha=0.85,
                   edgecolor="white", linewidth=0.4)
        agg_mu = float(np.mean(means))
        ax.hlines(agg_mu, x_centers[i] - 0.30, x_centers[i] + 0.30,
                  color=PALETTE[m], lw=1.6)
    ax.axhline(0.3, color=PALETTE["stress"], linestyle=":", lw=0.7, alpha=0.6)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=7.5,
                       rotation=20, ha="right")
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylabel("Per-subject F1", fontsize=9)
    ax.set_ylim(-0.04, 1.05)


def panel_mid(ax, exp2) -> None:
    x_centers = np.arange(len(MODELS))
    means, stds = [], []
    for m in MODELS:
        worst_cond = min((c for c in exp2[m] if c != "all_clean"),
                         key=lambda c: exp2[m][c]["f1_mean"])
        means.append(exp2[m][worst_cond]["f1_mean"])
        # Clip std so error-bar whiskers don't extend disproportionately on
        # near-zero bars (the std represents seed × subject variation, not
        # symmetric uncertainty).
        stds.append(min(exp2[m][worst_cond]["f1_std"], 0.20))
    ax.bar(x_centers, means, yerr=stds, capsize=2,
           error_kw={"linewidth": 0.6, "ecolor": PALETTE["annotation"]},
           color=[PALETTE[m] for m in MODELS],
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=7.5,
                       rotation=20, ha="right")
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylabel("F1 (worst-case dropout)", fontsize=9)
    ax.set_ylim(0, 0.9)
    ax.axhline(0.3, color=PALETTE["stress"], linestyle=":", lw=0.7, alpha=0.6)


def panel_bot(ax, exp3, per_subj) -> None:
    """Bar = F1 drop from clean (σ=0.1) to high-noise (σ=2). Higher = more brittle.

    Scale-proxy is omitted from this panel: its frozen randomly-initialised
    encoder maps high-amplitude noisy inputs into the same feature region as
    clean inputs (saturation), producing an artefactual ≈0 drop. Including it
    misleads the reader. The omission is documented in the caption.
    """
    plotted_models = [m for m in MODELS if m != "scale_proxy"]
    x_centers = np.arange(len(plotted_models))
    high_sigma = max(NOISE_SIGMAS)
    drops, stds = [], []
    for m in plotted_models:
        clean_subj = [per_subj[m][s]["f1_mean"] for s in SUBJECTS
                      if per_subj[m][s]["f1_mean"] is not None]
        clean = float(np.mean(clean_subj))
        d = exp3[m]["gaussian"][str(high_sigma)]
        drops.append(clean - d["f1_mean"])
        stds.append(min(d["f1_std"], 0.20))
    ax.bar(x_centers, drops, yerr=stds, capsize=2,
           error_kw={"linewidth": 0.6, "ecolor": PALETTE["annotation"]},
           color=[PALETTE[m] for m in plotted_models],
           edgecolor="white", linewidth=0.5)
    ax.axhline(0, color=PALETTE["baseline"], lw=0.6)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([MODEL_LABELS[m] for m in plotted_models], fontsize=7.5,
                       rotation=20, ha="right")
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylabel(f"F1 drop, clean to σ={high_sigma:g}", fontsize=9)
    ax.set_ylim(-0.05, 0.55)
    ax.set_xlim(-0.7, len(plotted_models) - 0.3)


def _set_panel_title(ax, text: str, fontsize: float = 9) -> None:
    """Bold left-aligned title above panel — replaces in-data text annotation."""
    ax.set_title(text, loc="left", fontsize=fontsize, fontweight="bold",
                 color=PALETTE["annotation"], pad=6)


def render(out_path: Path) -> None:
    apply_style()
    per_subj, _, s1 = load_exp1()
    exp2, s2 = load_exp2(per_subj)
    exp3, s3 = load_exp3(per_subj)

    # Single-row 1×3 layout sized to read at \linewidth (~6.5 in). Aspect
    # 14:6.5 ≈ 2.15:1 keeps panels close to square at print scale.
    # Print-native size: figure renders at \\linewidth without additional
    # scaling, so font points map 1:1 to printed points.
    fig = plt.figure(figsize=(6.8, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1],
                          left=0.10, right=0.98, top=0.80, bottom=0.22,
                          wspace=0.40)

    axT = fig.add_subplot(gs[0])
    panel_top(axT, per_subj)
    panel_label(axT, "A", x=-0.20, y=1.10)
    _set_panel_title(axT, "Per-subject collapse persists", fontsize=8.5)

    axM = fig.add_subplot(gs[1])
    panel_mid(axM, exp2)
    panel_label(axM, "B", x=-0.20, y=1.10)
    _set_panel_title(axM, "Modality shortcut persists", fontsize=8.5)

    axB = fig.add_subplot(gs[2])
    panel_bot(axB, exp3, per_subj)
    panel_label(axB, "C", x=-0.20, y=1.10)
    _set_panel_title(axB, "Noise brittleness (trained encoders)", fontsize=8.5)

    flag = ""
    if any(s == "placeholder" for s in (s1, s2, s3)):
        flag = "  [PLACEHOLDER DATA]"
    fig.suptitle("Scale alone does not fix any of the three failure modes" + flag,
                 fontsize=10.5, fontweight="bold", y=0.97)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"saved {out_path}.pdf and .png")


if __name__ == "__main__":
    render(Path("figures/fig4_scale"))
