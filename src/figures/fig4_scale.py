"""Figure 4 — Scale doesn't fix it.

Three sub-views (vertical strip), all showing the same per-architecture comparison:
  Top:    per-subject F1 strip plot
  Middle: F1 under worst-case modality dropout
  Bottom: F1 under high noise

Annotate parameter count beneath each architecture label.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.figures.style import apply_style, PALETTE, MODEL_LABELS, MODEL_PARAMS, panel_label
from src.figures._data_loader import (
    load_exp1, load_exp2, load_exp3,
    SUBJECTS, MODELS, NOISE_SIGMAS,
)


def panel_top(ax, per_subj):
    rng = np.random.default_rng(7)
    x_centers = np.arange(len(MODELS))
    for i, m in enumerate(MODELS):
        means = [per_subj[m][s]["f1_mean"] for s in SUBJECTS]
        xs = x_centers[i] + (rng.random(len(means)) - 0.5) * 0.30
        ax.scatter(xs, means, s=22, c=PALETTE[m], alpha=0.85,
                   edgecolor="white", linewidth=0.5)
        agg_mu = np.mean(means)
        ax.hlines(agg_mu, x_centers[i] - 0.30, x_centers[i] + 0.30,
                  color=PALETTE[m], lw=1.6)
    ax.axhline(0.3, color=PALETTE["stress"], linestyle=":", lw=0.7, alpha=0.4)
    ax.set_xticks(x_centers)
    labels = [f"{MODEL_LABELS[m]}\n({MODEL_PARAMS[m]} params)" for m in MODELS]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Per-subject F1")
    ax.set_ylim(-0.02, 1.05)


def panel_mid(ax, exp2):
    x_centers = np.arange(len(MODELS))
    means, stds = [], []
    for m in MODELS:
        worst = min(c for c in exp2[m] if c != "all_clean")  # name-only fallback if needed
        # Pick the cond with the lowest mean F1 per arch
        worst_cond = min((c for c in exp2[m] if c != "all_clean"),
                         key=lambda c: exp2[m][c]["f1_mean"])
        means.append(exp2[m][worst_cond]["f1_mean"])
        stds.append(exp2[m][worst_cond]["f1_std"])
    bars = ax.bar(x_centers, means, yerr=stds, capsize=2,
                  color=[PALETTE[m] for m in MODELS],
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=8)
    ax.set_ylabel("F1 (worst-case dropout)")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.3, color=PALETTE["stress"], linestyle=":", lw=0.7, alpha=0.4)


def panel_bot(ax, exp3, per_subj):
    """Panel C: relative F1 drop from clean to high-noise.
    Negative values = robust; positive = degraded.
    """
    x_centers = np.arange(len(MODELS))
    high_sigma = max(NOISE_SIGMAS)
    drops, stds = [], []
    for m in MODELS:
        clean_subj = [per_subj[m][s]["f1_mean"] for s in SUBJECTS
                      if per_subj[m][s]["f1_mean"] is not None]
        clean = float(np.mean(clean_subj))
        d = exp3[m]["gaussian"][str(high_sigma)]
        drops.append(clean - d["f1_mean"])
        stds.append(d["f1_std"])
    ax.bar(x_centers, drops, yerr=stds, capsize=2,
           color=[PALETTE[m] for m in MODELS],
           edgecolor="white", linewidth=0.5)
    ax.axhline(0, color=PALETTE["baseline"], lw=0.6)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=8)
    ax.set_ylabel(f"F1 drop, clean to σ={high_sigma:g}")
    ax.set_ylim(-0.05, 0.5)


def render(out_path: Path) -> None:
    apply_style()
    per_subj, _, s1 = load_exp1()
    exp2, s2 = load_exp2(per_subj)
    exp3, s3 = load_exp3(per_subj)

    fig = plt.figure(figsize=(5.0, 7.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1],
                          left=0.18, right=0.96, top=0.94, bottom=0.06,
                          hspace=0.55)

    axT = fig.add_subplot(gs[0])
    panel_top(axT, per_subj); panel_label(axT, "A", x=-0.16, y=1.04)
    axT.text(0.02, 0.96, "Per-subject collapse persists",
             transform=axT.transAxes, fontsize=8.5, fontweight="bold",
             color=PALETTE["annotation"], va="top")

    axM = fig.add_subplot(gs[1])
    panel_mid(axM, exp2); panel_label(axM, "B", x=-0.16, y=1.04)
    axM.text(0.02, 0.96, "Modality shortcut persists",
             transform=axM.transAxes, fontsize=8.5, fontweight="bold",
             color=PALETTE["annotation"], va="top")

    axB = fig.add_subplot(gs[2])
    panel_bot(axB, exp3, per_subj); panel_label(axB, "C", x=-0.16, y=1.04)
    axB.text(0.02, 0.96, "Noise brittleness: late-fusion + cross-attn collapse;\n"
             "scale-proxy's 'robustness' is frozen-encoder saturation",
             transform=axB.transAxes, fontsize=8.0, fontweight="bold",
             color=PALETTE["annotation"], va="top")

    flag = ""
    if any(s == "placeholder" for s in (s1, s2, s3)):
        flag = "  [PLACEHOLDER DATA]"
    fig.suptitle("Scale alone does not fix any of the three failure modes" + flag,
                 fontsize=11, fontweight="bold", y=0.99)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"saved {out_path}.pdf and .png")


if __name__ == "__main__":
    render(Path("figures/fig4_scale"))
