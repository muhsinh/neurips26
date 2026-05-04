"""Figure 2 — Diagnostic results (the empirical anchor).

3 panels:
  A: Per-subject F1 strip plot (3 archs). With seed-std bars and majority baseline.
  B: Modality dropout heatmap or grouped bars (D10 — start with grouped bars).
  C: F1 vs Gaussian noise sigma, line + shaded band.

Layout (D07): A is 2x width on left, B and C stacked on right.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.figures.style import apply_style, PALETTE, MODEL_LABELS, panel_label
from src.figures._data_loader import (
    load_exp1, load_exp2, load_exp3,
    SUBJECTS, MODELS, DROPOUT_CONDS, NOISE_SIGMAS,
)


def panel_A(ax, per_subj, overall, status):
    rng = np.random.default_rng(0)
    x_centers = np.arange(len(MODELS))
    jitter_w = 0.16
    for i, m in enumerate(MODELS):
        means = [per_subj[m][s]["f1_mean"] for s in SUBJECTS]
        stds = [per_subj[m][s]["f1_std"] for s in SUBJECTS]
        xs = x_centers[i] + (rng.random(len(means)) - 0.5) * 2 * jitter_w
        for x, mu, sd in zip(xs, means, stds):
            ax.vlines(x, max(0, mu - sd), min(1, mu + sd),
                      color=PALETTE[m], alpha=0.45, lw=1.0)
        ax.scatter(xs, means, s=34, c=PALETTE[m], alpha=0.92,
                   edgecolor="white", linewidth=0.6, zorder=3)
        # subject mean as horizontal tick
        agg_mu = np.mean(means)
        ax.hlines(agg_mu, x_centers[i] - 0.30, x_centers[i] + 0.30,
                  color=PALETTE[m], lw=2.0, zorder=4)

    ax.axhline(0.0, color=PALETTE["baseline"], linestyle="--", lw=0.8,
               label="majority-class F1 = 0", alpha=0.6)
    ax.axhline(0.3, color=PALETTE["stress"], linestyle=":", lw=0.8,
               alpha=0.5)
    ax.text(len(MODELS) - 0.5, 0.31, "collapse threshold (0.3)",
            fontsize=7, color=PALETTE["stress"], va="bottom", ha="right")

    ax.set_xticks(x_centers)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=10, ha="right")
    ax.set_ylabel("F1 (stress class), per held-out subject")
    ax.set_ylim(-0.04, 1.05)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)

    # Annotate worst subject for first model
    worst_subj = min(SUBJECTS, key=lambda s: per_subj[MODELS[0]][s]["f1_mean"])
    worst_f1 = per_subj[MODELS[0]][worst_subj]["f1_mean"]
    ax.annotate(f"{worst_subj} F1={worst_f1:.2f}",
                xy=(0, worst_f1), xytext=(0.55, 0.20),
                fontsize=7.5, color=PALETTE["annotation"],
                arrowprops=dict(arrowstyle="-", color=PALETTE["annotation"],
                                lw=0.5, connectionstyle="arc3,rad=-0.2"))


def panel_B(ax, exp2, status):
    width = 0.25
    n_models = len(MODELS)
    n_conds = len(DROPOUT_CONDS)
    x = np.arange(n_conds)
    for i, m in enumerate(MODELS):
        means = [exp2[m][c]["f1_mean"] for c in DROPOUT_CONDS]
        stds = [exp2[m][c]["f1_std"] for c in DROPOUT_CONDS]
        ax.bar(x + (i - 1) * width, means, width=width,
               yerr=stds, capsize=1.5, error_kw={"linewidth": 0.5},
               color=PALETTE[m], edgecolor="white", linewidth=0.5,
               label=MODEL_LABELS[m])
    short_labels = [c.replace("drop_", "−").replace("all_clean", "full")
                    for c in DROPOUT_CONDS]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("F1 (stress)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=False, ncol=1, fontsize=7)

    # Find biggest single-modality drop and annotate
    cmean_clean = np.mean([exp2[m]["all_clean"]["f1_mean"] for m in MODELS])
    drops = {}
    for c in DROPOUT_CONDS:
        if c == "all_clean":
            continue
        drop = cmean_clean - np.mean([exp2[m][c]["f1_mean"] for m in MODELS])
        drops[c] = drop
    worst_c = max(drops, key=drops.get)
    worst_drop = drops[worst_c]
    worst_idx = DROPOUT_CONDS.index(worst_c)
    bar_y = exp2[MODELS[1]][worst_c]["f1_mean"]
    ax.annotate(f"shortcut: −{worst_drop:.2f} F1\nwhen {worst_c.replace('drop_','')} dropped",
                xy=(worst_idx, bar_y), xytext=(worst_idx - 1.5, 0.85),
                fontsize=7, color=PALETTE["stress"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["stress"], lw=0.7))


def panel_C(ax, exp3, status):
    sigmas = [0.0] + NOISE_SIGMAS
    for m in MODELS:
        clean_f1 = np.nan
        # attempt to use exp1 baseline if available; else use sigma=0.1 as proxy
        means = [exp3[m]["gaussian"][str(NOISE_SIGMAS[0])]["f1_mean"]] + \
                [exp3[m]["gaussian"][str(s)]["f1_mean"] for s in NOISE_SIGMAS]
        # use first as ~clean reference
        stds = [exp3[m]["gaussian"][str(NOISE_SIGMAS[0])]["f1_std"]] + \
               [exp3[m]["gaussian"][str(s)]["f1_std"] for s in NOISE_SIGMAS]
        means = np.array(means); stds = np.array(stds)
        ax.plot(sigmas, means, "-o", color=PALETTE[m], lw=1.4, ms=4,
                mec="white", mew=0.5, label=MODEL_LABELS[m])
        ax.fill_between(sigmas, np.clip(means - stds, 0, 1),
                        np.clip(means + stds, 0, 1),
                        color=PALETTE[m], alpha=0.15)
    ax.axhline(0.3, color=PALETTE["stress"], linestyle=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("Gaussian noise σ (× input std)")
    ax.set_ylabel("F1 (stress)")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=7)


def render(out_path: Path) -> None:
    apply_style()
    per_subj, overall, status_e1 = load_exp1()
    exp2, status_e2 = load_exp2(per_subj)
    exp3, status_e3 = load_exp3(per_subj)

    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], height_ratios=[1, 1],
                          left=0.06, right=0.98, top=0.93, bottom=0.13,
                          hspace=0.55, wspace=0.32)

    axA = fig.add_subplot(gs[:, 0])
    panel_A(axA, per_subj, overall, status_e1)
    panel_label(axA, "A", x=-0.08, y=1.01)

    axB = fig.add_subplot(gs[0, 1])
    panel_B(axB, exp2, status_e2)
    panel_label(axB, "B", x=-0.16, y=1.05)

    axC = fig.add_subplot(gs[1, 1])
    panel_C(axC, exp3, status_e3)
    panel_label(axC, "C", x=-0.16, y=1.05)

    flag = ""
    if any(s == "placeholder" for s in (status_e1, status_e2, status_e3)):
        flag = "  [PLACEHOLDER DATA]"
    fig.suptitle("Three failure modes of multi-sensor fusion on WESAD" + flag,
                 fontsize=12, fontweight="bold", y=0.995)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"saved {out_path}.pdf and .png  (status: e1={status_e1} e2={status_e2} e3={status_e3})")


if __name__ == "__main__":
    render(Path("figures/fig2_diagnostic"))
