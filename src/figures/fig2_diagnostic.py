"""Figure 2 — Diagnostic results (the empirical anchor).

3 panels:
  A: Per-subject F1 strip plot (3 archs). With seed-std bars and majority baseline.
  B: Modality dropout heatmap or grouped bars (D10 — start with grouped bars).
  C: F1 vs Gaussian noise sigma, line + shaded band.

Layout (D07): A is 2x width on left, B and C stacked on right.
"""
from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np


def _inset_ci_widths(parent_ax):
    """Small inset showing distribution of bootstrap CI widths from Exp6.

    Falls back silently if the data isn't there.
    """
    p = Path("results/exp6_bootstrap_ci/per_subject_ci.json")
    if not p.exists():
        return
    d = json.load(open(p))
    widths = []
    for m in d:
        for s in d[m]:
            for sd, info in d[m][s].get("per_seed", {}).items():
                widths.append(info["boot_ci_hi"] - info["boot_ci_lo"])
    if not widths:
        return
    inset = parent_ax.inset_axes([0.62, 0.04, 0.36, 0.20])
    inset.hist(widths, bins=20, color="#888888", edgecolor="white", linewidth=0.4)
    inset.set_xlabel("window-bootstrap 95% CI width", fontsize=6, labelpad=1)
    inset.set_ylabel("# (subj×arch×seed)", fontsize=6, labelpad=1)
    inset.tick_params(labelsize=6)
    inset.set_xlim(0, 1.0)
    median = float(np.median(widths))
    inset.axvline(median, color="#333333", linestyle="--", lw=0.6)
    inset.text(median + 0.02, inset.get_ylim()[1] * 0.85,
               f"median {median:.2f}", fontsize=6, color="#333333")
    for s in ("top", "right"):
        inset.spines[s].set_visible(False)

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
        pairs = [(s, per_subj[m][s]["f1_mean"], per_subj[m][s]["f1_std"])
                 for s in SUBJECTS
                 if per_subj[m][s]["f1_mean"] is not None]
        if not pairs:
            ax.text(x_centers[i], 0.5, "(no data)", ha="center", va="center",
                    color=PALETTE[m], alpha=0.6, fontsize=9)
            continue
        means = np.array([p[1] for p in pairs])
        stds = np.array([p[2] if p[2] is not None else 0.0 for p in pairs])
        xs = x_centers[i] + (rng.random(len(means)) - 0.5) * 2 * jitter_w
        for x, mu, sd in zip(xs, means, stds):
            ax.vlines(x, max(0, mu - sd), min(1, mu + sd),
                      color=PALETTE[m], alpha=0.45, lw=1.0)
        ax.scatter(xs, means, s=34, c=PALETTE[m], alpha=0.92,
                   edgecolor="white", linewidth=0.6, zorder=3)
        agg_mu = float(np.mean(means))
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

    _inset_ci_widths(ax)

    # Annotate worst subject for first model with real data
    for m_idx, m in enumerate(MODELS):
        valid = [(s, per_subj[m][s]["f1_mean"]) for s in SUBJECTS
                 if per_subj[m][s]["f1_mean"] is not None]
        if not valid:
            continue
        worst_subj, worst_f1 = min(valid, key=lambda p: p[1])
        ax.annotate(f"{worst_subj} F1={worst_f1:.2f}",
                    xy=(m_idx, worst_f1), xytext=(m_idx + 0.55, max(0.20, worst_f1 + 0.15)),
                    fontsize=7.5, color=PALETTE["annotation"],
                    arrowprops=dict(arrowstyle="-", color=PALETTE["annotation"],
                                    lw=0.5, connectionstyle="arc3,rad=-0.2"))
        break


def panel_B(ax, exp2, status):
    width = 0.25
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
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=7,
              bbox_to_anchor=(0.0, 1.13))

    # Find biggest drop and annotate near where the bars actually live
    cmean_clean = np.mean([exp2[m]["all_clean"]["f1_mean"] for m in MODELS])
    drops = {}
    for c in DROPOUT_CONDS:
        if c == "all_clean":
            continue
        drops[c] = cmean_clean - np.mean([exp2[m][c]["f1_mean"] for m in MODELS])
    worst_c = max(drops, key=drops.get)
    worst_drop = drops[worst_c]
    worst_idx = DROPOUT_CONDS.index(worst_c)
    cond_label = worst_c.replace("drop_", "").replace("_", "+")
    bar_y_low = min(exp2[m][worst_c]["f1_mean"] for m in MODELS)
    ax.annotate(f"−{worst_drop*100:.0f} pp F1\n(drop {cond_label})",
                xy=(worst_idx, bar_y_low + 0.04),
                xytext=(max(0.5, worst_idx - 2.5), 0.78),
                fontsize=7.5, color=PALETTE["stress"], fontweight="bold",
                ha="left",
                arrowprops=dict(arrowstyle="->", color=PALETTE["stress"], lw=0.7,
                                connectionstyle="arc3,rad=-0.15"))


def panel_C(ax, exp3, status):
    sigmas = NOISE_SIGMAS
    for m in MODELS:
        means = np.array([exp3[m]["gaussian"][str(s)]["f1_mean"] for s in sigmas])
        stds = np.array([exp3[m]["gaussian"][str(s)]["f1_std"] for s in sigmas])
        ax.plot(sigmas, means, "-o", color=PALETTE[m], lw=1.5, ms=5,
                mec="white", mew=0.6, label=MODEL_LABELS[m])
        ax.fill_between(sigmas, np.clip(means - stds, 0, 1),
                        np.clip(means + stds, 0, 1),
                        color=PALETTE[m], alpha=0.15)
    ax.axhline(0.3, color=PALETTE["stress"], linestyle=":", lw=0.8, alpha=0.5)
    ax.text(sigmas[-1], 0.31, "collapse 0.3",
            fontsize=7, color=PALETTE["stress"], ha="right", va="bottom")
    ax.set_xlabel("Gaussian noise σ (× input std)")
    ax.set_ylabel("F1 (stress)")
    ax.set_xscale("log")
    ax.set_xticks(sigmas)
    ax.set_xticklabels([f"{s:g}" for s in sigmas])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower left", frameon=False, fontsize=7)


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
