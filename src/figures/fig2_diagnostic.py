"""Figure 2 — Diagnostic results (the empirical anchor).

3 panels:
  A: Per-subject F1 strip plot (3 archs). With seed-std bars and majority baseline.
  B: Modality dropout grouped bars (D10 — start with grouped bars).
  C: F1 vs Gaussian noise sigma, line + shaded band.

Layout (D07): A is 2x width on left, B and C stacked on right.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.style import apply_style, MODEL_LABELS, PALETTE, panel_label
from src.figures._data_loader import (
    DROPOUT_CONDS,
    MODELS,
    NOISE_SIGMAS,
    SUBJECTS,
    load_exp1,
    load_exp2,
    load_exp3,
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

    ax.axhline(0.0, color=PALETTE["baseline"], linestyle="--", lw=0.8, alpha=0.5)
    ax.text(-0.55, 0.01, "always-predict-non-stress F1 = 0",
            fontsize=7, color=PALETTE["baseline"], va="bottom", ha="left")

    ax.set_xticks(x_centers)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=10, ha="right")
    ax.set_ylabel("F1 (stress class), per held-out subject")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(-0.7, len(MODELS) - 0.4)

    # Annotate the universal collapse subject — single label, three thin lines
    # to all three architectures' S17 points so the cross-arch story is explicit.
    target_subj = None
    targets = []  # (x_data, y_data) per architecture
    for m_idx, m in enumerate(MODELS):
        valid = [(s, per_subj[m][s]["f1_mean"]) for s in SUBJECTS
                 if per_subj[m][s]["f1_mean"] is not None]
        if not valid:
            continue
        s_subj, s_f1 = min(valid, key=lambda p: p[1])
        if target_subj is None:
            target_subj = s_subj
        if s_subj == target_subj:
            targets.append((m_idx, max(s_f1, 0.005)))
    if target_subj is not None and targets:
        # Anchor callout in the left margin, draw a single short line to the
        # leftmost S17 point. Caption explains the cross-arch persistence.
        text_xy = (-0.62, 0.18)
        ax.text(text_xy[0], text_xy[1],
                f"{target_subj} collapses across\n"
                f"all three architectures\n"
                f"(F1 ≈ {targets[0][1]:.2f})",
                fontsize=7.5, color=PALETTE["annotation"],
                ha="left", va="bottom", fontweight="bold")
        first_tx, first_ty = targets[0]
        ax.annotate("", xy=(first_tx - 0.05, first_ty + 0.02),
                    xytext=(text_xy[0] + 0.30, text_xy[1] + 0.02),
                    arrowprops=dict(arrowstyle="-",
                                    color=PALETTE["annotation"],
                                    lw=0.5, alpha=0.7,
                                    connectionstyle="arc3,rad=-0.2"))


def panel_B(ax, exp2, status):
    width = 0.25
    x = np.arange(len(DROPOUT_CONDS))
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
    ax.set_ylim(0, 1.18)
    # Legend outside on the right to free top margin for the annotation.
    ax.legend(loc="upper left", frameon=False, fontsize=7,
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # Annotate worst-drop pair: text in upper-left empty zone, short arrow.
    cmean_clean = np.mean([exp2[m]["all_clean"]["f1_mean"] for m in MODELS])
    drops = {c: cmean_clean - np.mean([exp2[m][c]["f1_mean"] for m in MODELS])
             for c in DROPOUT_CONDS if c != "all_clean"}
    worst_c = max(drops, key=drops.get)
    worst_drop = drops[worst_c]
    worst_idx = DROPOUT_CONDS.index(worst_c)
    cond_label = worst_c.replace("drop_", "").replace("_", "+")
    bar_y_low = min(exp2[m][worst_c]["f1_mean"] for m in MODELS)
    ax.annotate(f"−{worst_drop * 100:.0f} pp F1\n(drop {cond_label})",
                xy=(worst_idx, bar_y_low + 0.04),
                xytext=(worst_idx, 1.16),
                fontsize=7.5, color=PALETTE["stress"], fontweight="bold",
                ha="center", va="top",
                arrowprops=dict(arrowstyle="->", color=PALETTE["stress"], lw=0.7,
                                shrinkA=2, shrinkB=2))


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
    ax.set_xlabel("Gaussian noise σ (× input std)")
    ax.set_ylabel("F1 (stress)")
    ax.set_xscale("log")
    ax.set_xticks(sigmas)
    ax.set_xticklabels([f"{s:g}" for s in sigmas])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=False, fontsize=7,
              bbox_to_anchor=(0.0, -0.05))


def render(out_path: Path) -> None:
    apply_style()
    per_subj, overall, status_e1 = load_exp1()
    exp2, status_e2 = load_exp2(per_subj)
    exp3, status_e3 = load_exp3(per_subj)

    fig = plt.figure(figsize=(11.0, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0], height_ratios=[1, 1],
                          left=0.06, right=0.85, top=0.88, bottom=0.16,
                          hspace=0.65, wspace=0.34)

    axA = fig.add_subplot(gs[:, 0])
    panel_A(axA, per_subj, overall, status_e1)
    panel_label(axA, "A", x=-0.10, y=1.02)

    axB = fig.add_subplot(gs[0, 1])
    panel_B(axB, exp2, status_e2)
    panel_label(axB, "B", x=-0.18, y=1.04)

    axC = fig.add_subplot(gs[1, 1])
    panel_C(axC, exp3, status_e3)
    panel_label(axC, "C", x=-0.18, y=1.04)

    flag = ""
    if any(s == "placeholder" for s in (status_e1, status_e2, status_e3)):
        flag = "  [PLACEHOLDER DATA]"
    fig.suptitle("Three failure modes of multi-sensor fusion on WESAD" + flag,
                 fontsize=12, fontweight="bold", y=0.97)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"saved {out_path}.pdf and .png  (status: e1={status_e1} e2={status_e2} e3={status_e3})")


if __name__ == "__main__":
    render(Path("figures/fig2_diagnostic"))
