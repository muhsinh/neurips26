"""Figure 1 — Thesis schematic.

Three columns showing three failure modes converging via arrows to "Distribution shift
current fusion architectures cannot bridge." Below: small inset of cross-modal
generative prior schematic.

NO data plots. Pure schematic. Built with matplotlib patches/arrows.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from src.figures.style import apply_style, PALETTE


def _box(ax, x, y, w, h, ec, fc="white", lw=1.0, **kwargs):
    bb = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.02,rounding_size=0.04",
                        ec=ec, fc=fc, lw=lw, **kwargs)
    ax.add_patch(bb)
    return bb


def _arrow(ax, x1, y1, x2, y2, color="#444444", lw=1.2,
           shrinkA=4, shrinkB=4):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=14, shrinkA=shrinkA, shrinkB=shrinkB,
                        zorder=2)
    ax.add_patch(a)


def _failure_panel_subject(ax_inset):
    rng = np.random.default_rng(1)
    means = np.clip(rng.normal(0.78, 0.18, 12), 0.0, 1.0)
    means = np.concatenate([means, [0.05, 0.18, 0.22]])
    rng.shuffle(means)
    x = np.arange(len(means))
    ax_inset.scatter(x, means, s=14, c=PALETTE["cross_attention"],
                     alpha=0.85, edgecolor="white", linewidth=0.4)
    ax_inset.axhline(0.3, color=PALETTE["baseline"], linestyle="--", lw=0.6)
    ax_inset.set_ylim(0, 1)
    ax_inset.set_xlim(-1, len(means))
    ax_inset.set_yticks([0, 1])
    ax_inset.set_xticks([])
    ax_inset.set_ylabel("F1", fontsize=7, labelpad=1)
    ax_inset.tick_params(labelsize=6)


def _failure_panel_modality(ax_inset):
    bars = [0.78, 0.74, 0.71, 0.18, 0.76]
    labels = ["full", "−ACC", "−BVP", "−EDA", "−TEMP"]
    colors = [PALETTE["cross_attention"]] * 5
    colors[3] = PALETTE["stress"]
    ax_inset.bar(range(5), bars, color=colors, edgecolor="white", linewidth=0.5)
    ax_inset.set_xticks(range(5))
    ax_inset.set_xticklabels(labels, fontsize=6)
    ax_inset.set_ylim(0, 1)
    ax_inset.set_yticks([0, 1])
    ax_inset.set_ylabel("F1", fontsize=7, labelpad=1)
    ax_inset.tick_params(labelsize=6)


def _failure_panel_noise(ax_inset):
    sigmas = np.array([0, 0.1, 0.5, 1.0, 2.0])
    f1 = np.array([0.78, 0.72, 0.55, 0.38, 0.22])
    ax_inset.plot(sigmas, f1, "-o", color=PALETTE["cross_attention"],
                  ms=3.5, lw=1.2, mec="white", mew=0.5)
    ax_inset.fill_between(sigmas, f1 - 0.05, f1 + 0.05,
                          color=PALETTE["cross_attention"], alpha=0.18)
    ax_inset.axhline(0.3, color=PALETTE["baseline"], linestyle="--", lw=0.6)
    ax_inset.set_ylim(0, 1)
    ax_inset.set_xlim(0, 2.1)
    ax_inset.set_yticks([0, 1])
    ax_inset.set_xticks([0, 1, 2])
    ax_inset.set_xlabel("noise σ", fontsize=6, labelpad=1)
    ax_inset.set_ylabel("F1", fontsize=7, labelpad=1)
    ax_inset.tick_params(labelsize=6)


def _generative_prior_inset(ax):
    """Tiny schematic: 4 modality circles -> shared latent <- 4 modality circles."""
    ax.set_xlim(-0.2, 10.2); ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    mods = ["ACC", "BVP", "EDA", "TEMP"]
    ys = [3.4, 2.6, 1.8, 1.0]
    # Left-side input modalities. Larger radius (0.45) ensures every
    # 3-4 char label clears the circle border with margin. zorder explicit
    # so arrows draw beneath, circle border above, text on top.
    for y, m in zip(ys, mods):
        c = mpatches.Circle((1.0, y), 0.45, ec=PALETTE["cross_modal_recon"],
                            fc="white", lw=1.0, zorder=3)
        ax.add_patch(c)
        ax.text(1.0, y, m, ha="center", va="center", fontsize=5.5,
                color=PALETTE["annotation"], zorder=4)
    # latent — widened to 2.6 to keep label inside on every renderer
    latent = FancyBboxPatch((3.7, 1.3), 2.6, 1.8,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            ec=PALETTE["cross_modal_recon"],
                            fc="white", lw=1.4, zorder=3)
    ax.add_patch(latent)
    ax.text(5.0, 2.55, "shared", ha="center", va="center", fontsize=6.5,
            color=PALETTE["annotation"], fontweight="bold", zorder=4)
    ax.text(5.0, 2.20, "latent", ha="center", va="center", fontsize=6.5,
            color=PALETTE["annotation"], fontweight="bold", zorder=4)
    ax.text(5.0, 1.85, "prior", ha="center", va="center", fontsize=6.5,
            color=PALETTE["annotation"], fontweight="bold", zorder=4)
    # Right-side reconstructed modalities (dashed border).
    for y, m in zip(ys, mods):
        c = mpatches.Circle((8.6, y), 0.45, ec=PALETTE["cross_modal_recon"],
                            fc="white", lw=1.0, ls="--", zorder=3)
        ax.add_patch(c)
        ax.text(8.6, y, m, ha="center", va="center", fontsize=5.5,
                color=PALETTE["annotation"], zorder=4)
    # Arrows: shrinkA/shrinkB extra so they start/end outside the larger circles.
    for y in ys:
        _arrow(ax, 1.0, y, 3.7, 2.2, color=PALETTE["cross_modal_recon"],
               lw=0.8, shrinkA=12, shrinkB=4)
    for y in ys:
        _arrow(ax, 6.3, 2.2, 8.6, y, color=PALETTE["cross_modal_recon"],
               lw=0.8, shrinkA=4, shrinkB=12)
    ax.text(5.0, 0.45, "each modality reconstructs the others",
            ha="center", va="center", fontsize=7, style="italic",
            color=PALETTE["annotation"])


def render(out_path: Path) -> None:
    apply_style()
    fig = plt.figure(figsize=(7.2, 5.0))

    gs = fig.add_gridspec(3, 3, height_ratios=[2.4, 0.3, 1.4],
                          left=0.04, right=0.98, top=0.94, bottom=0.04,
                          hspace=0.0, wspace=0.18)

    titles = [
        "1. Per-subject collapse",
        "2. Modality shortcut",
        "3. Sensor degradation",
    ]
    blurbs = [
        "Aggregate accuracy hides\ncatastrophic failure on\na subset of subjects",
        "Removing one sensor at test\ntime collapses prediction.\nModel leaned on one modality.",
        "Test-time noise drops F1\ntoward chance across\narchitectures.",
    ]
    inset_drawers = [_failure_panel_subject, _failure_panel_modality, _failure_panel_noise]

    column_axes = []
    for col, (title, blurb, drawer) in enumerate(zip(titles, blurbs, inset_drawers)):
        ax = fig.add_subplot(gs[0, col])
        ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        # Header box
        _box(ax, 0.04, 0.83, 0.92, 0.13,
             ec=PALETTE["annotation"], fc="#FAFAFA", lw=1.0)
        ax.text(0.5, 0.895, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=PALETTE["annotation"])

        # Inset axes for tiny figure
        sub_ax = ax.inset_axes([0.10, 0.43, 0.80, 0.32])
        drawer(sub_ax)
        for s in ("top", "right"):
            sub_ax.spines[s].set_visible(False)

        # Blurb
        ax.text(0.5, 0.27, blurb, ha="center", va="top",
                fontsize=8, color=PALETTE["annotation"])

        column_axes.append(ax)

    # Convergence arrows row -> central conclusion. Three parallel arrows of
    # equal length, hitting the conclusion-box top at evenly-spaced anchors —
    # reads as "three independent failures of the same problem" without
    # arrowhead pile-up.
    ax_arrow = fig.add_subplot(gs[1, :])
    ax_arrow.axis("off")
    ax_arrow.set_xlim(0, 1); ax_arrow.set_ylim(0, 1)
    for src_x, tgt_x in [(0.18, 0.30), (0.50, 0.50), (0.82, 0.70)]:
        _arrow(ax_arrow, src_x, 0.95, tgt_x, 0.10,
               color=PALETTE["annotation"], lw=1.2)

    # Bottom row: conclusion box (everything inside) + generative prior inset.
    ax_concl = fig.add_subplot(gs[2, 0:2])
    ax_concl.axis("off"); ax_concl.set_xlim(0, 1); ax_concl.set_ylim(0, 1)
    _box(ax_concl, 0.02, 0.05, 0.96, 0.92,
         ec=PALETTE["stress"], fc="#FFF5F5", lw=1.6)
    ax_concl.text(0.5, 0.85, "Three faces of one structural problem:",
                  ha="center", va="center", fontsize=10,
                  color=PALETTE["annotation"])
    ax_concl.text(0.5, 0.66,
                  "distribution shift that current fusion\narchitectures cannot bridge",
                  ha="center", va="center", fontsize=11, fontweight="bold",
                  color=PALETTE["stress"])
    ax_concl.text(0.5, 0.32, "Proposed direction:",
                  ha="center", va="center", fontsize=9, style="italic",
                  color=PALETTE["annotation"])
    ax_concl.text(0.5, 0.16,
                  "neuro-inspired architectural priors\n(e.g., cross-modal generative — see inset)",
                  ha="center", va="center", fontsize=8.5, style="italic",
                  color=PALETTE["annotation"])

    ax_inset = fig.add_subplot(gs[2, 2])
    _generative_prior_inset(ax_inset)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"saved {out_path}.pdf and .png")


if __name__ == "__main__":
    render(Path("figures/fig1_thesis"))
