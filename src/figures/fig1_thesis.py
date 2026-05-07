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
    ax_inset.set_xlim(-1, len(means) + 3)
    ax_inset.set_yticks([0, 1])
    ax_inset.set_xticks([])
    ax_inset.set_ylabel("F1", fontsize=7, labelpad=1)
    ax_inset.set_xlabel("subject id (sorted)", fontsize=6, labelpad=1)
    ax_inset.tick_params(labelsize=6)
    ax_inset.text(len(means) + 1, 0.30, "chance", fontsize=5,
                  color=PALETTE["baseline"], va="center")


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
    ax_inset.text(3, 0.22, "EDA collapse", fontsize=5,
                  color=PALETTE["stress"], ha="center", fontweight="bold")


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
    """Tiny schematic: 4 modality ellipses -> shared latent <- 4 modality ellipses.

    Use horizontal ellipses (wider than tall) instead of circles so 3-4 char
    labels (ACC, BVP, EDA, TEMP) fit cleanly inside with margin at every
    render scale.
    """
    ax.set_xlim(-0.4, 10.4); ax.set_ylim(0, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")
    mods = ["ACC", "BVP", "EDA", "TEMP"]
    ys = [3.4, 2.6, 1.8, 1.0]
    ell_w, ell_h = 1.6, 0.65  # wider than tall — text fits
    # Left-side input modalities.
    for y, m in zip(ys, mods):
        e = mpatches.Ellipse((1.0, y), ell_w, ell_h,
                             ec=PALETTE["cross_modal_recon"],
                             fc="white", lw=1.0, zorder=3)
        ax.add_patch(e)
        ax.text(1.0, y, m, ha="center", va="center", fontsize=7,
                color=PALETTE["annotation"], zorder=4)
    # Latent box.
    latent = FancyBboxPatch((3.9, 1.3), 2.4, 1.8,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            ec=PALETTE["cross_modal_recon"],
                            fc="white", lw=1.4, zorder=3)
    ax.add_patch(latent)
    ax.text(5.1, 2.60, "shared", ha="center", va="center", fontsize=7.5,
            color=PALETTE["annotation"], fontweight="bold", zorder=4)
    ax.text(5.1, 2.20, "latent", ha="center", va="center", fontsize=7.5,
            color=PALETTE["annotation"], fontweight="bold", zorder=4)
    ax.text(5.1, 1.80, "prior", ha="center", va="center", fontsize=7.5,
            color=PALETTE["annotation"], fontweight="bold", zorder=4)
    # Right-side reconstructed modalities (dashed).
    for y, m in zip(ys, mods):
        e = mpatches.Ellipse((9.0, y), ell_w, ell_h,
                             ec=PALETTE["cross_modal_recon"],
                             fc="white", lw=1.0, ls="--", zorder=3)
        ax.add_patch(e)
        ax.text(9.0, y, m, ha="center", va="center", fontsize=7,
                color=PALETTE["annotation"], zorder=4)
    # Arrows shrunk well outside the ellipse perimeter.
    for y in ys:
        _arrow(ax, 1.0, y, 3.9, 2.2, color=PALETTE["cross_modal_recon"],
               lw=0.8, shrinkA=20, shrinkB=4)
    for y in ys:
        _arrow(ax, 6.3, 2.2, 9.0, y, color=PALETTE["cross_modal_recon"],
               lw=0.8, shrinkA=4, shrinkB=20)
    # Encode / decode labels.
    ax.text(2.45, 3.05, "encode", ha="center", va="center",
            fontsize=6.5, style="italic", color=PALETTE["annotation"])
    ax.text(7.55, 3.05, "decode-to-others", ha="center", va="center",
            fontsize=6.5, style="italic", color=PALETTE["annotation"])
    ax.text(5.1, 0.45, "each modality reconstructs the others",
            ha="center", va="center", fontsize=7, style="italic",
            color=PALETTE["annotation"])


def render(out_path: Path) -> None:
    apply_style()
    fig = plt.figure(figsize=(7.2, 5.6))

    gs = fig.add_gridspec(3, 3, height_ratios=[2.0, 0.25, 1.85],
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
    shift_labels = ["(subject shift)", "(modality shift)", "(corruption shift)"]

    column_axes = []
    for col, (title, blurb, drawer, shift) in enumerate(
            zip(titles, blurbs, inset_drawers, shift_labels)):
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
        # Shift-type label
        ax.text(0.5, 0.10, shift, ha="center", va="top",
                fontsize=7.5, style="italic", color=PALETTE["stress"])

        column_axes.append(ax)

    # Three parallel vertical arrows: same length, same angle, evenly spaced.
    # Reads as "three independent failures of the same problem" — no
    # asymmetric diagonals.
    ax_arrow = fig.add_subplot(gs[1, :])
    ax_arrow.axis("off")
    ax_arrow.set_xlim(0, 1); ax_arrow.set_ylim(0, 1)
    for col_x in (0.18, 0.50, 0.82):
        _arrow(ax_arrow, col_x, 0.95, col_x, 0.10,
               color=PALETTE["annotation"], lw=1.2)

    # Bottom row: conclusion box (slim) + generative prior inset.
    ax_concl = fig.add_subplot(gs[2, 0:2])
    ax_concl.axis("off"); ax_concl.set_xlim(0, 1); ax_concl.set_ylim(0, 1)
    _box(ax_concl, 0.02, 0.05, 0.96, 0.92,
         ec=PALETTE["stress"], fc="#FFF5F5", lw=1.6)
    ax_concl.text(0.5, 0.70, "Three faces of one structural problem",
                  ha="center", va="center", fontsize=10,
                  color=PALETTE["annotation"])
    ax_concl.text(0.5, 0.42,
                  "distribution shift current fusion\narchitectures cannot bridge",
                  ha="center", va="center", fontsize=12, fontweight="bold",
                  color=PALETTE["stress"])

    ax_inset = fig.add_subplot(gs[2, 2])
    _generative_prior_inset(ax_inset)

    # Figure-level arrow from conclusion box to inset, labeled "addressed by".
    try:
        fig.canvas.draw()
        concl_bbox = ax_concl.get_position()
        inset_bbox = ax_inset.get_position()
        arrow_y = (concl_bbox.y0 + concl_bbox.y1) / 2
        x_start = concl_bbox.x1 - 0.005
        x_end = inset_bbox.x0 + 0.005
        link_arrow = FancyArrowPatch(
            (x_start, arrow_y), (x_end, arrow_y),
            transform=fig.transFigure,
            arrowstyle="-|>", color=PALETTE["stress"], lw=1.4,
            mutation_scale=14, shrinkA=2, shrinkB=2, zorder=5,
        )
        fig.add_artist(link_arrow)
        fig.text((x_start + x_end) / 2, arrow_y + 0.018,
                 "addressed by:", ha="center", va="bottom",
                 fontsize=8, style="italic", fontweight="bold",
                 color=PALETTE["stress"])
    except Exception:
        # Fallback: inline annotation inside the conclusion box.
        ax_concl.text(0.96, 0.15, "↓ addressed by",
                      ha="right", va="center", fontsize=8,
                      style="italic", fontweight="bold",
                      color=PALETTE["stress"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"saved {out_path}.pdf and .png")


if __name__ == "__main__":
    render(Path("figures/fig1_thesis"))
