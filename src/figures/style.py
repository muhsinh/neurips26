"""Global figure style. Apply via `from src.figures.style import apply_style; apply_style()`."""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "late_fusion_mlp": "#5B8DEF",
    "cross_attention": "#E07B91",
    "scale_proxy": "#7BA05B",
    "cross_modal_recon": "#C77DFF",
    "annotation": "#333333",
    "baseline": "#888888",
    "background": "#FFFFFF",
    "stress": "#D7263D",
    "non_stress": "#4A7C59",
}

MODEL_LABELS = {
    "late_fusion_mlp": "Late-fusion MLP",
    "cross_attention": "Cross-attention",
    "scale_proxy": "Scale proxy",
    "cross_modal_recon": "Cross-modal recon",
}

MODEL_PARAMS = {
    "late_fusion_mlp": "~50k",
    "cross_attention": "~150k",
    "scale_proxy": "~2M",
    "cross_modal_recon": "~150k",
}


def apply_style() -> None:
    """Apply global rcParams. Inter font preferred, falls back gracefully."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_label(ax, label: str, x: float = -0.18, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left")
