"""Render all figures. Used after Exp1/2/3 complete and aggregate.py has run."""
from __future__ import annotations
from pathlib import Path

from . import fig1_thesis, fig2_diagnostic, fig3_unification, fig4_scale


def main():
    out = Path("figures")
    fig1_thesis.render(out / "fig1_thesis")
    fig2_diagnostic.render(out / "fig2_diagnostic")
    fig3_unification.render(out / "fig3_unification")
    fig4_scale.render(out / "fig4_scale")


if __name__ == "__main__":
    main()
