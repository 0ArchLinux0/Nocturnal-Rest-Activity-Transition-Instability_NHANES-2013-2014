"""Shared ROC figure style: rcParams + color map for publication figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

ROC_COLORS = {
    "p01_model": "#1f77b4",  # Royal blue — P01 + covariates / composite
    "baseline": "#ff7f0e",  # Orange — demographics-only baseline
    "standalone": "#2ca02c",  # Green — P01-only
    "chance": "#7f7f7f",  # Gray — diagonal reference
}


def apply_roc_rcparams() -> None:
    """Publication-friendly font sizes for ROC and other paper figures."""
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 16,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 18,
        }
    )
