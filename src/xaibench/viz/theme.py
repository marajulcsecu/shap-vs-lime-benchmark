"""One visual system for every figure: colourblind-safe and legible in greyscale."""
from __future__ import annotations

import matplotlib as mpl

# Okabe-Ito, chosen for deuteranopia safety AND separated luminance, so the figures
# survive a black-and-white print of the paper.
PALETTE = {
    "shap": "#0072B2",     # blue
    "lime": "#D55E00",     # vermillion
    "rankavg": "#009E73",  # bluish green
    "rrf": "#CC79A7",      # reddish purple
    "fidwt": "#56B4E9",    # sky blue
    "random": "#7F7F7F",   # grey
}
DATASET_ORDER = ("pima", "heart", "ckd")
MODEL_ORDER = ("logreg", "dtree", "rf", "xgb", "lgbm", "stack")
EXPLAINER_ORDER = ("shap", "lime", "rankavg", "rrf", "fidwt", "random")
DATASET_LABELS = {"pima": "PIMA Diabetes", "heart": "Heart Disease",
                  "ckd": "Chronic Kidney Disease"}
MODEL_LABELS = {"logreg": "LogReg", "dtree": "DecTree", "rf": "RF",
                "xgb": "XGBoost", "lgbm": "LightGBM", "stack": "Stacking"}
EXPLAINER_LABELS = {"shap": "SHAP", "lime": "LIME", "rankavg": "Fusion: mean-rank",
                    "rrf": "Fusion: RRF", "fidwt": "Fusion: fidelity-wt.",
                    "random": "Random control"}


def apply_theme() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "axes.titleweight": "normal",
        "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.25,
        "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.4,
        "figure.constrained_layout.use": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def watermark(fig, profile) -> None:
    """Stamp non-reportable profiles so a draft cannot be mistaken for results."""
    if not profile.is_reportable:
        fig.text(0.5, 0.5, f"PROFILE={profile.name}\nNOT FOR REPORTING",
                 ha="center", va="center", fontsize=26, color="red",
                 alpha=0.18, rotation=30, zorder=1000)


def colour(explainer: str) -> str:
    return PALETTE.get(explainer, "#333333")
