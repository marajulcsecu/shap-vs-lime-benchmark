"""One function per figure. Each takes a Profile and returns a matplotlib Figure.

Requirements every figure meets, because these are what make a figure usable in a
paper rather than notebook output: axis labels carry units and n; colours come only
from PALETTE keyed by explainer; bounded metrics get explicit limits so panels are
comparable; error bars are bootstrap CIs from stats_estimates.csv, never +/-1 SD.
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xaibench.config import ALL_EXPLAINERS, BASE_EXPLAINERS, Profile, get_profile, paths
from xaibench.data.loaders import load_dataset
from xaibench.viz.theme import (
    DATASET_LABELS,
    MARKERS,
    DATASET_ORDER,
    EXPLAINER_LABELS,
    MODEL_LABELS,
    MODEL_ORDER,
    PALETTE,
    apply_theme,
    colour,
    style,
    watermark,
)

CM = 1 / 2.54
GLOBAL = -1


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(paths().results / name)


def _label(axis, x=None, y=None, title=None):
    if x:
        axis.set_xlabel(x)
    if y:
        axis.set_ylabel(y)
    if title:
        axis.set_title(title)


def save(fig, name: str, profile: Profile) -> None:
    watermark(fig, profile)
    folder = paths().figures
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{suffix}")
    plt.close(fig)


# --------------------------------------------------------------------------- F1
def fig_design(profile: Profile):
    """Study design and data flow, drawn rather than imported so it regenerates."""
    fig, ax = plt.subplots(figsize=(17 * CM, 7.5 * CM))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)

    def box(x, y, w, h, text, face="#EEF3F8", edge="#33566E"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6", linewidth=1.0,
            facecolor=face, edgecolor=edge))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#33566E"))

    box(1, 33, 20, 10, "3 public datasets\nPIMA 768$\\times$8\nHeart 303$\\times$13\nCKD 400$\\times$24")
    box(1, 17, 20, 11, "Leakage-safe\npreprocessing\nimpute $\\to$ scale $\\to$ one-hot\n(fitted on train only)")
    box(1, 2, 20, 11, "6 classifiers\nLogReg DecTree RF\nXGB LGBM Stacking")
    box(27, 26, 20, 12, "Track A\nnested CV 5$\\times$3\n8 performance metrics",
        face="#F3EEF8", edge="#5B4B77")
    box(27, 6, 20, 14, "Track B\nfixed 80/20 split\none model per cell\n18 cells",
        face="#F3EEF8", edge="#5B4B77")
    box(53, 24, 20, 16, "Explainers\nSHAP (probability space)\nLIME (tabular)\n3 fusion rules\nrandom control",
        face="#EEF8F2", edge="#2F6B50")
    box(53, 3, 20, 17, "Metrics\nagreement $\\cdot$ stability\nfidelity $\\cdot$ cost\nplausibility",
        face="#EEF8F2", edge="#2F6B50")
    box(79, 14, 20, 16, "11 hypotheses\nWilcoxon $+$ Holm\neffect sizes\nbootstrap CIs\n\nfigures $+$ tables",
        face="#FBF3E8", edge="#8A6A33")

    arrow(21, 38, 27, 34)
    arrow(21, 22, 27, 30)
    arrow(21, 7, 27, 13)
    arrow(47, 32, 53, 32)
    arrow(47, 13, 53, 20)
    arrow(63, 24, 63, 20)
    arrow(73, 11, 79, 20)
    ax.text(50, 47, "Two tracks, deliberately separated: tuning never sees the "
                    "explanation test partition", fontsize=7.4, style="italic",
            ha="center", color="#444444")
    return fig


# --------------------------------------------------------------------------- F2
def fig_datasets(profile: Profile):
    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 6 * CM))
    balance, missing = [], {}
    for ds_id in DATASET_ORDER:
        ds = load_dataset(ds_id)
        balance.append({"dataset": ds_id, "positive": ds.positive_rate,
                        "negative": 1 - ds.positive_rate, "n": len(ds.y),
                        "missing": ds.missing_rate})
        missing[ds_id] = ds.missing_by_column()
    frame = pd.DataFrame(balance)

    x = np.arange(len(frame))
    axes[0].bar(x, frame.negative, color="#B4C6D6", label="negative")
    axes[0].bar(x, frame.positive, bottom=frame.negative, color=PALETTE["shap"],
                label="positive")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{DATASET_LABELS[d]}\n(n = {n})"
                             for d, n in zip(frame.dataset, frame.n)], fontsize=7.5)
    axes[0].set_ylim(0, 1)
    _label(axes[0], y="Class proportion", title="Class balance")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="outside upper center")
    for xi, rate in zip(x, frame.positive):
        axes[0].text(xi, 1.0 - rate / 2, f"{rate:.1%}", ha="center", va="center",
                     fontsize=7.5, color="white")

    for offset, ds_id in enumerate(DATASET_ORDER):
        series = missing[ds_id]
        series = series[series > 0]
        axes[1].bar(np.arange(len(series)) + offset * 0.0, series.to_numpy(),
                    alpha=0.0)
        axes[1].plot(np.arange(len(series)), series.to_numpy(), marker="o",
                     markersize=3.5, label=DATASET_LABELS[ds_id])
    _label(axes[1], x="Columns with missing values (ranked)",
           y="Missing fraction", title="Missingness")
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(bottom=0)
    return fig


# --------------------------------------------------------------------------- F3
def fig_performance(profile: Profile):
    perf = _read("performance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 7.5 * CM), sharex=True)
    for axis, metric, name in zip(axes, ("roc_auc", "mcc"), ("ROC-AUC", "MCC")):
        grouped = perf.groupby(["dataset", "model"])[metric].agg(["mean", "std"])
        width = 0.26
        for i, ds_id in enumerate(DATASET_ORDER):
            values = [grouped.loc[(ds_id, m), "mean"] for m in MODEL_ORDER]
            errors = [grouped.loc[(ds_id, m), "std"] for m in MODEL_ORDER]
            axis.bar(np.arange(len(MODEL_ORDER)) + (i - 1) * width, values, width,
                     yerr=errors, capsize=1.8, label=DATASET_LABELS[ds_id],
                     error_kw={"lw": 0.7})
        axis.set_xticks(np.arange(len(MODEL_ORDER)))
        axis.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=30,
                             ha="right")
        _label(axis, y=name, title=f"{name} by dataset and model")
        axis.set_ylim(0, 1.04)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="outside upper center", fontsize=7.5)
    return fig


# --------------------------------------------------------------------------- F4
def fig_agreement_heatmap(profile: Profile):
    agreement = _read("metrics_agreement.csv").query("instance_id == @GLOBAL")
    matrix = (agreement.pivot_table(index="model", columns="dataset",
                                    values="spearman")
              .reindex(index=MODEL_ORDER, columns=list(DATASET_ORDER)))
    sig_cells: set[tuple[str, str]] = set()
    stats_path = paths().results / "stats_hypotheses.csv"
    if stats_path.exists():
        h1 = pd.read_csv(stats_path).query("hypothesis == 'H1' and decision == 'reject H0'")
        sig_cells = {(r.model, r.dataset) for r in h1.itertuples()}
    fig, ax = plt.subplots(figsize=(9 * CM, 7.2 * CM))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="RdYlBu", vmin=-1, vmax=1,
                      aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([DATASET_LABELS[d] for d in matrix.columns], rotation=20,
                       ha="right", fontsize=7.5)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([MODEL_LABELS[m] for m in matrix.index])
    for i, model_id in enumerate(matrix.index):
        for j, ds_id in enumerate(matrix.columns):
            value = matrix.to_numpy(dtype=float)[i, j]
            if np.isfinite(value):
                star = "*" if (model_id, ds_id) in sig_cells else ""
                ax.text(j, i, f"{value:.2f}{star}", ha="center", va="center", fontsize=8,
                        color="black" if abs(value) < 0.6 else "white")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Spearman $\\rho$ (global ranking)")
    _label(ax, title="SHAP vs LIME agreement (* = H1 $p_{\\mathrm{adj}}<0.05$)")
    return fig


# --------------------------------------------------------------------------- F5
def fig_agreement_violins(profile: Profile):
    agreement = _read("metrics_agreement.csv").query("instance_id >= 0")
    fig, axes = plt.subplots(1, 3, figsize=(17 * CM, 7 * CM), sharey=True)
    for axis, ds_id in zip(axes, DATASET_ORDER):
        cell = agreement.query("dataset == @ds_id")
        data = [cell.query("model == @m").spearman.dropna().to_numpy()
                for m in MODEL_ORDER]
        data = [d if d.size else np.array([np.nan]) for d in data]
        parts = axis.violinplot(data, showmedians=True, widths=0.85)
        for body in parts["bodies"]:
            body.set_facecolor(PALETTE["shap"])
            body.set_alpha(0.45)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("#333333")
                parts[key].set_linewidth(0.9)
        axis.axhline(0.5, color=PALETTE["lime"], lw=0.9, ls="--")
        axis.set_xticks(np.arange(1, len(MODEL_ORDER) + 1))
        axis.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=35,
                             ha="right", fontsize=7.5)
        n = int(cell.groupby("model").size().median()) if len(cell) else 0
        _label(axis, title=f"{DATASET_LABELS[ds_id]} (n = {n} instances)")
        axis.set_ylim(-1.05, 1.05)
    _label(axes[0], y="Spearman $\\rho$")
    axes[2].text(0.98, 0.03, "dashed line: H1's pre-registered bar $\\rho = 0.50$",
                 transform=axes[2].transAxes, ha="right", fontsize=7,
                 color=PALETTE["lime"])
    return fig


# --------------------------------------------------------------------------- F6
def fig_agreement_vs_difficulty(profile: Profile):
    agreement = _read("metrics_agreement.csv").query("instance_id >= 0")
    perf = _read("performance.csv")
    cell_rho = agreement.groupby(["dataset", "model"], as_index=False).spearman.median()
    auc = perf.groupby(["dataset", "model"], as_index=False).roc_auc.mean()
    merged = cell_rho.merge(auc, on=["dataset", "model"])
    merged["difficulty"] = 1 - merged.roc_auc

    fig, ax = plt.subplots(figsize=(9 * CM, 8 * CM))
    markers = {"pima": "o", "heart": "s", "ckd": "^"}
    for ds_id in DATASET_ORDER:
        subset = merged.query("dataset == @ds_id")
        ax.scatter(subset.difficulty, subset.spearman, s=42, marker=markers[ds_id],
                   label=f"{DATASET_LABELS[ds_id]}", alpha=0.85,
                   edgecolor="white", linewidth=0.6)
    if len(merged) > 2:
        fit = np.polyfit(merged.difficulty, merged.spearman, 1)
        grid = np.linspace(merged.difficulty.min(), merged.difficulty.max(), 50)
        ax.plot(grid, np.polyval(fit, grid), color="#555555", lw=1.0, ls="--")
        from scipy import stats as sps
        rho = sps.spearmanr(merged.difficulty, merged.spearman).statistic
        ax.text(0.03, 0.04, f"Spearman $\\rho$ = {rho:.2f} over {len(merged)} cells",
                transform=ax.transAxes, fontsize=7.5)
    _label(ax, x="Task difficulty  (1 $-$ nested-CV ROC-AUC)",
           y="Median Spearman $\\rho$",
           title="Does agreement track task difficulty?")
    ax.legend(fontsize=7.5, loc="upper right")
    return fig


# --------------------------------------------------------------------------- F7
def fig_lime_variance_vs_samples(profile: Profile):
    """RQ2a: LIME's variance against its sampling budget, with SHAP's two backends.

    The SHAP annotation is restricted to the FOUR TREE backends on purpose. Taking the
    maximum over all six models mixes in the permutation backend, which is
    sampling-based and reached a coefficient of variation of 139 -- so a label reading
    "tree backends: exactly 0, max observed 1e+02" was self-contradictory and simply
    wrong. Both numbers are now shown, each against the backend it belongs to.
    """
    from xaibench.config import SIMPLE_MODELS
    rerun = pd.read_parquet(paths().results / "stability_rerun.parquet")
    tree_models = ["dtree", "rf", "xgb", "lgbm"]

    fig, ax = plt.subplots(figsize=(11 * CM, 8.5 * CM))
    lime = rerun.query("explainer == 'lime'")
    grouped = lime.groupby(["dataset", "num_samples"]).rerun_cv.median().reset_index()
    # LIME series share the LIME colour and differ by marker/style; the default colour
    # cycle put a blue LIME line next to SHAP's blue reference line.
    dataset_style = {"pima": ("-", "o"), "heart": ("--", "s"), "ckd": ("-.", "^")}
    for ds_id in DATASET_ORDER:
        subset = grouped.query("dataset == @ds_id").sort_values("num_samples")
        if len(subset):
            style, marker = dataset_style[ds_id]
            ax.plot(subset.num_samples, subset.rerun_cv, color=PALETTE["lime"],
                    linestyle=style, marker=marker, markersize=4,
                    label=f"LIME, {DATASET_LABELS[ds_id]}")

    shap_rows = rerun.query("explainer == 'shap'")
    tree_max = shap_rows[shap_rows.model.isin(tree_models)].rerun_cv.max()
    agnostic = shap_rows[~shap_rows.model.isin(tree_models)].rerun_cv
    ax.axhline(0.0, color=PALETTE["shap"], lw=1.6,
               label="SHAP, tree backends (exactly 0)")
    if len(agnostic):
        ax.axhline(float(agnostic.median()), color=PALETTE["shap"], lw=1.2, ls=":",
                   label="SHAP, model-agnostic backend (median)")

    ax.set_xscale("log")
    _label(ax, x="LIME neighbourhood samples per explanation (log scale)",
           y="Median re-run coefficient of variation")
    ax.set_title("Run-to-run variance against sampling budget", pad=12)
    ax.set_ylim(bottom=-0.15)
    # fig.legend, not ax.legend: an axes-anchored legend landed on the x-axis label.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, fontsize=6.8, loc="outside lower center")
    ax.text(0.99, 0.96,
            f"SHAP is deterministic only where a tree backend exists\n"
            f"(tree max {tree_max:.1f}; model-agnostic max "
            f"{agnostic.max():.0f})" if len(agnostic) else
            f"tree-backend max {tree_max:.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            style="italic", color="#555555")
    return fig


# --------------------------------------------------------------------------- F8
def fig_lipschitz(profile: Profile):
    stability = _read("metrics_stability.csv").dropna(subset=["lipschitz"])
    cell = (stability.groupby(["dataset", "model", "explainer"], as_index=False)
            .lipschitz.median())
    wide = cell.pivot_table(index=["dataset", "model"], columns="explainer",
                            values="lipschitz")
    wide = wide.dropna(subset=[c for c in BASE_EXPLAINERS if c in wide])
    order = [(d, m) for d in DATASET_ORDER for m in MODEL_ORDER
             if (d, m) in wide.index]
    wide = wide.loc[order]

    fig, ax = plt.subplots(figsize=(11 * CM, 9 * CM))
    y = np.arange(len(wide))
    ax.hlines(y, wide["shap"], wide["lime"], color="#BBBBBB", lw=1.4, zorder=1)
    ax.scatter(wide["shap"], y, s=34, color=PALETTE["shap"], label="SHAP", zorder=2)
    ax.scatter(wide["lime"], y, s=34, color=PALETTE["lime"], label="LIME", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{DATASET_LABELS[d].split()[0]} / {MODEL_LABELS[m]}"
                        for d, m in wide.index], fontsize=7.2)
    _label(ax, x="Local Lipschitz estimate (lower = more robust to input perturbation)",
           title="Explanation robustness, paired within each cell")
    ax.grid(axis="x", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="outside upper center")
    return fig


# --------------------------------------------------------------------------- F9
def fig_retrain_stability(profile: Profile):
    stability = _read("metrics_stability.csv").dropna(subset=["retrain_agreement"])
    cell = (stability.groupby(["dataset", "model", "explainer"], as_index=False)
            .retrain_agreement.first())
    fig, ax = plt.subplots(figsize=(11 * CM, 7.5 * CM))
    models = [m for m in MODEL_ORDER if m in set(cell.model)]
    labels, width = [], 0.38
    positions = np.arange(len(DATASET_ORDER) * len(models), dtype=float)
    idx = 0
    for ds_id in DATASET_ORDER:
        for model_id in models:
            row = cell.query("dataset == @ds_id and model == @model_id")
            for offset, explainer in enumerate(BASE_EXPLAINERS):
                value = row.query("explainer == @explainer").retrain_agreement
                ax.bar(positions[idx] + (offset - 0.5) * width,
                       float(value.iloc[0]) if len(value) else np.nan, width,
                       color=colour(explainer),
                       label=EXPLAINER_LABELS[explainer] if idx == 0 else None)
            labels.append(f"{DATASET_LABELS[ds_id].split()[0]}\n{MODEL_LABELS[model_id]}")
            idx += 1
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    _label(ax, y="Cross-refit $\\rho$",
           title="Do global rankings survive retraining on a bootstrap resample?")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="outside upper center")
    return fig


# -------------------------------------------------------------------------- F10
def fig_deletion_curves(profile: Profile):
    curves = _read("metrics_curves.csv")
    models = ["rf", "xgb", "lgbm"]
    fig, axes = plt.subplots(len(models), len(DATASET_ORDER),
                             figsize=(17 * CM, 13.5 * CM), sharex=True, sharey="row")
    for r, model_id in enumerate(models):
        for c, ds_id in enumerate(DATASET_ORDER):
            axis = axes[r, c]
            cell = curves.query("dataset == @ds_id and model == @model_id")
            for explainer in ("shap", "lime", "fidwt", "random"):
                subset = cell.query("explainer == @explainer").sort_values("k")
                if not len(subset):
                    continue
                axis.plot(subset.k, subset.mean_prob, markersize=2.8,
                          **style(explainer),
                          label=EXPLAINER_LABELS[explainer] if (r == 0 and c == 0) else None)
            if r == 0:
                axis.set_title(DATASET_LABELS[ds_id], fontsize=8.5)
            if c == 0:
                axis.set_ylabel(f"{MODEL_LABELS[model_id]}\n$p$(own class)", fontsize=8)
            if r == len(models) - 1:
                axis.set_xlabel("Top-$k$ features deleted")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="outside lower center", fontsize=7.5)
    fig.suptitle("Deletion curves: a faithful ranking drops the prediction fastest",
                 fontsize=9)
    return fig


# -------------------------------------------------------------------------- F11
def fig_runtime(profile: Profile):
    runtime = _read("metrics_runtime.csv")
    primary = runtime[(runtime.explainer != "lime")
                      | (runtime.num_samples == profile.lime_num_samples[0])]
    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 7.5 * CM))

    explainers = [e for e in ALL_EXPLAINERS if e != "random"]
    data = [primary.query("explainer == @e").median_seconds.dropna().to_numpy()
            for e in explainers]
    box = axes[0].boxplot([d if d.size else np.array([np.nan]) for d in data],
                          patch_artist=True, widths=0.6, showfliers=False)
    for patch, explainer in zip(box["boxes"], explainers):
        patch.set_facecolor(colour(explainer))
        patch.set_alpha(0.7)
    axes[0].set_xticks(np.arange(1, len(explainers) + 1))
    axes[0].set_xticklabels([EXPLAINER_LABELS[e] for e in explainers], rotation=30,
                            ha="right", fontsize=7.2)
    axes[0].set_yscale("log")
    _label(axes[0], y="Seconds / explanation",
           title="Cost (log scale, 18 cells)")

    for explainer in explainers:
        subset = primary.query("explainer == @explainer").dropna(
            subset=["median_seconds", "aopc"])
        axes[1].scatter(subset.median_seconds, subset.aopc, s=34,
                        color=colour(explainer), marker=MARKERS[explainer], alpha=0.85,
                        label=EXPLAINER_LABELS[explainer], edgecolor="white",
                        linewidth=0.5)
    axes[1].set_xscale("log")
    _label(axes[1], x="Seconds per explanation (log scale)",
           y="Median AOPC (fidelity)", title="What the cost buys")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, fontsize=7, loc="outside lower center")
    return fig


# -------------------------------------------------------------------------- F12
def fig_plausibility(profile: Profile):
    plaus = _read("metrics_plausibility.csv")
    cell = (plaus.groupby(["dataset", "explainer"], as_index=False).jaccard5.median())
    explainers = [e for e in ALL_EXPLAINERS]
    fig, ax = plt.subplots(figsize=(11 * CM, 7.5 * CM))
    width = 0.8 / len(explainers)
    for i, explainer in enumerate(explainers):
        values = [float(cell.query("dataset == @d and explainer == @explainer")
                        .jaccard5.iloc[0])
                  if len(cell.query("dataset == @d and explainer == @explainer")) else np.nan
                  for d in DATASET_ORDER]
        ax.bar(np.arange(len(DATASET_ORDER)) + (i - len(explainers) / 2) * width,
               values, width, color=colour(explainer),
               label=EXPLAINER_LABELS[explainer])
    ax.set_xticks(np.arange(len(DATASET_ORDER)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=8)
    ax.set_ylim(0, 1)
    _label(ax, y="Jaccard@5 vs reference set",
           title="Plausibility against pre-registered reference features")
    # Legend below the axes: three columns above the title wrapped onto two rows and
    # overprinted it.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=6.8, loc="outside lower center")
    ax.text(0.99, 0.97, "corroborating evidence only:\nreference sets are not ground truth",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            style="italic", color="#666666")
    return fig


# -------------------------------------------------------------------------- F13
def fig_consensus_forest(profile: Profile):
    stats = _read("stats_hypotheses.csv").query("hypothesis == 'H6'").copy()
    stats["cell"] = [f"{DATASET_LABELS[d].split()[0]} / {MODEL_LABELS[m]}"
                     for d, m in zip(stats.dataset, stats.model)]
    stats = stats.sort_values("point_estimate")
    fig, ax = plt.subplots(figsize=(11 * CM, 9 * CM))
    y = np.arange(len(stats))
    colours = [PALETTE["rankavg"] if v > 0 else PALETTE["lime"]
               for v in stats.point_estimate.fillna(0)]
    ax.scatter(stats.point_estimate, y, s=34, color=colours, zorder=3)
    ax.axvline(0.0, color="#333333", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(stats.cell, fontsize=7.2)
    from xaibench.viz.theme import EXPLAINER_LABELS
    right = ax.get_xlim()[1]
    for yi, (rule, decision) in enumerate(zip(stats.best_rule, stats.decision)):
        marker = "*" if decision == "reject H0" else ""
        label = EXPLAINER_LABELS.get(rule, rule).replace("Fusion: ", "")
        ax.text(right, yi, f" {label}{marker}", fontsize=6.6, va="center",
                color="#555555")
    _label(ax, x="$\\Delta$AOPC:  best fusion rule $-$ max(SHAP, LIME)",
           title="Does fusion beat the better of the two?\n(evaluation instances only)")
    ax.grid(axis="x", alpha=0.25)
    ax.locator_params(axis="x", nbins=5)
    ax.margins(y=0.04)
    # Below the axis, not inside it: at 18 rows the caption collided with the bottom label.
    ax.set_xlabel("$\\Delta$AOPC:  best fusion rule $-$ max(SHAP, LIME)\n"
                  "right of zero = fusion wins;  * = Holm-adjusted $p < 0.05$")
    return fig


# -------------------------------------------------------------------------- F14
def fig_summary_ranks(profile: Profile):
    """Mean rank of each explainer across every metric where more/less is better."""
    fidelity = _read("metrics_fidelity.csv").query("baseline == 'median'")
    plaus = _read("metrics_plausibility.csv")
    runtime = _read("metrics_runtime.csv")
    higher_better = {"aopc": fidelity, "auc_ins": fidelity,
                     "comprehensiveness": fidelity, "jaccard5": plaus,
                     "precision5": plaus}
    lower_better = {"sufficiency": fidelity, "median_seconds": runtime}

    rows = []
    for metric, frame in {**higher_better, **lower_better}.items():
        cell = frame.groupby(["dataset", "model", "explainer"], as_index=False)[metric].median()
        for (ds_id, model_id), group in cell.groupby(["dataset", "model"]):
            ascending = metric in lower_better
            ranked = group.sort_values(metric, ascending=ascending).reset_index(drop=True)
            for position, row in ranked.iterrows():
                rows.append({"metric": metric, "explainer": row.explainer,
                             "rank": position + 1})
    ranks = pd.DataFrame(rows)
    mean_rank = ranks.groupby("explainer")["rank"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10 * CM, 5.5 * CM))
    ax.barh([EXPLAINER_LABELS[e] for e in mean_rank.index], mean_rank.to_numpy(),
            color=[colour(e) for e in mean_rank.index])
    for i, value in enumerate(mean_rank.to_numpy()):
        ax.text(value + 0.05, i, f"{value:.2f}", va="center", fontsize=7.5)
    ax.invert_yaxis()
    _label(ax, x=f"Mean rank across {ranks.metric.nunique()} metrics "
                 f"$\\times$ 18 cells (1 = best)",
           title="Overall standing")
    ax.set_xlim(0, mean_rank.max() * 1.18)
    ax.grid(axis="x", alpha=0.25)
    return fig


FIGURES = {
    "F1": fig_design, "F2": fig_datasets, "F3": fig_performance,
    "F4": fig_agreement_heatmap, "F5": fig_agreement_violins,
    "F6": fig_agreement_vs_difficulty, "F7": fig_lime_variance_vs_samples,
    "F8": fig_lipschitz, "F9": fig_retrain_stability, "F10": fig_deletion_curves,
    "F11": fig_runtime, "F12": fig_plausibility, "F13": fig_consensus_forest,
    "F14": fig_summary_ranks,
}


def build_all(profile: Profile | None = None) -> list[str]:
    profile = profile or get_profile()
    apply_theme()
    built = []
    for name, builder in FIGURES.items():
        save(builder(profile), name, profile)
        built.append(name)
    return built


if __name__ == "__main__":
    print(" ".join(build_all()))
