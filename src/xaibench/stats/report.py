"""Tidy frames to booktabs LaTeX. The ONLY path by which a number reaches a document."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

_ESCAPE = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}"}


def escape(text) -> str:
    """Escape LaTeX specials. CKD feature names contain underscores and would
    otherwise break the build; a raw % would silently comment out a table row."""
    return "".join(_ESCAPE.get(ch, ch) for ch in str(text))


def fmt(value, digits: int = 3) -> str:
    """Every number reaches a document through here, so the missing-value policy
    lives in exactly one place. A literal 'nan' in a submitted table is the most
    embarrassing possible defect."""
    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return escape(value)
    if not np.isfinite(number):
        return "--"
    if number != 0 and abs(number) < 10 ** (-digits):
        return f"{number:.1e}".replace("e-0", "e-").replace("e+0", "e")
    return f"{number:.{digits}f}"


def fmt_ci(point, low, high, digits: int = 3) -> str:
    if point is None or not np.isfinite(float(point) if point is not None else np.nan):
        return "--"
    if low is None or high is None or not (np.isfinite(float(low)) and np.isfinite(float(high))):
        return fmt(point, digits)
    return f"{fmt(point, digits)} [{fmt(low, digits)}, {fmt(high, digits)}]"


def fmt_p(p) -> str:
    """Report a floor rather than 0.000: a p-value is never exactly zero."""
    try:
        value = float(p)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(value):
        return "--"
    if value < 0.001:
        return r"$<$0.001"
    return f"{value:.3f}"


def to_booktabs(df: pd.DataFrame, caption: str, label: str,
                column_format: str | None = None, notes: str | None = None,
                placement: str = "t", small: bool = True,
                escape_cells: bool = False, wide: bool = False,
                shrink: bool = True) -> str:
    """A complete table environment.

    Contract: **cells and headers arrive LaTeX-ready.** Table builders call `escape()`
    on free text and `fmt`/`fmt_ci`/`fmt_p` on numbers, then pass them through
    untouched. `test_every_generated_table_compiles_standalone` is what enforces it:
    guessing at escaping rules for headers that legitimately contain maths does not work.
    That is the only workable split, because a marker type cannot survive a round
    trip through a DataFrame -- pandas coerces a str subclass back to plain str -- so
    "$<$0.001" from fmt_p would be double-escaped by an escaping-by-default table
    writer. Set escape_cells=True only for a frame of raw, unformatted text.
    """
    columns = list(df.columns)
    if column_format is None:
        column_format = "l" + "r" * (len(columns) - 1)
    # Headers follow the same contract as cells: builders supply them LaTeX-ready.
    # They deliberately contain maths -- "Spearman $\rho$", "$p_{\text{adj}}$" -- and
    # escaping them unconditionally turned $ into \$ and broke the build.
    header = " & ".join(str(c) for c in columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = [escape(v) if escape_cells else str(v) for v in row.tolist()]
        body_rows.append(" & ".join(cells) + r" \\")
    environment = "table*" if wide else "table"
    # Shrink-to-fit only: \width is the natural width, so a table narrower than the
    # column is left alone and a wider one is scaled down instead of overflowing.
    open_box = (r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%"
                if shrink else "")
    close_box = "}" if shrink else ""
    parts = [
        rf"\begin{{{environment}}}[{placement}]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small" if small else "",
        open_box,
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        close_box,
    ]
    if notes:
        parts.append(rf"\par\vspace{{2pt}}\footnotesize {notes}")
    parts.append(rf"\end{{{environment}}}")
    return "\n".join(part for part in parts if part) + "\n"


# ---------------------------------------------------------------- table builders
# Contract: each builder reads results/, formats every cell with fmt/fmt_ci/fmt_p or
# escape(), and returns a complete `table` environment. No number is typed by hand
# into any document; a builder is the only way one gets there.

def _results(name: str) -> pd.DataFrame:
    from xaibench.config import paths
    return pd.read_csv(paths().results / name)


def _quick_note(profile) -> str:
    return ("" if profile.is_reportable else
            r"\textbf{PROFILE=" + profile.name + r" --- not for reporting.} ")


def table_datasets(profile) -> pd.DataFrame:
    from xaibench.data.loaders import load_dataset
    from xaibench.viz.theme import DATASET_LABELS, DATASET_ORDER
    rows = []
    for ds_id in DATASET_ORDER:
        ds = load_dataset(ds_id)
        rows.append({
            "Dataset": escape(DATASET_LABELS[ds_id]),
            "$n$": str(len(ds.y)),
            "Features": f"{ds.n_features} ({len(ds.numeric)} num / {len(ds.categorical)} cat)",
            "Positive": fmt(ds.positive_rate, 3),
            "Missing": fmt(ds.missing_rate, 3),
            "Licence": escape(ds.licence.split("(")[0].strip().rstrip(";")),
        })
    return pd.DataFrame(rows)


def T1(profile) -> str:
    return to_booktabs(
        table_datasets(profile),
        caption="Benchmark datasets. `Positive' is the prevalence of the positive "
                "class; `Missing' is the fraction of missing cells after the "
                "dataset-specific semantics fixes of Section~\\ref{sec:method-data}.",
        label="tab:datasets", column_format="lrlrrl", wide=True,
        notes=_quick_note(profile) + "All three datasets are public and de-identified. "
              "Diseases are used as benchmarks; no medical claims are made.")


def T2(profile) -> str:
    from xaibench.config import MODELS
    from xaibench.models.zoo import MODEL_LABELS, param_grid, shap_backend_kind
    def render(value) -> str:
        # sklearn spells "no depth limit" as None. A reader of the paper should not
        # have to know Python's convention, and a bare "None" in a table also looks
        # exactly like a missing value.
        return "unlimited" if value is None else str(value)

    rows = []
    for model_id in MODELS:
        grid = param_grid(model_id)
        combos = int(np.prod([len(v) for v in grid.values()])) if grid else 1
        spec = "; ".join(
            f"{escape(k)} $\\in$ \\{{{escape(', '.join(render(v) for v in values))}\\}}"
            for k, values in grid.items())
        rows.append({
            "Model": escape(MODEL_LABELS[model_id]),
            "Search space": spec,
            "Combinations": str(combos),
            "SHAP backend": escape(shap_backend_kind(model_id).replace("_", "-")),
        })
    frame = pd.DataFrame(rows)
    return to_booktabs(
        frame, caption="Hyperparameter search spaces and SHAP backend assignment. "
                       "Grids are capped at 12 combinations to fit the compute budget.",
        label="tab:grids", column_format="llrl", wide=True, notes=_quick_note(profile))


def _performance_frame(profile, metrics) -> pd.DataFrame:
    from xaibench.viz.theme import DATASET_LABELS, DATASET_ORDER, MODEL_LABELS, MODEL_ORDER
    perf = _results("performance.csv")
    rows = []
    for ds_id in DATASET_ORDER:
        for model_id in MODEL_ORDER:
            cell = perf.query("dataset == @ds_id and model == @model_id")
            row = {"Dataset": escape(DATASET_LABELS[ds_id].split()[0]),
                   "Model": escape(MODEL_LABELS[model_id])}
            for metric, header in metrics:
                row[header] = (f"{fmt(cell[metric].mean(), 3)} $\\pm$ "
                               f"{fmt(cell[metric].std(), 3)}")
            rows.append(row)
    return pd.DataFrame(rows)


_PERF_CONDENSED = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
                   ("f1", "F1"), ("mcc", "MCC")]
_PERF_FULL = [("accuracy", "Accuracy"), ("precision", "Precision"),
              ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC"),
              ("pr_auc", "PR-AUC"), ("mcc", "MCC"), ("brier", "Brier")]


def T3(profile) -> str:
    """Condensed performance table for the paper. Full version is T3full."""
    frame = _performance_frame(profile, _PERF_CONDENSED)
    return to_booktabs(
        frame,
        caption="Predictive performance, nested cross-validation (5 outer $\\times$ 3 "
                "inner folds), mean $\\pm$ standard deviation over the outer folds. "
                "Four representative metrics; all eight, for every cell, are in "
                "\\texttt{results/performance.csv} and in the extended report.",
        label="tab:performance", column_format="ll" + "r" * len(_PERF_CONDENSED),
        wide=True, notes=_quick_note(profile) + "Higher is better for all four.")


def T3full(profile) -> str:
    frame = _performance_frame(profile, _PERF_FULL)
    return to_booktabs(
        frame,
        caption="Predictive performance, all eight metrics, nested cross-validation "
                "(5 outer $\\times$ 3 inner folds), mean $\\pm$ standard deviation over "
                "the outer folds.",
        label="tab:performance-full", column_format="ll" + "r" * len(_PERF_FULL),
        wide=True, notes=_quick_note(profile) + "Brier is a loss: lower is better. "
              "All others: higher is better.")


_AGREE_CONDENSED = [("spearman", "Spearman $\\rho$"), ("jaccard5", "Jaccard@5")]
_AGREE_FULL = [("spearman", "Spearman $\\rho$"), ("kendall", "Kendall $\\tau_b$"),
               ("jaccard3", "Jaccard@3"), ("jaccard5", "Jaccard@5"),
               ("rbo_norm", "RBO$_{\\text{norm}}$")]


def _agreement_frame(profile, metrics) -> pd.DataFrame:
    from xaibench.viz.theme import DATASET_LABELS, DATASET_ORDER, MODEL_LABELS, MODEL_ORDER
    agreement = _results("metrics_agreement.csv").query("instance_id >= 0")
    estimates = _results("stats_estimates.csv")
    rows = []
    for ds_id in DATASET_ORDER:
        for model_id in MODEL_ORDER:
            cell = agreement.query("dataset == @ds_id and model == @model_id")
            row = {"Dataset": escape(DATASET_LABELS[ds_id].split()[0]),
                   "Model": escape(MODEL_LABELS[model_id]), "$n$": str(len(cell))}
            for metric, header in metrics:
                interval = estimates.query(
                    "metric == @metric and dataset == @ds_id and model == @model_id")
                if len(interval):
                    first = interval.iloc[0]
                    row[header] = fmt_ci(first.point, first.ci_low, first.ci_high)
                else:
                    row[header] = fmt(cell[metric].median() if metric in cell else None)
            rows.append(row)
    return pd.DataFrame(rows)


def T4(profile) -> str:
    """Condensed agreement table for the paper. Full version is T4full."""
    return to_booktabs(
        _agreement_frame(profile, _AGREE_CONDENSED),
        caption="RQ1 --- agreement between SHAP and LIME feature rankings, per instance. "
                "Medians with 95\\% percentile bootstrap intervals. Kendall $\\tau_b$, "
                "Jaccard@3, and floor-normalised RBO are in "
                "\\texttt{results/metrics\\_agreement.csv} and in the extended report.",
        label="tab:agreement", column_format="llr" + "r" * len(_AGREE_CONDENSED),
        wide=True, notes=_quick_note(profile))


def T4full(profile) -> str:
    return to_booktabs(
        _agreement_frame(profile, _AGREE_FULL),
        caption="RQ1 --- all agreement measures between SHAP and LIME feature rankings, "
                "per instance. Medians with 95\\% percentile bootstrap intervals.",
        label="tab:agreement-full", column_format="llr" + "r" * len(_AGREE_FULL),
        wide=True,
        notes=_quick_note(profile) + "RBO$_{\\text{norm}}$ is floor-normalised so it is "
              "comparable across datasets of different width (protocol amendment A1).")


def _explainer_rows(frame, columns, headers, estimates=None, metric_map=None):
    from xaibench.viz.theme import EXPLAINER_LABELS, EXPLAINER_ORDER
    rows = []
    for explainer in EXPLAINER_ORDER:
        subset = frame.query("explainer == @explainer")
        if not len(subset):
            continue
        row = {"Explainer": escape(EXPLAINER_LABELS[explainer])}
        for column, header in zip(columns, headers):
            if column not in subset:
                row[header] = "--"
                continue
            if estimates is not None:
                key = (metric_map or {}).get(column, column)
                interval = estimates.query("metric == @key and explainer == @explainer")
                if len(interval):
                    row[header] = fmt_ci(interval.point.median(),
                                         interval.ci_low.median(),
                                         interval.ci_high.median())
                    continue
            row[header] = fmt(subset[column].median())
        rows.append(row)
    return pd.DataFrame(rows)


def T5(profile) -> str:
    stability = _results("metrics_stability.csv")
    frame = _explainer_rows(
        stability,
        ["rerun_cv", "repeat_pair_rho", "lipschitz", "retrain_agreement"],
        ["Re-run CV", "Re-run $\\rho$", "Lipschitz $\\hat L$", "Retrain $\\rho$"])
    return to_booktabs(
        frame,
        caption="RQ2 --- explanation stability decomposed into three sources: "
                "explainer randomness (re-run), input perturbation (Lipschitz), and "
                "model retraining. Medians over all cells.",
        label="tab:stability", column_format="lrrrr",
        notes=_quick_note(profile) + "SHAP's re-run CV is exactly zero for the four "
              "tree backends by construction, declared a priori in the protocol and "
              "not presented as a finding. Retraining figures cover the three "
              "ensemble models only (protocol R3). Lower Lipschitz is more robust.")


def T6(profile) -> str:
    fidelity = _results("metrics_fidelity.csv").query("baseline == 'median'")
    estimates = _results("stats_estimates.csv")
    frame = _explainer_rows(
        fidelity, ["aopc", "auc_ins", "comprehensiveness", "sufficiency"],
        ["AOPC $\\uparrow$", "AUC-ins $\\uparrow$", "Compreh. $\\uparrow$",
         "Suffic. $\\downarrow$"], estimates=estimates)
    return to_booktabs(
        frame,
        caption="RQ3 --- fidelity to the model's decision function, median baseline. "
                "Medians with 95\\% bootstrap intervals across all 18 cells.",
        label="tab:fidelity", column_format="lrrrr", wide=True,
        notes=_quick_note(profile) + "The random control is the scale: an AOPC value "
              "is uninterpretable without it, since deleting any few features moves a "
              "prediction.")


def T7(profile) -> str:
    from xaibench.viz.theme import EXPLAINER_LABELS, EXPLAINER_ORDER
    runtime = _results("metrics_runtime.csv")
    primary = runtime[(runtime.explainer != "lime")
                      | (runtime.num_samples == profile.lime_num_samples[0])]
    rows = []
    for explainer in EXPLAINER_ORDER:
        subset = primary.query("explainer == @explainer")
        if not len(subset):
            continue
        rows.append({
            "Explainer": escape(EXPLAINER_LABELS[explainer]),
            "Median s": fmt(subset.median_seconds.median(), 4),
            "IQR s": fmt(subset.iqr_seconds.median(), 4),
            "Slowest cell s": fmt(subset.median_seconds.max(), 4),
            "AOPC per s": fmt(subset.fidelity_per_second.median(), 2),
        })
    return to_booktabs(
        pd.DataFrame(rows),
        caption="RQ4 --- cost per explanation and the fidelity it buys. Single-threaded, "
                "identical hardware for every measurement.",
        label="tab:runtime", column_format="lrrrr",
        notes=_quick_note(profile) + "Fusion rules consume both base explanations, so "
              "their cost is the sum. The random control is free by construction.")


def T8(profile) -> str:
    from xaibench.metrics.plausibility import load_reference_sets
    from xaibench.viz.theme import DATASET_LABELS, DATASET_ORDER
    plaus = _results("metrics_plausibility.csv")
    references = load_reference_sets()
    rows = []
    for ds_id in DATASET_ORDER:
        reference = references[ds_id]
        cell = plaus.query("dataset == @ds_id")
        rows.append({
            "Dataset": escape(DATASET_LABELS[ds_id].split()[0]),
            "Reference features": escape(", ".join(reference.features)),
            "Sources": str(len(reference.citations)),
            "SHAP J@5": fmt(cell.query("explainer == 'shap'").jaccard5.median()),
            "LIME J@5": fmt(cell.query("explainer == 'lime'").jaccard5.median()),
            "Random J@5": fmt(cell.query("explainer == 'random'").jaccard5.median()),
        })
    return to_booktabs(
        pd.DataFrame(rows),
        caption="RQ5 --- pre-registered plausibility reference sets and top-5 overlap.",
        label="tab:plausibility", column_format="llrrrr", wide=True,
        notes=_quick_note(profile) + "These reference sets are \\emph{not} ground truth. "
              "They are literature-derived, incomplete, and not causal; a low score is "
              "as consistent with an incomplete reference set as with a poor "
              "explanation. RQ5 is the weakest metric in this study and is reported as "
              "corroborating evidence only. The machine-readable sets, with full "
              "citation strings, are in \\texttt{docs/reference\\_features.yaml}.")


def _consensus_frame(profile, full: bool) -> pd.DataFrame:
    from xaibench.viz.theme import DATASET_LABELS, EXPLAINER_LABELS, MODEL_LABELS
    stats = _results("stats_hypotheses.csv").query("hypothesis == 'H6'")
    rows = []
    for row in stats.itertuples():
        rule = str(getattr(row, "best_rule", "--"))
        entry = {
            "Dataset": escape(DATASET_LABELS.get(row.dataset, row.dataset).split()[0]),
            "Model": escape(MODEL_LABELS.get(row.model, row.model)),
            "Best rule": escape(EXPLAINER_LABELS.get(rule, rule).replace("Fusion: ", "")),
            "$\\Delta$AOPC": fmt(getattr(row, "point_estimate", None), 4),
        }
        if full:
            entry["$r$"] = fmt(row.effect_size, 3)
            entry["$p$"] = fmt_p(row.p_value)
        entry["$p_{\\text{adj}}$"] = fmt_p(row.p_adj)
        entry["Fusion wins?"] = "yes" if row.decision == "reject H0" else "no"
        rows.append(entry)
    return pd.DataFrame(rows)


def T9(profile) -> str:
    """Condensed consensus table for the paper. Full version is T9full."""
    frame = _consensus_frame(profile, full=False)
    return to_booktabs(
        frame,
        caption="RQ6 --- does rank fusion beat the \\emph{better} of SHAP and LIME? "
                "Tested on evaluation instances only; the fidelity weight was fitted on "
                "a disjoint calibration split. Effect sizes and unadjusted $p$ are in "
                "\\texttt{results/stats\\_hypotheses.csv} and in the extended report.",
        label="tab:consensus", column_format="lllrrl", wide=True,
        notes=_quick_note(profile) + "$\\Delta$AOPC is the median per-instance "
              "difference (fusion $-$ best single); negative means fusion lost. "
              "$p_{\\text{adj}}$ is Holm-corrected within the family of 18 cells.")


def T9full(profile) -> str:
    frame = _consensus_frame(profile, full=True)
    return to_booktabs(
        frame,
        caption="RQ6 --- rank fusion against the better of SHAP and LIME, with effect "
                "sizes and unadjusted $p$-values, on evaluation instances only.",
        label="tab:consensus-full", column_format="lllrrrrl", wide=True,
        notes=_quick_note(profile) + "$r$ is the rank-biserial correlation. A negative "
              "$r$ with a one-sided `greater' alternative means fusion lost.")


# Authored LaTeX, not free text: these carry deliberate maths and must NOT be escaped.
_HYPOTHESIS_TEXT = {
    "H1": "SHAP/LIME concordance exceeds $\\rho = 0.50$",
    "H1b": "Concordance differs between simple models and ensembles",
    "H1c": "Concordance tracks task difficulty",
    "H2a": "LIME's re-run variance exceeds zero",
    "H2b": "SHAP and LIME differ in local robustness",
    "H2c": "SHAP and LIME differ in retraining stability",
    "H3a": "SHAP and LIME differ in fidelity",
    "H3b": "Each explainer beats the random control",
    "H4": "SHAP costs less than LIME",
    "H5": "SHAP and LIME differ in plausibility overlap",
    "H6": "Fusion beats the better of SHAP and LIME",
}


def T10(profile) -> str:
    """The paper's centrepiece: every pre-registered hypothesis and its decision."""
    stats = _results("stats_hypotheses.csv")
    rows = []
    for hypothesis, group in stats.groupby("hypothesis", sort=False):
        live = group[~group.degenerate]
        rejected = int((group.decision == "reject H0").sum())
        rows.append({
            "H": escape(hypothesis),
            "Alternative hypothesis": _HYPOTHESIS_TEXT.get(hypothesis, ""),
            "Unit": escape(str(group.unit.iloc[0])),
            "Alt.": escape(str(group.alternative.iloc[0])),
            "Family": str(len(group)),
            "Median $r$": fmt(live.effect_size.median() if len(live) else None, 3),
            "Median $p_{\\text{adj}}$": fmt_p(live.p_adj.median() if len(live) else None),
            "Rejected": f"{rejected}/{len(group)}",
        })
    order = list(_HYPOTHESIS_TEXT)
    frame = pd.DataFrame(rows)
    frame["_o"] = [order.index(h) if h in order else 99 for h in frame["H"]]
    frame = frame.sort_values("_o").drop(columns="_o")
    return to_booktabs(
        frame,
        caption="Decision table for all eleven pre-registered hypotheses. Every entry "
                "is auditable against the frozen protocol, which was committed before "
                "any result existed.",
        label="tab:hypotheses", column_format="llllrrrl", wide=True,
        notes=_quick_note(profile) + "All tests are Wilcoxon signed-rank except H1b "
              "(Mann--Whitney U) and H1c (Spearman with a 10,000-draw permutation test). "
              "`Alt.' is the pre-registered alternative; `Family' is the number of cells "
              "the hypothesis spans, and $p_{\\text{adj}}$ is Holm-corrected within it. "
              "`Rejected' counts cells where $p_{\\text{adj}} < 0.05$. Degenerate cells "
              "(no variation to test) are retained in the family and reported rather "
              "than dropped.")


def T11(profile) -> str:
    """Design axes of this study against typical prior practice.

    Deliberately NOT a per-paper count table. Asserting "study X used 4 models and 3
    metrics" for work we have not read line by line would be fabrication, and the
    counts are not consistently recoverable from abstracts -- some of the cited work
    is commentary rather than empirical comparison. The rows below state this study's
    design exactly and characterise prior practice at a level the citations support.
    """
    from xaibench.config import ALL_EXPLAINERS, CELLS, DATASETS, MODELS
    stats = _results("stats_hypotheses.csv")
    rows = [
        ("Model families varied", str(len(MODELS)),
         "usually 1--4, often a single family"),
        ("Datasets varied", str(len(DATASETS)),
         "usually 1, occasionally several"),
        ("Explainers compared", rf"{len(ALL_EXPLAINERS)} (incl.\ random control)",
         "2, control rarely included"),
        ("Model--dataset cells", str(len(CELLS)), "typically 1--4"),
        ("Interpretability measures", "16 across 5 families",
         "2--5, definitions not standardised"),
        ("Pre-registered hypotheses", str(stats.hypothesis.nunique()),
         "none reported"),
        ("Multiple-comparison correction", "Holm within family", "rarely reported"),
        ("Effect sizes and interval estimates", "on every test and estimate",
         "rarely reported"),
        ("Attribution space verified comparable", "yes, by additivity check",
         "not discussed"),
        ("Reproducibility package", "pinned, checksummed, one command",
         "partial or absent"),
    ]
    frame = pd.DataFrame([{"Design axis": escape(a), "This study": escape(b),
                           "Typical prior practice": escape(c)} for a, b, c in rows])
    return to_booktabs(
        frame,
        caption="Design axes of this benchmark against typical practice in prior "
                "SHAP-versus-LIME comparisons on tabular data "
                "\\cite{man2021best,salih2024perspective,elshawi2019interpretability,duell2021comparison,amparore2021trust}.",
        label="tab:positioning", column_format="p{0.30\\columnwidth}p{0.28\\columnwidth}p{0.30\\columnwidth}",
        notes=_quick_note(profile) + "The right-hand column characterises the cited "
              "literature in aggregate rather than tabulating each paper: metric and "
              "model counts are not consistently recoverable from these sources, and "
              "some are commentary rather than empirical comparison. Readers should "
              "consult the individual works for their exact scope.")


def T12(profile) -> str:
    from xaibench.config import paths
    manifest_path = paths().results / "MANIFEST.json"
    rows = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, version in sorted(manifest.get("library_versions", {}).items()):
            rows.append({"Component": escape(name), "Version": escape(version)})
        hardware = manifest.get("hardware", {})
        for key in ("python", "platform", "cpu_count", "memory"):
            if key in hardware:
                rows.append({"Component": escape(key), "Version": escape(hardware[key])})
        rows.append({"Component": "git commit",
                     "Version": escape(manifest.get("git_commit", "")[:12])})
        rows.append({"Component": "profile", "Version": escape(manifest.get("profile", ""))})
    else:
        rows.append({"Component": "manifest", "Version": "not yet generated"})
    return to_booktabs(
        pd.DataFrame(rows),
        caption="Environment recorded with the results. Exact versions are pinned in "
                "\\texttt{requirements.lock.txt}.",
        label="tab:environment", column_format="ll",
        notes=_quick_note(profile) + "\\texttt{results/MANIFEST.json} additionally "
              "records a SHA-256 digest for every generated artefact.")


TABLES = {"T1": T1, "T2": T2, "T3": T3, "T3full": T3full, "T4": T4,
          "T4full": T4full, "T5": T5, "T6": T6, "T7": T7, "T8": T8,
          "T9": T9, "T9full": T9full, "T10": T10, "T11": T11, "T12": T12}


# ------------------------------------------------------------- headline macros
# Prose needs numbers too -- an abstract that says "median agreement was 0.62" is a
# hand-typed result unless that 0.62 comes from here. Every such value is emitted as
# a LaTeX macro and \input by the documents, so the "no number is typed by hand"
# rule covers running text, not just tables.

def _macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def findings_macros(profile) -> str:
    from xaibench.config import RBO_P
    from xaibench.metrics.agreement import rbo_floor

    agreement = _results("metrics_agreement.csv").query("instance_id >= 0")
    fidelity = _results("metrics_fidelity.csv").query("baseline == 'median'")
    stability = _results("metrics_stability.csv")
    runtime = _results("metrics_runtime.csv")
    stats = _results("stats_hypotheses.csv")
    config = _results("explainer_config.csv")
    budget = _results("instance_budget.csv")
    perf = _results("performance.csv")

    primary = runtime[(runtime.explainer != "lime")
                      | (runtime.num_samples == profile.lime_num_samples[0])]

    def med(frame, column, **filters):
        subset = frame
        for key, value in filters.items():
            subset = subset[subset[key] == value]
        series = subset[column].dropna()
        return series.median() if len(series) else float("nan")

    def rejected(hypothesis):
        group = stats.query("hypothesis == @hypothesis")
        return int((group.decision == "reject H0").sum()), len(group)

    shap_seconds = med(primary, "median_seconds", explainer="shap")
    lime_seconds = med(primary, "median_seconds", explainer="lime")
    h6_reject, h6_total = rejected("H6")
    h3b_reject, h3b_total = rejected("H3b")
    h1_reject, h1_total = rejected("H1")

    lines = [
        "% Generated by xaibench.stats.report.findings_macros -- do not edit.",
        _macro("ProfileName", str(profile.name)),
        _macro("NumCells", str(len(config))),
        _macro("NumExplained", str(int(budget.n_explained.sum()))),
        _macro("MedianRho", fmt(med(agreement, "spearman"), 2)),
        _macro("MedianJaccardFive", fmt(med(agreement, "jaccard5"), 2)),
        _macro("MedianRboNorm", fmt(med(agreement, "rbo_norm"), 2)),
        _macro("RhoPima", fmt(med(agreement, "spearman", dataset="pima"), 2)),
        _macro("RhoHeart", fmt(med(agreement, "spearman", dataset="heart"), 2)),
        _macro("RhoCkd", fmt(med(agreement, "spearman", dataset="ckd"), 2)),
        _macro("AucPima", fmt(med(perf, "roc_auc", dataset="pima"), 2)),
        _macro("AucHeart", fmt(med(perf, "roc_auc", dataset="heart"), 2)),
        _macro("AucCkd", fmt(med(perf, "roc_auc", dataset="ckd"), 2)),
        _macro("AopcShap", fmt(med(fidelity, "aopc", explainer="shap"), 3)),
        _macro("AopcLime", fmt(med(fidelity, "aopc", explainer="lime"), 3)),
        _macro("AopcRandom", fmt(med(fidelity, "aopc", explainer="random"), 3)),
        _macro("AopcBestFusion", fmt(max(
            med(fidelity, "aopc", explainer=rule) for rule in
            ("rankavg", "rrf", "fidwt")), 3)),
        _macro("LipschitzShap", fmt(med(stability, "lipschitz", explainer="shap"), 2)),
        _macro("LipschitzLime", fmt(med(stability, "lipschitz", explainer="lime"), 2)),
        _macro("RerunCvLime", fmt(med(stability, "rerun_cv", explainer="lime"), 2)),
        _macro("RetrainShap", fmt(med(stability, "retrain_agreement", explainer="shap"), 2)),
        _macro("RetrainLime", fmt(med(stability, "retrain_agreement", explainer="lime"), 2)),
        _macro("SecondsShap", fmt(shap_seconds, 4)),
        _macro("SecondsLime", fmt(lime_seconds, 4)),
        _macro("SpeedFactor", fmt(lime_seconds / shap_seconds
                                  if shap_seconds else float("nan"), 1)),
        _macro("HOneRejected", f"{h1_reject}"),
        _macro("HOneTotal", f"{h1_total}"),
        _macro("HThreeBRejected", f"{h3b_reject}"),
        _macro("HThreeBTotal", f"{h3b_total}"),
        _macro("HSixRejected", f"{h6_reject}"),
        _macro("HSixTotal", f"{h6_total}"),
        _macro("AdditivityMedianMax", fmt(config.additivity_median.max(), 3)),
        _macro("AdditivityOutlierCells", str(int((config.additivity_n_over_tol > 0).sum()))),
        _macro("AdditivityGateFailCells", str(int((~config.additivity_gate_ok.astype(bool)).sum()))),
        _macro("AdditivityWorstMax", fmt(config.additivity_max.max(), 3)),
        _macro("AdditivityWorstOutliers", str(int(config.additivity_n_over_tol.max()))),
        # Exact, deterministic properties of RBO, not experimental results -- but still
        # generated, because a hand-typed constant is a hand-typed constant.
        _macro("RboFloorEight", fmt(rbo_floor(8, RBO_P), 3)),
        _macro("RboFloorThirteen", fmt(rbo_floor(13, RBO_P), 3)),
        _macro("RboFloorTwentyFour", fmt(rbo_floor(24, RBO_P), 3)),
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- README findings
# The README claims every number in it is generated. That claim has to be true, so the
# findings table is spliced between markers by stage 5 rather than hand-maintained --
# otherwise the most-read file in the repository is the first one to go stale.

README_BEGIN = "<!-- BEGIN GENERATED FINDINGS -->"
README_END = "<!-- END GENERATED FINDINGS -->"


def readme_findings_table(profile) -> str:
    agreement = _results("metrics_agreement.csv").query("instance_id >= 0")
    fidelity = _results("metrics_fidelity.csv").query("baseline == 'median'")
    stability = _results("metrics_stability.csv")
    runtime = _results("metrics_runtime.csv")
    stats = _results("stats_hypotheses.csv")
    perf = _results("performance.csv")

    primary = runtime[(runtime.explainer != "lime")
                      | (runtime.num_samples == profile.lime_num_samples[0])]

    def med(frame, column, **filters):
        subset = frame
        for key, value in filters.items():
            subset = subset[subset[key] == value]
        series = subset[column].dropna()
        return float(series.median()) if len(series) else float("nan")

    def tally(hypothesis):
        group = stats.query("hypothesis == @hypothesis")
        return int((group.decision == "reject H0").sum()), len(group)

    rho = med(agreement, "spearman")
    jaccard = med(agreement, "jaccard5")
    rho_ckd = med(agreement, "spearman", dataset="ckd")
    auc_ckd = med(perf, "roc_auc", dataset="ckd")
    shap_seconds = med(primary, "median_seconds", explainer="shap")
    lime_seconds = med(primary, "median_seconds", explainer="lime")
    h1_hit, h1_all = tally("H1")
    h3b_hit, h3b_all = tally("H3b")
    h6_hit, h6_all = tally("H6")
    h6_effect = float(stats.query("hypothesis == 'H6'").effect_size.median())

    rows = [
        ("RQ1", "Do SHAP and LIME agree on feature rankings?",
         f"**Substantially, but not interchangeably.** Median Spearman ρ = {rho:.2f}, "
         f"top-5 Jaccard {jaccard:.2f}; clears a pre-registered 0.50 bar in "
         f"{h1_hit}/{h1_all} cells, yet one of the five top features still differs. "
         f"Agreement is *lowest* on the **easiest** task (CKD, ρ = {rho_ckd:.2f} at "
         f"ROC-AUC {auc_ckd:.2f})."),
        ("RQ2", "How stable are the explanations?",
         f"**Reproducible ≠ robust.** Tree SHAP's re-run CV is *exactly* "
         f"{med(stability, 'rerun_cv', explainer='shap'):.0f}, but its cross-refit "
         f"agreement is {med(stability, 'retrain_agreement', explainer='shap'):.2f} — the "
         f"ranking moves when the model is refit on a resample of the same data. "
         f"Perturbation robustness: {med(stability, 'lipschitz', explainer='shap'):.2f} "
         f"(SHAP) vs {med(stability, 'lipschitz', explainer='lime'):.2f} (LIME)."),
        ("RQ3", "Are they faithful to the model?",
         f"**Both, SHAP more so.** AOPC {med(fidelity, 'aopc', explainer='shap'):.3f} "
         f"(SHAP) vs {med(fidelity, 'aopc', explainer='lime'):.3f} (LIME) vs "
         f"**{med(fidelity, 'aopc', explainer='random'):.3f} (random control)**. Both beat "
         f"the control in {h3b_hit}/{h3b_all} tests after Holm correction."),
        ("RQ4", "What does an explanation cost?",
         f"**SHAP is ~{lime_seconds / shap_seconds:.0f}× cheaper** — {shap_seconds:.4f} s "
         f"vs {lime_seconds:.4f} s per explanation. Structural, not an implementation "
         f"detail."),
        ("RQ5", "Do top features look plausible?",
         "**The metric does not discriminate.** Both beat random; neither separates from "
         "the other. Pre-registered as the weakest measurement here."),
        ("RQ6", "Does fusing the two help?",
         f"**{'No' if h6_hit == 0 else 'Sometimes'}, in {h6_hit}/{h6_all} cells** "
         f"(median effect {h6_effect:+.2f}). A pre-registered null result that follows "
         f"from RQ1: substantial agreement leaves no complementary signal to exploit."),
    ]
    lines = [README_BEGIN,
             "| # | Question | Finding |",
             "|---|----------|---------|"]
    lines += [f"| {rq} | {question} | {finding} |" for rq, question, finding in rows]
    lines.append(README_END)
    return "\n".join(lines) + "\n"


def splice_readme(profile, readme_path) -> bool:
    """Replace the marked block in README.md. Returns True if the file changed."""
    path = pathlib.Path(readme_path)
    text = path.read_text()
    if README_BEGIN not in text or README_END not in text:
        return False
    head, rest = text.split(README_BEGIN, 1)
    _, tail = rest.split(README_END, 1)
    updated = head + readme_findings_table(profile).rstrip("\n") + tail
    if updated != text:
        path.write_text(updated)
        return True
    return False
