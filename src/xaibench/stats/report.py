"""Tidy frames to booktabs LaTeX. The ONLY path by which a number reaches a document."""
from __future__ import annotations

import json

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
                escape_cells: bool = False) -> str:
    """A complete table environment.

    Contract: **cells arrive LaTeX-ready.** Table builders call `escape()` on text
    columns and `fmt`/`fmt_ci`/`fmt_p` on numbers, then pass them through untouched.
    That is the only workable split, because a marker type cannot survive a round
    trip through a DataFrame -- pandas coerces a str subclass back to plain str -- so
    "$<$0.001" from fmt_p would be double-escaped by an escaping-by-default table
    writer. Set escape_cells=True only for a frame of raw, unformatted text.
    """
    columns = list(df.columns)
    if column_format is None:
        column_format = "l" + "r" * (len(columns) - 1)
    header = " & ".join(escape(c) for c in columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = [escape(v) if escape_cells else str(v) for v in row.tolist()]
        body_rows.append(" & ".join(cells) + r" \\")
    parts = [
        rf"\begin{{table}}[{placement}]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small" if small else "",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if notes:
        parts.append(rf"\par\vspace{{2pt}}\footnotesize {notes}")
    parts.append(r"\end{table}")
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
        label="tab:datasets", column_format="lrlrrl",
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
        label="tab:grids", column_format="llrl", notes=_quick_note(profile))


def T3(profile) -> str:
    from xaibench.viz.theme import DATASET_LABELS, DATASET_ORDER, MODEL_LABELS, MODEL_ORDER
    perf = _results("performance.csv")
    metrics = ["accuracy", "f1", "roc_auc", "pr_auc", "mcc", "brier"]
    rows = []
    for ds_id in DATASET_ORDER:
        for model_id in MODEL_ORDER:
            cell = perf.query("dataset == @ds_id and model == @model_id")
            row = {"Dataset": escape(DATASET_LABELS[ds_id].split()[0]),
                   "Model": escape(MODEL_LABELS[model_id])}
            for metric in metrics:
                row[metric.replace("_", "-").upper()] = (
                    f"{fmt(cell[metric].mean(), 3)} $\\pm$ {fmt(cell[metric].std(), 3)}")
            rows.append(row)
    return to_booktabs(
        pd.DataFrame(rows),
        caption="Predictive performance, nested cross-validation (5 outer $\\times$ 3 "
                "inner folds), mean $\\pm$ standard deviation over the outer folds.",
        label="tab:performance", column_format="ll" + "r" * len(metrics),
        notes=_quick_note(profile) + "Brier is a loss: lower is better. All others: "
              "higher is better.")


def T4(profile) -> str:
    from xaibench.viz.theme import DATASET_LABELS, DATASET_ORDER, MODEL_LABELS, MODEL_ORDER
    agreement = _results("metrics_agreement.csv").query("instance_id >= 0")
    estimates = _results("stats_estimates.csv")
    rows = []
    for ds_id in DATASET_ORDER:
        for model_id in MODEL_ORDER:
            cell = agreement.query("dataset == @ds_id and model == @model_id")
            row = {"Dataset": escape(DATASET_LABELS[ds_id].split()[0]),
                   "Model": escape(MODEL_LABELS[model_id]), "$n$": str(len(cell))}
            for metric, header in (("spearman", "Spearman $\\rho$"),
                                   ("kendall", "Kendall $\\tau_b$"),
                                   ("jaccard5", "Jaccard@5"),
                                   ("rbo_norm", "RBO$_{\\text{norm}}$")):
                interval = estimates.query(
                    "metric == @metric and dataset == @ds_id and model == @model_id")
                if len(interval):
                    first = interval.iloc[0]
                    row[header] = fmt_ci(first.point, first.ci_low, first.ci_high)
                else:
                    row[header] = fmt(cell[metric].median() if metric in cell else None)
            rows.append(row)
    return to_booktabs(
        pd.DataFrame(rows),
        caption="RQ1 --- agreement between SHAP and LIME feature rankings, per "
                "instance. Point estimates are medians with 95\\% percentile "
                "bootstrap intervals.",
        label="tab:agreement", column_format="llrrrrr",
        notes=_quick_note(profile) + "RBO$_{\\text{norm}}$ is floor-normalised so it "
              "is comparable across datasets of different width (protocol amendment A1).")


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
        label="tab:fidelity", column_format="lrrrr",
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
        label="tab:plausibility", column_format="llrrrr",
        notes=_quick_note(profile) + "These reference sets are \\emph{not} ground truth. "
              "They are literature-derived, incomplete, and not causal; a low score is "
              "as consistent with an incomplete reference set as with a poor "
              "explanation. RQ5 is the weakest metric in this study and is reported as "
              "corroborating evidence only. Full citations in Appendix~\\ref{app:refsets}.")


def T9(profile) -> str:
    from xaibench.viz.theme import DATASET_LABELS, MODEL_LABELS
    stats = _results("stats_hypotheses.csv").query("hypothesis == 'H6'")
    rows = []
    for row in stats.itertuples():
        rows.append({
            "Dataset": escape(DATASET_LABELS.get(row.dataset, row.dataset).split()[0]),
            "Model": escape(MODEL_LABELS.get(row.model, row.model)),
            "Best rule": escape(str(getattr(row, "best_rule", "--"))),
            "$\\Delta$AOPC": fmt(getattr(row, "point_estimate", None), 4),
            "$r$": fmt(row.effect_size, 3),
            "$p$": fmt_p(row.p_value),
            "$p_{\\text{adj}}$": fmt_p(row.p_adj),
            "Decision": escape(row.decision),
        })
    return to_booktabs(
        pd.DataFrame(rows),
        caption="RQ6 --- does rank fusion beat the \\emph{better} of SHAP and LIME? "
                "Tested on evaluation instances only; the fidelity weight was fitted "
                "on a disjoint calibration split.",
        label="tab:consensus", column_format="lllrrrrl",
        notes=_quick_note(profile) + "$\\Delta$AOPC is the median per-instance "
              "difference (fusion $-$ best single). $r$ is the rank-biserial "
              "correlation. $p_{\\text{adj}}$ is Holm-corrected within the family of "
              "18 cells.")


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
            "Alternative hypothesis": escape(_HYPOTHESIS_TEXT.get(hypothesis, "")),
            "Unit": escape(str(group.unit.iloc[0])),
            "Test": escape(str(group.test.iloc[0])),
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
        label="tab:hypotheses", column_format="llllrrrl",
        notes=_quick_note(profile) + "`Family' is the number of cells the hypothesis "
              "spans; $p_{\\text{adj}}$ is Holm-corrected within that family. "
              "`Rejected' counts cells where $p_{\\text{adj}} < 0.05$. Degenerate "
              "cells (no variation to test) are retained in the family and reported "
              "rather than dropped.")


def T11(profile) -> str:
    """Positioning against prior SHAP-vs-LIME comparisons."""
    rows = [
        ("This study", "6", "3", "16", "Yes (11, Holm)", "Yes"),
        ("Man \\& Chan~\\cite{man2021best}", "4", "2", "3", "No", "Partial"),
        ("Salih et al.~\\cite{salih2024perspective}", "2", "1", "2", "No", "No"),
        ("ElShawi et al.~\\cite{elshawi2019interpretability}", "3", "1", "5", "No", "No"),
        ("Duell et al.~\\cite{duell2021comparison}", "3", "1", "3", "No", "No"),
        ("Vieira \\& Digiampietri~\\cite{vieira2020study}", "2", "1", "2", "No", "No"),
        ("Amparore et al.~\\cite{amparore2021trust}", "3", "4", "4", "Partial", "Yes"),
    ]
    frame = pd.DataFrame(
        [{"Study": s, "Models": m, "Datasets": d, "Metrics": k,
          "Statistical testing": t, "Code released": c} for s, m, d, k, t, c in rows])
    return to_booktabs(
        frame, caption="This study against prior SHAP-versus-LIME comparisons on "
                       "tabular data.",
        label="tab:positioning", column_format="lrrrll",
        notes=_quick_note(profile) + "Metric counts are as reported by each study; "
              "definitions are not standardised across this literature, which is part "
              "of the gap this benchmark addresses.")


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


TABLES = {"T1": T1, "T2": T2, "T3": T3, "T4": T4, "T5": T5, "T6": T6,
          "T7": T7, "T8": T8, "T9": T9, "T10": T10, "T11": T11, "T12": T12}
