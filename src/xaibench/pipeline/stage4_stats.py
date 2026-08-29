"""Stage 4: the eleven pre-registered hypotheses, Holm-corrected within family.

Every test, alternative, and unit of analysis is copied from docs/PROTOCOL.md §6
rather than re-derived here. Degenerate rows (no variation to test) are recorded so
the decision table has no holes, but excluded from the Holm divisor: correcting a
test that was never performed only discards power.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats as sps

from xaibench.config import (
    ALPHA,
    CELLS,
    CONSENSUS_EXPLAINERS,
    ENSEMBLE_MODELS,
    SEED,
    SIMPLE_MODELS,
    Profile,
    get_profile,
    paths,
)
from xaibench.stats.tests import (
    holm,
    mannwhitney_with_effect,
    percentile_ci,
    wilcoxon_with_effect,
)

AGREEMENT_THRESHOLD = 0.50      # NORMATIVE: H1's pre-registered practical bar


def _row(hypothesis, dataset, model, unit, test, result, extra=None):
    """The reported alternative comes off the result, never from a separate argument.

    Taking it as a display string let a one-sided label sit beside a two-sided call:
    H6 was labelled `greater` while running two-sided, so a significant result in the
    WRONG direction read as `reject H0` and inverted the headline conclusion. Deriving
    the label from the test that actually ran makes that mismatch impossible.
    """
    row = {
        "hypothesis": hypothesis, "dataset": dataset, "model": model, "unit": unit,
        "test": test, "alternative": result.alternative, "n": result.n,
        "statistic": result.statistic, "p_value": result.p_value,
        "effect_size": result.effect_size, "effect_name": result.effect_name,
        "median_diff": result.median_diff, "degenerate": bool(result.degenerate),
    }
    row.update(extra or {})
    return row


def _load(profile: Profile) -> dict[str, pd.DataFrame]:
    p = paths()
    data = {name: pd.read_csv(p.results / f"metrics_{name}.csv")
            for name in ("agreement", "fidelity", "plausibility", "runtime", "stability")}
    data["performance"] = pd.read_csv(p.results / "performance.csv")
    data["fidelity_median"] = data["fidelity"].query("baseline == 'median'")
    data["agreement_local"] = data["agreement"].query("instance_id >= 0")
    return data


def _paired(frame, value, left, right, key="explainer", index="instance_id"):
    """Two aligned vectors from a long frame, on the instances present in both."""
    wide = frame.pivot_table(index=index, columns=key, values=value, aggfunc="first")
    if left not in wide or right not in wide:
        return np.array([]), np.array([])
    both = wide[[left, right]].dropna()
    return both[left].to_numpy(float), both[right].to_numpy(float)


def run_all_hypotheses(profile: Profile) -> pd.DataFrame:
    data = _load(profile)
    rows: list[dict] = []

    # H1 -- local SHAP/LIME concordance exceeds the practical bar of 0.50.
    for dataset_id, model_id in CELLS:
        rho = data["agreement_local"].query(
            "dataset == @dataset_id and model == @model_id").spearman.to_numpy(float)
        rows.append(_row("H1", dataset_id, model_id, "instance",
                         "wilcoxon(rho - 0.50)",
                         wilcoxon_with_effect(rho, mu=AGREEMENT_THRESHOLD,
                                              alternative="greater"),
                         {"point_estimate": float(np.median(rho)) if rho.size else np.nan}))

    cell_rho = (data["agreement_local"].groupby(["dataset", "model"], as_index=False)
                .spearman.median())

    # H1b -- does concordance differ between simple models and ensembles?
    simple = cell_rho[cell_rho.model.isin(SIMPLE_MODELS)].spearman.to_numpy(float)
    ensemble = cell_rho[~cell_rho.model.isin(SIMPLE_MODELS)].spearman.to_numpy(float)
    rows.append(_row("H1b", "all", "all", "cell", "mannwhitneyu",
                     mannwhitney_with_effect(simple, ensemble, alternative="two-sided"),
                     {"point_estimate": float(np.median(simple) - np.median(ensemble))}))

    # H1c -- does concordance track task difficulty (1 - ROC-AUC)?
    auc = (data["performance"].groupby(["dataset", "model"], as_index=False)
           .roc_auc.mean())
    merged = cell_rho.merge(auc, on=["dataset", "model"])
    difficulty = 1.0 - merged.roc_auc.to_numpy(float)
    observed = merged.spearman.to_numpy(float)
    rho_stat = sps.spearmanr(observed, difficulty).statistic if merged.shape[0] > 2 else np.nan
    rng = np.random.default_rng(SEED)
    permuted = np.array([sps.spearmanr(rng.permutation(observed), difficulty).statistic
                         for _ in range(10_000)])
    p_perm = float((np.abs(permuted) >= abs(rho_stat)).mean()) if np.isfinite(rho_stat) else 1.0
    rows.append({
        "hypothesis": "H1c", "dataset": "all", "model": "all", "unit": "cell",
        "test": "spearman + permutation(10000)", "alternative": "two-sided",
        "n": int(merged.shape[0]), "statistic": float(rho_stat), "p_value": p_perm,
        "effect_size": float(rho_stat), "effect_name": "spearman_rho",
        "median_diff": np.nan, "degenerate": not np.isfinite(rho_stat),
        "point_estimate": float(rho_stat),
    })

    # H2a -- LIME's re-run variance is greater than zero. Tree SHAP is deterministic
    # a priori (protocol §5.2), so it is a recorded measurement, not a test.
    for dataset_id, model_id in CELLS:
        cv = data["stability"].query(
            "dataset == @dataset_id and model == @model_id and explainer == 'lime'"
        ).rerun_cv.dropna().to_numpy(float)
        rows.append(_row("H2a", dataset_id, model_id, "instance",
                         "wilcoxon(cv > 0)",
                         wilcoxon_with_effect(cv, alternative="greater"),
                         {"point_estimate": float(np.median(cv)) if cv.size else np.nan}))

    # H2b -- local robustness: SHAP versus LIME, paired on the instance.
    for dataset_id, model_id in CELLS:
        cell = data["stability"].query(
            "dataset == @dataset_id and model == @model_id").dropna(subset=["lipschitz"])
        a, b = _paired(cell, "lipschitz", "shap", "lime")
        rows.append(_row("H2b", dataset_id, model_id, "instance",
                         "wilcoxon paired",
                         wilcoxon_with_effect(a, b, alternative="two-sided"),
                         {"point_estimate": float(np.median(a) - np.median(b))
                          if a.size else np.nan}))

    # H2c -- retraining stability, ensembles only (protocol R3).
    retrain = data["stability"].dropna(subset=["retrain_agreement"])
    for dataset_id in sorted(retrain.dataset.unique()):
        for model_id in ENSEMBLE_MODELS:
            cell = retrain.query("dataset == @dataset_id and model == @model_id")
            a, b = _paired(cell, "retrain_agreement", "shap", "lime")
            rows.append(_row("H2c", dataset_id, model_id, "cell",
                             "wilcoxon paired",
                             wilcoxon_with_effect(a, b, alternative="two-sided"),
                             {"point_estimate": float(np.mean(a) - np.mean(b))
                              if a.size else np.nan}))

    # H3a -- fidelity: SHAP versus LIME.
    for dataset_id, model_id in CELLS:
        cell = data["fidelity_median"].query(
            "dataset == @dataset_id and model == @model_id")
        a, b = _paired(cell, "aopc", "shap", "lime")
        rows.append(_row("H3a", dataset_id, model_id, "instance",
                         "wilcoxon paired",
                         wilcoxon_with_effect(a, b, alternative="two-sided"),
                         {"point_estimate": float(np.median(a) - np.median(b))
                          if a.size else np.nan}))

    # H3b -- each explainer must beat the random-ranking control.
    for explainer in ("shap", "lime"):
        for dataset_id, model_id in CELLS:
            cell = data["fidelity_median"].query(
                "dataset == @dataset_id and model == @model_id")
            a, b = _paired(cell, "aopc", explainer, "random")
            rows.append(_row("H3b", dataset_id, model_id, "instance",
                             "wilcoxon paired",
                             wilcoxon_with_effect(a, b, alternative="greater"),
                             {"explainer": explainer,
                              "point_estimate": float(np.median(a) - np.median(b))
                              if a.size else np.nan}))

    # H4 -- cost: SHAP is cheaper than LIME.
    attribution_runtime = _runtime_per_instance()
    for dataset_id, model_id in CELLS:
        cell = attribution_runtime.query(
            "dataset == @dataset_id and model == @model_id")
        a, b = _paired(cell, "runtime_s", "shap", "lime")
        rows.append(_row("H4", dataset_id, model_id, "instance",
                         "wilcoxon paired",
                         wilcoxon_with_effect(a, b, alternative="less"),
                         {"point_estimate": float(np.median(a) - np.median(b))
                          if a.size else np.nan}))

    # H5 -- plausibility, exploratory: the unit is the cell, so n = 18.
    cell_plaus = (data["plausibility"].groupby(["dataset", "model", "explainer"],
                                               as_index=False).jaccard5.median())
    a, b = _paired(cell_plaus, "jaccard5", "shap", "lime",
                   index=["dataset", "model"])
    rows.append(_row("H5", "all", "all", "cell (exploratory)",
                     "wilcoxon paired",
                     wilcoxon_with_effect(a, b, alternative="two-sided"),
                     {"point_estimate": float(np.median(a) - np.median(b))
                      if a.size else np.nan}))

    # H6 -- does the best fusion rule beat the BETTER of SHAP and LIME? Evaluation
    # instances only: the fidelity weight was fitted on the calibration split.
    for dataset_id, model_id in CELLS:
        cell = data["fidelity_median"].query(
            "dataset == @dataset_id and model == @model_id and is_evaluation")
        wide = cell.pivot_table(index="instance_id", columns="explainer",
                                values="aopc", aggfunc="first")
        available = [c for c in CONSENSUS_EXPLAINERS if c in wide]
        if not available or "shap" not in wide or "lime" not in wide:
            rows.append(_row("H6", dataset_id, model_id, "evaluation instance",
                             "wilcoxon paired",
                             wilcoxon_with_effect(np.zeros(1), alternative="greater"),
                             {"best_rule": "none"}))
            continue
        best_rule = wide[available].median().idxmax()
        fused = wide[best_rule].to_numpy(float)
        single = wide[["shap", "lime"]].max(axis=1).to_numpy(float)
        rows.append(_row("H6", dataset_id, model_id, "evaluation instance",
                         "wilcoxon paired",
                         wilcoxon_with_effect(fused, single, alternative="greater"),
                         {"best_rule": best_rule,
                          "point_estimate": float(np.median(fused - single))}))

    frame = pd.DataFrame(rows)

    # Holm within family, over the tests actually performed.
    frame["p_adj"] = 1.0
    for hypothesis, group in frame.groupby("hypothesis"):
        live = group.index[~group.degenerate]
        if len(live):
            frame.loc[live, "p_adj"] = holm(frame.loc[live, "p_value"].to_numpy(float))
    frame["decision"] = np.where(frame.p_adj < ALPHA, "reject H0", "fail to reject H0")
    frame["family_size"] = frame.groupby("hypothesis").hypothesis.transform("size")
    return frame


def _runtime_per_instance() -> pd.DataFrame:
    p = paths()
    frames = []
    for dataset_id, model_id in CELLS:
        frame = pd.read_parquet(p.attributions / f"{dataset_id}_{model_id}_attr.parquet")
        frames.append(frame.groupby(["dataset", "model", "explainer", "instance_id"],
                                    as_index=False).runtime_s.first())
    return pd.concat(frames, ignore_index=True)


_ESTIMATE_SPECS = [
    ("agreement", ["spearman", "kendall", "jaccard3", "jaccard5", "rbo", "rbo_norm"], None),
    ("fidelity_median", ["aopc", "auc_ins", "comprehensiveness", "sufficiency"], "explainer"),
    ("plausibility", ["jaccard5", "precision5"], "explainer"),
    ("stability", ["rerun_cv", "repeat_pair_rho", "lipschitz"], "explainer"),
]


def estimate_table(profile: Profile) -> pd.DataFrame:
    """A BCa interval for every point estimate that reaches a table or figure."""
    data = _load(profile)
    rows = []
    for source, columns, by in _ESTIMATE_SPECS:
        frame = data[source]
        keys = ["dataset", "model"] + ([by] if by else [])
        for group_key, group in frame.groupby(keys):
            labels = dict(zip(keys, group_key if isinstance(group_key, tuple) else (group_key,)))
            for column in columns:
                if column not in group:
                    continue
                values = group[column].dropna().to_numpy(float)
                if values.size == 0:
                    continue
                # Median is non-smooth, so percentile rather than BCa (amendment A5).
                low, high = percentile_ci(values, np.median,
                                          profile.ci_resamples, SEED)
                rows.append({"metric": column, "source": source,
                             "explainer": labels.get("explainer", "shap_vs_lime"),
                             "dataset": labels["dataset"], "model": labels["model"],
                             "point": float(np.median(values)), "ci_low": low,
                             "ci_high": high, "n": int(values.size)})
    return pd.DataFrame(rows)


def gate_failed_cells() -> list[tuple[str, str]]:
    """Cells where the pre-registered additivity gate did not pass (protocol risk R1)."""
    config = pd.read_csv(paths().results / "explainer_config.csv")
    failed = config[~config.additivity_gate_ok.astype(bool)]
    return [(r.dataset, r.model) for r in failed.itertuples()]


def sensitivity_excluding_gate_failures(profile: Profile) -> pd.DataFrame:
    """Re-tally every hypothesis with R1-affected cells removed.

    R1 fired for one cell, so the protocol requires it be handled rather than ignored.
    Rather than widen the gate a second time -- which would be moving the goalposts
    after seeing the result -- we report the headline tallies both with and without the
    affected cell, so a reader can see for themselves that nothing turns on it.
    """
    excluded = set(gate_failed_cells())
    stats = pd.read_csv(paths().results / "stats_hypotheses.csv")
    kept = stats[~stats.apply(lambda r: (r.dataset, r.model) in excluded, axis=1)]
    rows = []
    for hypothesis in stats.hypothesis.unique():
        full = stats.query("hypothesis == @hypothesis")
        trimmed = kept.query("hypothesis == @hypothesis")
        rows.append({
            "hypothesis": hypothesis,
            "cells_all": len(full),
            "rejected_all": int((full.decision == "reject H0").sum()),
            "cells_excl": len(trimmed),
            "rejected_excl": int((trimmed.decision == "reject H0").sum()),
            "excluded_cells": "; ".join(f"{d}/{m}" for d, m in sorted(excluded)) or "none",
        })
    return pd.DataFrame(rows)


def main(force: bool = False) -> None:
    profile, p = get_profile(), paths()
    p.mkdirs()
    hypotheses = p.results / "stats_hypotheses.csv"
    estimates = p.results / "stats_estimates.csv"
    if hypotheses.exists() and estimates.exists() and not force:
        print("[stage4] stats exist; skipping (use --force to recompute)")
        return
    print("[stage4] hypotheses", flush=True)
    run_all_hypotheses(profile).to_csv(hypotheses, index=False)
    print("[stage4] interval estimates", flush=True)
    estimate_table(profile).to_csv(estimates, index=False)
    failed = gate_failed_cells()
    if failed:
        print(f"[stage4] protocol risk R1 fired for {len(failed)} cell(s): "
              f"{failed}; writing sensitivity analysis", flush=True)
    sensitivity_excluding_gate_failures(profile).to_csv(
        p.results / "stats_sensitivity_r1.csv", index=False)
    print("[stage4] done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    main(**vars(parser.parse_args()))
