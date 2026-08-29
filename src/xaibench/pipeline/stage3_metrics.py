"""Stage 3: turn attributions into the five metric families.

Reads only what Stage 2 wrote. Cells are rebuilt with build_cell, which is
deterministic and cheap, rather than pickled.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from xaibench.config import (
    ALL_EXPLAINERS,
    AOPC_MAX_K,
    CELLS,
    COMP_SUFF_K,
    JACCARD_KS,
    RBO_P,
    SEED,
    Profile,
    get_profile,
    paths,
)
from xaibench.metrics.agreement import (
    jaccard_at_k,
    kendall_tau_b,
    rank_vector,
    ranking_from_attributions,
    rbo,
    rbo_floor,
    rbo_normalised,
    spearman,
)
from xaibench.metrics.fidelity import (
    batched_curve_metrics,
    marginal_baselines,
    median_baseline,
)
from xaibench.metrics.plausibility import (
    jaccard_vs_reference,
    load_reference_sets,
    precision_at_k,
)
from xaibench.metrics.stability import retrain_agreement
from xaibench.models.cells import build_cell
from xaibench.pipeline.stage2_explain import transformed_ranking

GLOBAL_INSTANCE = -1     # sentinel for the dataset-level (global) ranking row


def _attribution_matrix(frame: pd.DataFrame, feature_order: list[str]) -> dict:
    """{instance_id: attribution vector} in feature_order, from a long-format frame."""
    wide = frame.pivot_table(index="instance_id", columns="feature",
                             values="attribution", aggfunc="first")
    wide = wide.reindex(columns=feature_order)
    return {int(i): row.to_numpy(dtype=float) for i, row in wide.iterrows()}


def _load_cell_attributions(p, dataset_id: str, model_id: str) -> pd.DataFrame:
    return pd.read_parquet(p.attributions / f"{dataset_id}_{model_id}_attr.parquet")


def _primary(frame: pd.DataFrame, explainer: str, num_samples: int) -> pd.DataFrame:
    mask = frame.explainer == explainer
    if explainer == "lime":
        mask &= frame.num_samples == num_samples
    return frame[mask]


def agreement_metrics(profile: Profile) -> pd.DataFrame:
    """RQ1: SHAP versus the primary LIME configuration, per instance and globally."""
    p = paths()
    primary_n = profile.lime_num_samples[0]
    rows = []
    for dataset_id, model_id in CELLS:
        frame = _load_cell_attributions(p, dataset_id, model_id)
        order = sorted(frame.feature.unique(), key=lambda f: f)
        cell_order = list(build_cell(dataset_id, model_id, {}, profile).feature_order)
        order = cell_order if set(cell_order) == set(order) else order
        shap_vectors = _attribution_matrix(_primary(frame, "shap", primary_n), order)
        lime_vectors = _attribution_matrix(_primary(frame, "lime", primary_n), order)

        def record(instance_id, a, b):
            rank_a = ranking_from_attributions(a)
            rank_b = ranking_from_attributions(b)
            rows.append({
                "dataset": dataset_id, "model": model_id, "instance_id": instance_id,
                "spearman": spearman(rank_vector(rank_a), rank_vector(rank_b)),
                "kendall": kendall_tau_b(rank_vector(rank_a), rank_vector(rank_b)),
                **{f"jaccard{k}": jaccard_at_k(rank_a, rank_b, k) for k in JACCARD_KS},
                "rbo": rbo(rank_a, rank_b, p=RBO_P),
                "rbo_norm": rbo_normalised(rank_a, rank_b, p=RBO_P),
                "rbo_floor": rbo_floor(len(order), RBO_P),
                "n_features": len(order),
            })

        for instance_id in sorted(set(shap_vectors) & set(lime_vectors)):
            record(instance_id, shap_vectors[instance_id], lime_vectors[instance_id])
        # Global ranking: mean |attribution| over the explained instances.
        record(GLOBAL_INSTANCE,
               np.mean([np.abs(v) for v in shap_vectors.values()], axis=0),
               np.mean([np.abs(v) for v in lime_vectors.values()], axis=0))
    return pd.DataFrame(rows)


def fidelity_metrics(profile: Profile) -> pd.DataFrame:
    """RQ3 and RQ6: deletion/insertion fidelity for every explainer, both baselines."""
    p = paths()
    primary_n = profile.lime_num_samples[0]
    splits = pd.read_csv(p.results / "consensus_splits.csv")
    eval_ids = {(r.dataset, r.model): {int(v) for v in str(r.evaluation_ids).split()}
                for r in splits.itertuples()}
    rows = []
    for dataset_id, model_id in CELLS:
        params = json.loads((p.results / "best_params.json").read_text())
        cell = build_cell(dataset_id, model_id, params.get(f"{dataset_id}|{model_id}", {}),
                          profile)
        frame = _load_cell_attributions(p, dataset_id, model_id)
        baselines = {
            "median": [median_baseline(cell.X_train_t)],
            "marginal": list(marginal_baselines(cell.X_train_t,
                                                profile.marginal_draws, SEED)),
        }
        for explainer in ALL_EXPLAINERS:
            vectors = _attribution_matrix(_primary(frame, explainer, primary_n),
                                          cell.feature_order)
            for instance_id, attribution in vectors.items():
                groups = transformed_ranking(
                    cell, _FakeExplanation(attribution, cell.feature_order))
                x = cell.X_test_t[instance_id]
                for baseline_name, draws in baselines.items():
                    averaged = batched_curve_metrics(cell.predict_proba, x, groups,
                                                     draws, AOPC_MAX_K, COMP_SUFF_K)
                    rows.append({
                        "dataset": dataset_id, "model": model_id,
                        "explainer": explainer, "instance_id": int(instance_id),
                        "baseline": baseline_name, **averaged,
                        "is_evaluation": int(instance_id) in eval_ids[
                            (dataset_id, model_id)],
                    })
    return pd.DataFrame(rows)


class _FakeExplanation:
    """Minimal duck-type so transformed_ranking works on a bare attribution vector."""

    def __init__(self, attributions, feature_order):
        self.attributions = np.asarray(attributions, dtype=float)
        self._order = list(feature_order)

    @property
    def ranking(self):
        return ranking_from_attributions(self.attributions)


def plausibility_metrics(profile: Profile) -> pd.DataFrame:
    """RQ5: overlap with the pre-registered reference sets. Corroborating only."""
    p = paths()
    primary_n = profile.lime_num_samples[0]
    references = load_reference_sets()
    rows = []
    for dataset_id, model_id in CELLS:
        cell_order = build_cell(dataset_id, model_id, {}, profile).feature_order
        frame = _load_cell_attributions(p, dataset_id, model_id)
        reference = references[dataset_id].features
        for explainer in ALL_EXPLAINERS:
            vectors = _attribution_matrix(_primary(frame, explainer, primary_n),
                                          cell_order)
            for instance_id, attribution in vectors.items():
                ranking = ranking_from_attributions(attribution)
                rows.append({
                    "dataset": dataset_id, "model": model_id, "explainer": explainer,
                    "instance_id": int(instance_id),
                    "jaccard5": jaccard_vs_reference(ranking, cell_order, reference, k=5),
                    "precision5": precision_at_k(ranking, cell_order, reference, k=5),
                })
    return pd.DataFrame(rows)


def runtime_metrics(profile: Profile) -> pd.DataFrame:
    """RQ4: cost per explanation, plus the fidelity it buys."""
    p = paths()
    frames = [_load_cell_attributions(p, d, m) for d, m in CELLS]
    per_explanation = (pd.concat(frames, ignore_index=True)
                       .groupby(["dataset", "model", "explainer", "instance_id",
                                 "num_samples"], as_index=False)
                       .runtime_s.first())
    summary = (per_explanation
               .groupby(["dataset", "model", "explainer", "num_samples"])
               .runtime_s.agg(median_seconds="median",
                              q1_seconds=lambda s: s.quantile(0.25),
                              q3_seconds=lambda s: s.quantile(0.75),
                              n="size")
               .reset_index())
    summary["iqr_seconds"] = summary.q3_seconds - summary.q1_seconds

    fidelity = pd.read_csv(p.results / "metrics_fidelity.csv")
    aopc = (fidelity.query("baseline == 'median'")
            .groupby(["dataset", "model", "explainer"], as_index=False).aopc.median())
    summary = summary.merge(aopc, on=["dataset", "model", "explainer"], how="left")
    summary["fidelity_per_second"] = summary.aopc / summary.median_seconds.replace(0, np.nan)
    return summary


def stability_summary(profile: Profile) -> pd.DataFrame:
    """RQ2: the three components joined into one frame, plus per-cell retrain rho."""
    p = paths()
    primary_n = profile.lime_num_samples[0]
    rerun = pd.read_parquet(p.results / "stability_rerun.parquet")
    rerun = rerun[(rerun.explainer != "lime") | (rerun.num_samples == primary_n)]
    robust = pd.read_parquet(p.results / "stability_robust.parquet")
    merged = rerun.merge(robust, on=["dataset", "model", "explainer", "instance_id"],
                         how="outer")

    retrain_path = p.results / "stability_retrain.parquet"
    if retrain_path.exists():
        retrain = pd.read_parquet(retrain_path)
        agreement = []
        for (dataset_id, model_id, explainer), group in retrain.groupby(
                ["dataset", "model", "explainer"]):
            matrix = group.pivot_table(index="refit", columns="feature",
                                       values="global_attribution").to_numpy(dtype=float)
            agreement.append({"dataset": dataset_id, "model": model_id,
                              "explainer": explainer,
                              "retrain_agreement": retrain_agreement(matrix)})
        merged = merged.merge(pd.DataFrame(agreement),
                              on=["dataset", "model", "explainer"], how="left")
    else:
        merged["retrain_agreement"] = np.nan
    return merged


def main(force: bool = False) -> None:
    profile, p = get_profile(), paths()
    p.mkdirs()
    targets = {name: p.results / f"metrics_{name}.csv"
               for name in ["agreement", "fidelity", "plausibility", "runtime", "stability"]}
    if all(t.exists() for t in targets.values()) and not force:
        print("[stage3] metrics exist; skipping (use --force to recompute)")
        return

    print("[stage3] agreement", flush=True)
    agreement_metrics(profile).to_csv(targets["agreement"], index=False)
    print("[stage3] fidelity", flush=True)
    fidelity_metrics(profile).to_csv(targets["fidelity"], index=False)
    print("[stage3] plausibility", flush=True)
    plausibility_metrics(profile).to_csv(targets["plausibility"], index=False)
    print("[stage3] runtime", flush=True)
    runtime_metrics(profile).to_csv(targets["runtime"], index=False)
    print("[stage3] stability", flush=True)
    stability_summary(profile).to_csv(targets["stability"], index=False)
    print("[stage3] done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    main(**vars(parser.parse_args()))
