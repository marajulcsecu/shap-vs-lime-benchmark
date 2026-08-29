"""Stage 2: every explanation the study needs, checkpointed per cell.

Per cell this produces base attributions (SHAP, and LIME at each num_samples), the
three fusion variants, the random control, and all three stability components. One
parquet per cell, so a crash at cell 15 does not discard cells 1-14.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.utils import resample

from xaibench.config import (
    AOPC_MAX_K,
    CELLS,
    ENSEMBLE_MODELS,
    LIPSCHITZ_SIGMA,
    SEED,
    Profile,
    get_profile,
    paths,
)
from xaibench.explain.base import explanations_to_frame
from xaibench.explain.consensus import (
    CONSENSUS_RULES,
    build_consensus,
    calibration_split,
    fidelity_weight,
    random_control,
)
from xaibench.explain.lime_backend import explain_lime
from xaibench.explain.shap_backend import (
    additivity_gate_ok,
    additivity_report,
    build_background,
    explain_shap,
    raw_shap_values,
)
from xaibench.metrics.fidelity import (
    curve_metrics,
    deletion_curve_grouped,
    median_baseline,
)
from xaibench.metrics.stability import (
    local_lipschitz,
    repeat_pair_rho,
    rerun_cv,
    sample_neighbours,
)
from xaibench.models.cells import build_cell
from xaibench.models.zoo import build_model, shap_backend_kind


@dataclass
class CellArtifacts:
    attributions: pd.DataFrame
    rerun: pd.DataFrame
    robust: pd.DataFrame
    retrain: pd.DataFrame
    config_row: dict
    budget_row: dict
    calib_ids: np.ndarray
    eval_ids: np.ndarray


def transformed_ranking(cell, explanation) -> list[np.ndarray]:
    """Lift an original-feature ranking to per-feature groups of transformed columns.

    Fidelity perturbs the model's actual inputs, which are transformed columns, but
    rankings live in original-feature space. Deleting "the top-k features" must delete
    every one-hot column belonging to those k features -- deleting k *columns* instead
    would under-perturb Heart (13 features across 25 columns) while leaving PIMA
    unaffected, so the datasets would not be measured on the same scale. Returning
    groups rather than a flat order makes that impossible to get wrong downstream.
    """
    groups = np.asarray(cell.groups)
    return [np.flatnonzero(groups == cell.feature_order[int(i)])
            for i in explanation.ranking]


def _mean_aopc(cell, explanations, baseline) -> float:
    if not explanations:
        return 0.0
    return float(np.mean([
        curve_metrics(
            deletion_curve_grouped(cell.predict_proba, cell.X_test_t[exp.instance_id],
                                   transformed_ranking(cell, exp), baseline, AOPC_MAX_K),
            np.zeros(2))["aopc"]
        for exp in explanations]))


def _rerun_rows(cell, profile: Profile, ids: np.ndarray) -> list[dict]:
    """RQ2a: repeat each explainer R times with different seeds on a small subset."""
    rows: list[dict] = []
    subset = ids[: profile.rerun_instances]
    background = build_background(cell, profile.shap_background, SEED)

    shap_runs = np.array([
        [exp.attributions for exp in
         explain_shap(cell, subset, seed=1000 + r, background=background)]
        for r in range(profile.rerun_repeats)])
    for i, inst in enumerate(subset):
        rows.append({"dataset": cell.dataset, "model": cell.model, "explainer": "shap",
                     "instance_id": int(inst), "num_samples": 0,
                     "rerun_cv": rerun_cv(shap_runs[:, i, :]),
                     "repeat_pair_rho": repeat_pair_rho(shap_runs[:, i, :])})

    for n_samples in profile.lime_num_samples:
        runs = np.array([
            [exp.attributions for exp in
             explain_lime(cell, subset, num_samples=n_samples, seed=1000 + r)]
            for r in range(profile.rerun_repeats)])
        for i, inst in enumerate(subset):
            rows.append({"dataset": cell.dataset, "model": cell.model,
                         "explainer": "lime", "instance_id": int(inst),
                         "num_samples": int(n_samples),
                         "rerun_cv": rerun_cv(runs[:, i, :]),
                         "repeat_pair_rho": repeat_pair_rho(runs[:, i, :])})
    return rows


def _robust_rows(cell, profile: Profile, ids: np.ndarray) -> list[dict]:
    """RQ2b: local Lipschitz estimate over Gaussian neighbours in standardised space."""
    rows: list[dict] = []
    subset = ids[: profile.robust_instances]
    background = build_background(cell, profile.shap_background, SEED)
    sigma = cell.train_sigma * LIPSCHITZ_SIGMA
    n_test = cell.X_test_t.shape[0]

    for inst in subset:
        x = cell.X_test_t[int(inst)]
        neighbours = sample_neighbours(x, sigma, profile.robust_neighbours,
                                       seed=SEED + int(inst),
                                       fixed_mask=cell.fixed_mask)
        # Cell is frozen, so replace() gives a view whose test matrix carries the
        # neighbours; they can then be explained by index like any other row.
        probe = replace(cell, X_test_t=np.vstack([cell.X_test_t, neighbours]))
        neighbour_ids = np.arange(n_test, n_test + neighbours.shape[0])

        backends = {
            "shap": lambda i: explain_shap(probe, i, seed=SEED, background=background),
            "lime": lambda i: explain_lime(probe, i, seed=SEED,
                                           num_samples=profile.lime_num_samples[0]),
        }
        for name, explain in backends.items():
            phi_x = explain(np.array([int(inst)]))[0].attributions
            phi_n = np.array([exp.attributions for exp in explain(neighbour_ids)])
            rows.append({"dataset": cell.dataset, "model": cell.model,
                         "explainer": name, "instance_id": int(inst),
                         "lipschitz": local_lipschitz(phi_x, x, phi_n, neighbours)})
    return rows


def _retrain_rows(cell, profile: Profile, ids: np.ndarray) -> list[dict]:
    """RQ2c: refit on bootstrap resamples and compare GLOBAL rankings.

    Restricted to ensemble models (protocol R3): the LIME cost of running this for
    all six models does not fit the compute budget, and the restriction is disclosed
    in the paper rather than buried.
    """
    if cell.model not in ENSEMBLE_MODELS:
        return []
    rows: list[dict] = []
    subset = ids[: profile.bootstrap_instances]
    for b in range(profile.bootstrap_refits):
        index = resample(np.arange(cell.X_train_t.shape[0]), replace=True,
                         n_samples=cell.X_train_t.shape[0], stratify=cell.y_train,
                         random_state=2000 + b)
        classifier = build_model(cell.model, seed=SEED).fit(
            cell.X_train_t[index], cell.y_train[index])
        refit = replace(cell, classifier=classifier)
        background = build_background(refit, profile.shap_background, SEED)
        variants = {
            "shap": explain_shap(refit, subset, seed=SEED, background=background),
            "lime": explain_lime(refit, subset, seed=SEED,
                                 num_samples=profile.lime_num_samples[0]),
        }
        for name, explanations in variants.items():
            global_attr = np.mean([np.abs(e.attributions) for e in explanations], axis=0)
            for j, feature in enumerate(cell.feature_order):
                rows.append({"dataset": cell.dataset, "model": cell.model,
                             "explainer": name, "refit": b, "feature": feature,
                             "global_attribution": float(global_attr[j])})
    return rows


def explain_cell(dataset_id: str, model_id: str, params: dict,
                 profile: Profile) -> CellArtifacts:
    cell = build_cell(dataset_id, model_id, params, profile)
    ids = cell.instance_ids
    background = build_background(cell, profile.shap_background, SEED)
    baseline = median_baseline(cell.X_train_t)

    values, base = raw_shap_values(cell, ids, background, SEED)
    additivity = additivity_report(cell, values, base, ids)

    shap_exps = explain_shap(cell, ids, seed=SEED, background=background)
    shap_frame = explanations_to_frame(shap_exps)
    shap_frame["num_samples"] = 0
    frames = [shap_frame]

    lime_by_n: dict[int, list] = {}
    for n_samples in profile.lime_num_samples:
        lime_by_n[n_samples] = explain_lime(cell, ids, num_samples=n_samples, seed=SEED)
        frame = explanations_to_frame(lime_by_n[n_samples])
        frame["num_samples"] = int(n_samples)
        frames.append(frame)

    primary_lime = lime_by_n[profile.lime_num_samples[0]]
    calib_ids, eval_ids = calibration_split(ids, seed=SEED)
    position = {int(inst): k for k, inst in enumerate(ids)}
    calib_positions = [position[int(i)] for i in calib_ids]
    weight = fidelity_weight(
        _mean_aopc(cell, [shap_exps[k] for k in calib_positions], baseline),
        _mean_aopc(cell, [primary_lime[k] for k in calib_positions], baseline))

    for rule in CONSENSUS_RULES:
        frame = explanations_to_frame(
            build_consensus(shap_exps, primary_lime, weight, rule))
        frame["num_samples"] = 0
        frames.append(frame)

    control = explanations_to_frame(random_control(cell, ids, seed=SEED))
    control["num_samples"] = 0
    frames.append(control)

    return CellArtifacts(
        attributions=pd.concat(frames, ignore_index=True),
        rerun=pd.DataFrame(_rerun_rows(cell, profile, ids)),
        robust=pd.DataFrame(_robust_rows(cell, profile, ids)),
        retrain=pd.DataFrame(_retrain_rows(cell, profile, ids)),
        config_row={
            "dataset": dataset_id, "model": model_id,
            "shap_backend": shap_backend_kind(model_id), "space": "probability",
            **additivity,
            "additivity_gate_ok": additivity_gate_ok(additivity, ids.size),
            "background": int(background.shape[0]),
            "fidelity_weight_shap": weight,
            "lime_num_samples_primary": int(profile.lime_num_samples[0]),
            "profile": profile.name,
        },
        budget_row={
            "dataset": dataset_id, "model": model_id,
            "n_test": int(cell.X_test_t.shape[0]), "n_explained": int(ids.size),
            "n_calibration": int(calib_ids.size), "n_evaluation": int(eval_ids.size),
        },
        calib_ids=calib_ids, eval_ids=eval_ids)


def main(force: bool = False) -> None:
    profile, p = get_profile(), paths()
    p.mkdirs()
    best = json.loads((p.results / "best_params.json").read_text())
    configs, budgets, splits = [], [], []

    for dataset_id, model_id in CELLS:
        stem = f"{dataset_id}_{model_id}"
        target = p.attributions / f"{stem}_attr.parquet"
        side = p.attributions / f"{stem}_side.json"
        if target.exists() and side.exists() and not force:
            print(f"[stage2] {dataset_id:6s} {model_id:7s} cached")
            payload = json.loads(side.read_text())
        else:
            print(f"[stage2] {dataset_id:6s} {model_id:7s} explaining "
                  f"(profile={profile.name})", flush=True)
            art = explain_cell(dataset_id, model_id,
                               best.get(f"{dataset_id}|{model_id}", {}), profile)
            art.attributions.to_parquet(target, index=False)
            for name, frame in (("rerun", art.rerun), ("robust", art.robust),
                                ("retrain", art.retrain)):
                if not frame.empty:
                    frame.to_parquet(p.attributions / f"{stem}_{name}.parquet",
                                     index=False)
            payload = {"config": art.config_row, "budget": art.budget_row,
                       "calib_ids": art.calib_ids.tolist(),
                       "eval_ids": art.eval_ids.tolist()}
            side.write_text(json.dumps(payload, indent=2) + "\n")

        configs.append(payload["config"])
        budgets.append(payload["budget"])
        splits.append({"dataset": dataset_id, "model": model_id,
                       "calibration_ids": " ".join(map(str, payload["calib_ids"])),
                       "evaluation_ids": " ".join(map(str, payload["eval_ids"]))})

    for name, frame in (("explainer_config", pd.DataFrame(configs)),
                        ("instance_budget", pd.DataFrame(budgets)),
                        ("consensus_splits", pd.DataFrame(splits))):
        frame.to_csv(p.results / f"{name}.csv", index=False)

    for name in ("rerun", "robust", "retrain"):
        parts = [pd.read_parquet(f)
                 for f in sorted(p.attributions.glob(f"*_{name}.parquet"))]
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(
                p.results / f"stability_{name}.parquet", index=False)
    print("[stage2] done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    main(**vars(parser.parse_args()))
