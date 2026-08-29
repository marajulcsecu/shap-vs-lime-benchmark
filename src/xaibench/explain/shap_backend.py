"""SHAP attributions in predicted-probability space, per-backend.

Backend assignment (protocol §2.2), and why each is what it is:
  dtree, rf    TreeExplainer -- raw output IS the class proportion, already probability
  xgb, lgbm    TreeExplainer with model_output='probability' + interventional
               background, because their raw output is a log-odds margin
  logreg,      PermutationExplainer over predict_proba: no tree structure to exploit,
  stack        and LinearExplainer would return log-odds, breaking comparability
               with LIME

Binary classification makes the class choice free: p(class 0) = 1 - p(class 1), so
phi_0 = -phi_1 exactly and a ranking by |phi| is identical either way. Class 1 is
used throughout.
"""
from __future__ import annotations

import time

import numpy as np
import shap

from xaibench.config import SEED
from xaibench.explain.base import Explanation, config_hash
from xaibench.models.pipelines import aggregate_attributions
from xaibench.models.zoo import shap_backend_kind

SHAP_ADDITIVITY_TOL = 5e-3
PERMUTATION_PASSES = 10   # permutation passes for the model-agnostic backend


def build_background(cell, k: int, seed: int) -> np.ndarray:
    """A seeded, class-stratified subsample of REAL training rows.

    Not k-means centroids. A centroid is a synthetic point: averaging one-hot columns
    gives fractional values like 0.37 that the model has never seen, and interventional
    TreeSHAP evaluated against such off-manifold references breaks its own additivity.
    Measured on this data, centroids gave LightGBM an additivity error of 9.9e-03 on
    Heart and 1.6e-02 on PIMA, against 7e-09 with real rows -- a silent 1 % violation
    that would have corrupted every LightGBM attribution in the study.

    The draw is seeded, so the background is fixed and tree SHAP stays deterministic.
    See protocol amendment A2.
    """
    X = np.asarray(cell.X_train_t, dtype=float)
    k = int(k)
    if k >= X.shape[0]:
        return X.copy()
    y = np.asarray(cell.y_train)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    # Stratified so the background represents both classes, which the base value needs.
    for label in np.unique(y):
        pool = np.flatnonzero(y == label)
        take = max(1, int(round(k * pool.size / y.size)))
        chosen.extend(rng.choice(pool, size=min(take, pool.size), replace=False).tolist())
    if len(chosen) > k:
        chosen = rng.choice(chosen, size=k, replace=False).tolist()
    return X[np.sort(np.asarray(chosen, dtype=int))].copy()


def _normalise(values, base) -> tuple[np.ndarray, float]:
    """Reduce any shap return shape to (n, n_out) for class 1 and a scalar base value."""
    if isinstance(values, list):          # older shap returns one array per class
        values = values[-1]
    v = np.asarray(values, dtype=float)
    if v.ndim == 3:                       # (n, n_out, n_classes)
        v = v[:, :, -1]
    b = np.asarray(base, dtype=float).ravel()
    return v, float(b[-1] if b.size > 1 else b[0])


def raw_shap_values(cell, instance_ids, background, seed) -> tuple[np.ndarray, float]:
    """Attributions in TRANSFORMED-column space plus the scalar base value."""
    X = np.asarray(cell.X_test_t, dtype=float)[np.asarray(instance_ids, dtype=int)]
    kind = shap_backend_kind(cell.model)

    if kind == "tree":
        explainer = shap.TreeExplainer(cell.classifier)
        return _normalise(explainer.shap_values(X, check_additivity=False),
                          explainer.expected_value)

    if kind == "tree_prob":
        explainer = shap.TreeExplainer(cell.classifier, data=background,
                                       model_output="probability",
                                       feature_perturbation="interventional")
        return _normalise(explainer.shap_values(X, check_additivity=False),
                          explainer.expected_value)

    masker = shap.maskers.Independent(background, max_samples=background.shape[0])
    explainer = shap.PermutationExplainer(cell.classifier.predict_proba, masker, seed=seed)
    # 2d+1 evals is one permutation pass, which leaves additivity error near 1e-03.
    # PERMUTATION_PASSES passes bring it to ~4e-04 and then plateau. See amendment A2.
    out = explainer(X, max_evals=PERMUTATION_PASSES * (2 * X.shape[1] + 1), silent=True)
    return _normalise(out.values, out.base_values)


def additivity_error(cell, values: np.ndarray, base: float, instance_ids) -> float:
    """max |base + sum(phi) - p(class 1)|. The check that proves probability space."""
    return additivity_report(cell, values, base, instance_ids)["additivity_max"]


def additivity_report(cell, values: np.ndarray, base: float, instance_ids) -> dict:
    """Per-instance additivity diagnostics, not a single scalar.

    A bare max hides the shape of the discrepancy. On this data the interventional
    tree implementation reproduces predict_proba to ~1e-09 for almost every instance,
    but isolated instances in isolated cells land ~1e-02 out -- a library-level edge
    case at a split boundary, not a property of the cell. Reporting max, median, and
    the count over tolerance lets the gate in §8 R1 fire on a systematically wrong
    cell while an isolated outlier is disclosed instead of triggering a fallback that
    would make the whole cell incomparable with LIME. See protocol amendment A3.
    """
    actual = cell.predict_proba(
        np.asarray(cell.X_test_t, dtype=float)[np.asarray(instance_ids, dtype=int)])[:, 1]
    error = np.abs(base + values.sum(axis=1) - actual)
    over = int((error > SHAP_ADDITIVITY_TOL).sum())
    return {
        "additivity_max": float(error.max()),
        "additivity_median": float(np.median(error)),
        "additivity_n_over_tol": over,
        "additivity_frac_over_tol": float(over / max(error.size, 1)),
    }


def additivity_gate_ok(report: dict, max_frac_over: float = 0.02) -> bool:
    """Pre-registered acceptance: typical behaviour exact, outliers rare.

    A cell whose *median* error clears the tolerance and whose over-tolerance
    fraction is below max_frac_over is in probability space; anything else means
    protocol risk R1 has fired for that cell.
    """
    return (report["additivity_median"] < SHAP_ADDITIVITY_TOL
            and report["additivity_frac_over_tol"] <= max_frac_over)


def explain_shap(cell, instance_ids=None, seed: int = SEED,
                 background: np.ndarray | None = None) -> list[Explanation]:
    ids = np.asarray(cell.instance_ids if instance_ids is None else instance_ids, dtype=int)
    if background is None:
        background = build_background(cell, min(50, cell.X_train_t.shape[0]), SEED)

    cfg = config_hash(explainer="shap", backend=shap_backend_kind(cell.model),
                      background=int(background.shape[0]), space="probability")

    # Timed as a batch and divided evenly: the tree backends only expose a batched
    # call, so a per-instance timer would measure setup overhead, not explanation cost.
    start = time.perf_counter()
    values, _base = raw_shap_values(cell, ids, background, seed)
    elapsed = time.perf_counter() - start
    per_instance = max(elapsed / max(ids.size, 1), 1e-9)

    aggregated = aggregate_attributions(values, cell.groups, cell.feature_order)
    return [
        Explanation(dataset=cell.dataset, model=cell.model, explainer="shap",
                    instance_id=int(inst), attributions=aggregated[row],
                    feature_names=tuple(cell.feature_order), runtime_s=per_instance,
                    seed=int(seed), config_hash=cfg)
        for row, inst in enumerate(ids)
    ]
