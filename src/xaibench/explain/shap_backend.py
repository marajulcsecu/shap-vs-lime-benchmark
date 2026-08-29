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
from sklearn.cluster import KMeans

from xaibench.config import SEED
from xaibench.explain.base import Explanation, config_hash
from xaibench.models.pipelines import aggregate_attributions
from xaibench.models.zoo import shap_backend_kind

SHAP_ADDITIVITY_TOL = 5e-3


def build_background(cell, k: int, seed: int) -> np.ndarray:
    """k-means centroids of the training rows, used as the interventional reference.

    Centroids rather than a random subsample: with 50 references the variance of a
    random draw is visible in the attributions, and the protocol needs a fixed
    background so tree SHAP stays deterministic.
    """
    X = np.asarray(cell.X_train_t, dtype=float)
    if int(k) >= X.shape[0]:
        return X.copy()
    km = KMeans(n_clusters=int(k), n_init=10, random_state=seed).fit(X)
    return np.asarray(km.cluster_centers_, dtype=float)


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
    out = explainer(X, max_evals=2 * X.shape[1] + 1, silent=True)
    return _normalise(out.values, out.base_values)


def additivity_error(cell, values: np.ndarray, base: float, instance_ids) -> float:
    """max |base + sum(phi) - p(class 1)|. The check that proves probability space."""
    actual = cell.predict_proba(
        np.asarray(cell.X_test_t, dtype=float)[np.asarray(instance_ids, dtype=int)])[:, 1]
    return float(np.max(np.abs(base + values.sum(axis=1) - actual)))


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
