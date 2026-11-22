"""Unit tests for Post-Review Amendment A8 revision analyses."""
from __future__ import annotations

import numpy as np
import pandas as pd

from xaibench.data.loaders import Dataset
from xaibench.metrics.conditional_fidelity import (
    conditional_knn_deletion_curve,
    conditional_knn_metrics,
    roar_for_cell,
)
from xaibench.pipeline.stage6_revision import compute_redundancy_metrics


def test_conditional_knn_preserves_joint_correlation_on_known_linear_model():
    """On perfectly correlated features (x0 == x1), deleting x0 conditionally on x1
    leaves x0 almost unchanged because nearest neighbours on x1 share x0's value!
    Meanwhile deleting BOTH (x0, x1) restores full drop.
    """
    rng = np.random.default_rng(42)
    z = rng.normal(size=200)
    X_train = np.column_stack([z, z + rng.normal(scale=1e-3, size=200), rng.normal(size=200)])

    def predict_proba(rows):
        rows = np.atleast_2d(rows)
        p1 = 1.0 / (1.0 + np.exp(-(2.5 * rows[:, 0])))
        return np.column_stack([1.0 - p1, p1])

    x = np.array([2.0, 2.0, 0.0])
    groups = [np.array([0]), np.array([1]), np.array([2])]
    curve = conditional_knn_deletion_curve(
        predict_proba, x, groups, X_train, k_max=3, n_neighbors=20, seed=42,
    )
    assert curve.shape == (4,)
    # Deleting x0 alone while conditioning on x1 ~= 2.0 keeps p(class 1) very high.
    assert abs(curve[0] - curve[1]) < 0.05
    # Deleting BOTH x0 and x1 drops p(class 1) dramatically toward base rate ~0.5.
    assert (curve[0] - curve[2]) > 0.30


def test_conditional_knn_metrics_returns_finite_values():
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(50, 4))

    def predict_proba(rows):
        rows = np.atleast_2d(rows)
        p1 = 1.0 / (1.0 + np.exp(-rows[:, 0]))
        return np.column_stack([1.0 - p1, p1])

    x = np.array([1.5, -0.5, 0.2, 0.1])
    groups = [np.array([i]) for i in range(4)]
    m = conditional_knn_metrics(predict_proba, x, groups, X_train, k_max=4, k=2, n_neighbors=5, seed=7)
    assert np.isfinite(m["aopc_cond"]) and np.isfinite(m["comp_cond"])
    assert m["aopc_cond"] > 0.0


def test_roar_drops_auc_more_for_true_features_than_irrelevant_ones():
    """Synthetic dataset where f0, f1 drive y and f2..f5 are pure noise."""
    rng = np.random.default_rng(123)
    n = 160
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(6)})
    logits = 2.0 * X["f0"] - 2.0 * X["f1"]
    y = pd.Series((logits + rng.normal(scale=0.5, size=n) > 0).astype(int))
    ds = Dataset(
        id="synth", X=X, y=y, numeric=list(X.columns), categorical=[],
        source_url="test", licence="MIT", citation="test",
    )
    rankings = {
        "shap": ["f0", "f1", "f2", "f3", "f4", "f5"],
        "lime": ["f5", "f4", "f3", "f2", "f1", "f0"],  # deliberately inverted
    }
    res = roar_for_cell(ds, "logreg", {}, rankings, fractions=(0.2, 0.35),
                        n_splits=3, n_random_draws=2, seed=42)
    pos = res.query("fraction > 0")
    shap_drop = pos.query("explainer == 'shap'").auc_drop.mean()
    lime_drop = pos.query("explainer == 'lime'").auc_drop.mean()
    assert shap_drop > lime_drop + 0.15


def test_redundancy_metrics_separates_collinear_from_independent():
    rng = np.random.default_rng(99)
    z = rng.normal(size=120)
    y = pd.Series((z > 0).astype(int))
    X_col = pd.DataFrame({f"c{i}": z + rng.normal(scale=0.05, size=120) for i in range(5)})
    X_ind = pd.DataFrame({f"i{i}": rng.normal(size=120) for i in range(5)})
    X_ind["i0"] = z + rng.normal(scale=0.5, size=120)

    r_col = compute_redundancy_metrics(Dataset("col", X_col, y, list(X_col.columns), [], "", "", ""))
    r_ind = compute_redundancy_metrics(Dataset("ind", X_ind, y, list(X_ind.columns), [], "", "", ""))
    assert r_col["mean_abs_spearman"] > 0.85
    assert r_col["effective_rank_ratio"] < r_ind["effective_rank_ratio"]
    assert r_col["n_uni_auc_ge_090"] == 5
    assert r_ind["n_uni_auc_ge_090"] <= 1
