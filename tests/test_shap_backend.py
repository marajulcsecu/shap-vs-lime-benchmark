import numpy as np
import pytest

from xaibench.config import MODELS, QUICK
from xaibench.explain.shap_backend import (
    SHAP_ADDITIVITY_TOL,
    additivity_gate_ok,
    additivity_report,
    build_background,
    explain_shap,
    raw_shap_values,
)
from xaibench.models.cells import build_cell


@pytest.mark.parametrize("model_id", MODELS)
def test_attributions_have_the_shared_contract_for_every_backend(model_id):
    cell = build_cell("pima", model_id, params={}, profile=QUICK)
    exps = explain_shap(cell)
    assert len(exps) == cell.instance_ids.size
    for exp in exps:
        assert exp.attributions.shape == (len(cell.feature_order),)
        assert np.isfinite(exp.attributions).all()


@pytest.mark.parametrize("model_id", MODELS)
def test_additivity_holds_within_tolerance(model_id):
    # The definitive check: base_value + sum(phi) must reproduce predict_proba.
    # Passing this for xgb/lgbm proves model_output='probability' actually took
    # effect, which cannot be established by reading the shap docs.
    cell = build_cell("pima", model_id, params={}, profile=QUICK)
    background = build_background(cell, QUICK.shap_background, 42)
    values, base = raw_shap_values(cell, cell.instance_ids, background, 42)
    report = additivity_report(cell, values, base, cell.instance_ids)
    assert report["additivity_median"] < SHAP_ADDITIVITY_TOL
    assert additivity_gate_ok(report)


@pytest.mark.parametrize("model_id", ["dtree", "rf", "xgb", "lgbm"])
def test_tree_backends_are_bit_identical_across_reruns(model_id):
    # The protocol declares tree SHAP deterministic a priori; this is that claim
    # under test, and it is what makes H2a degenerate for these four models.
    cell = build_cell("pima", model_id, params={}, profile=QUICK)
    a = explain_shap(cell, seed=1)
    b = explain_shap(cell, seed=999)      # a different seed must change nothing
    for ea, eb in zip(a, b):
        np.testing.assert_array_equal(ea.attributions, eb.attributions)


def test_permutation_backend_is_stochastic_so_h2a_is_not_vacuous_everywhere():
    cell = build_cell("pima", "stack", params={}, profile=QUICK)
    a = explain_shap(cell, seed=1)
    b = explain_shap(cell, seed=2)
    diffs = [np.abs(ea.attributions - eb.attributions).max() for ea, eb in zip(a, b)]
    assert max(diffs) > 0.0


def test_attributions_are_aggregated_to_original_features_on_expanded_data():
    cell = build_cell("heart", "rf", params={}, profile=QUICK)
    assert len(cell.groups) > len(cell.feature_order)      # one-hot expanded
    for exp in explain_shap(cell):
        assert exp.attributions.shape == (len(cell.feature_order),)
        assert exp.feature_names == tuple(cell.feature_order)


def test_background_is_a_subsample_of_real_training_rows():
    # Real rows, not k-means centroids: a centroid averages one-hot columns into
    # fractional values the model never sees, and interventional TreeSHAP evaluated
    # against such off-manifold references violates its own additivity by ~1% on
    # LightGBM. See protocol amendment A2.
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    bg = build_background(cell, k=15, seed=42)
    assert bg.shape[1] == cell.X_train_t.shape[1]
    assert 14 <= bg.shape[0] <= 16
    np.testing.assert_allclose(bg, build_background(cell, k=15, seed=42))
    train_rows = {tuple(np.round(r, 12)) for r in cell.X_train_t}
    for row in bg:
        assert tuple(np.round(row, 12)) in train_rows, "background row is not real data"


def test_additivity_holds_for_lightgbm_on_every_dataset():
    # The cell that exposed the centroid problem. Heart has 8 categorical features
    # (25 one-hot columns), so its centroids were the most fractional and the most
    # wrong. This test is the regression guard.
    for dataset in ("pima", "heart", "ckd"):
        cell = build_cell(dataset, "lgbm", params={}, profile=QUICK)
        bg = build_background(cell, QUICK.shap_background, 42)
        values, base = raw_shap_values(cell, cell.instance_ids, bg, 42)
        report = additivity_report(cell, values, base, cell.instance_ids)
        assert report["additivity_median"] < SHAP_ADDITIVITY_TOL, dataset
        assert additivity_gate_ok(report), f"{dataset}: {report}"


def test_background_falls_back_when_k_exceeds_the_training_rows():
    cell = build_cell("heart", "rf", params={}, profile=QUICK)
    bg = build_background(cell, k=10_000, seed=42)
    assert bg.shape[0] == cell.X_train_t.shape[0]


def test_runtime_is_recorded_and_positive():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    assert all(exp.runtime_s > 0 for exp in explain_shap(cell))


def test_explainer_label_and_config_hash_are_set():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    exp = explain_shap(cell)[0]
    assert exp.explainer == "shap"
    assert len(exp.config_hash) == 16


def test_explaining_a_subset_returns_only_that_subset():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    subset = cell.instance_ids[:3]
    exps = explain_shap(cell, subset)
    assert [e.instance_id for e in exps] == [int(i) for i in subset]
