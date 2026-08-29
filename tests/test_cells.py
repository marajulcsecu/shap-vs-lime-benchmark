import numpy as np
import pytest

from xaibench.config import QUICK
from xaibench.explain.base import Explanation, config_hash, explanations_to_frame
from xaibench.models.cells import build_cell


def _explanation(**kw):
    base = dict(dataset="pima", model="rf", explainer="shap", instance_id=0,
                attributions=np.array([0.3, -0.7, 0.1]),
                feature_names=("a", "b", "c"), runtime_s=0.01, seed=42,
                config_hash="deadbeef")
    return Explanation(**{**base, **kw})


def test_ranking_orders_by_absolute_attribution():
    np.testing.assert_array_equal(_explanation().ranking, [1, 0, 2])


def test_explanation_rejects_a_length_mismatch_immediately():
    with pytest.raises(ValueError):
        _explanation(attributions=np.array([0.1, 0.2]))


def test_frame_is_long_with_one_row_per_feature():
    df = explanations_to_frame([_explanation(), _explanation(instance_id=1)])
    assert len(df) == 6
    assert set(df.columns) == {"dataset", "model", "explainer", "instance_id",
                               "feature", "attribution", "rank", "runtime_s",
                               "seed", "config_hash"}
    assert df.loc[df.instance_id == 0, "rank"].tolist() == [1, 0, 2]


def test_config_hash_is_stable_and_order_independent():
    assert config_hash(a=1, b="x") == config_hash(b="x", a=1)
    assert config_hash(a=1) != config_hash(a=2)
    assert len(config_hash(a=1)) == 16


@pytest.mark.parametrize("model_id", ["logreg", "rf", "xgb", "lgbm", "stack", "dtree"])
def test_cell_builds_for_every_model_with_a_consistent_contract(model_id):
    cell = build_cell("pima", model_id, params={}, profile=QUICK)
    assert cell.X_train_t.shape[1] == cell.X_test_t.shape[1] == len(cell.groups)
    assert len(cell.feature_order) == 8                       # pima has 8 features
    assert set(cell.groups) == set(cell.feature_order)
    assert cell.instance_ids.size == min(QUICK.n_explain, cell.X_test_t.shape[0])
    proba = cell.predict_proba(cell.X_test_t[cell.instance_ids])
    assert proba.shape == (cell.instance_ids.size, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_test_partition_is_twenty_percent_and_stratified():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    assert cell.X_test_t.shape[0] == 154                      # ceil(768 * 0.2)
    assert abs(cell.y_test.mean() - 0.349) < 0.05


def test_cell_is_reproducible_from_the_seed():
    a = build_cell("heart", "rf", params={}, profile=QUICK)
    b = build_cell("heart", "rf", params={}, profile=QUICK)
    np.testing.assert_allclose(a.X_test_t, b.X_test_t)
    np.testing.assert_array_equal(a.instance_ids, b.instance_ids)
    np.testing.assert_allclose(a.predict_proba(a.X_test_t), b.predict_proba(b.X_test_t))


def test_explained_instances_are_class_stratified():
    cell = build_cell("ckd", "rf", params={}, profile=QUICK)
    chosen = cell.y_test[cell.instance_ids]
    assert 0 in set(chosen) and 1 in set(chosen)


def test_params_are_applied_to_the_classifier():
    cell = build_cell("pima", "rf", params={"max_depth": 3}, profile=QUICK)
    assert cell.classifier.get_params()["max_depth"] == 3


def test_fixed_mask_flags_exactly_the_one_hot_columns():
    cell = build_cell("heart", "rf", params={}, profile=QUICK)
    assert cell.fixed_mask.sum() == len(cell.categorical_idx)
    assert cell.fixed_mask.size == len(cell.groups)
    assert not cell.fixed_mask[:len(cell.feature_order) - len(set(
        g for i, g in enumerate(cell.groups) if cell.fixed_mask[i]))].all()


def test_replace_swaps_the_test_matrix_so_neighbours_can_be_explained_by_index():
    # RQ2b explains perturbed neighbours and RQ2c explains refit classifiers.
    # Cell is frozen, so dataclasses.replace is the clean way to do both.
    from dataclasses import replace
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    extra = np.vstack([cell.X_test_t, cell.X_test_t[:3] + 0.01])
    probe = replace(cell, X_test_t=extra)
    assert probe.X_test_t.shape[0] == cell.X_test_t.shape[0] + 3
    assert probe.classifier is cell.classifier
    assert probe.groups == cell.groups
