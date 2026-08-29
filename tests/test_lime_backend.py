import numpy as np
import pytest

from xaibench.config import MODELS, QUICK
from xaibench.explain.lime_backend import explain_lime
from xaibench.models.cells import build_cell


@pytest.mark.parametrize("model_id", MODELS)
def test_contract_matches_shap_so_metrics_need_no_special_cases(model_id):
    cell = build_cell("pima", model_id, params={}, profile=QUICK)
    exps = explain_lime(cell, num_samples=200, seed=42)
    assert len(exps) == cell.instance_ids.size
    for exp in exps:
        assert exp.explainer == "lime"
        assert exp.attributions.shape == (len(cell.feature_order),)
        assert exp.feature_names == tuple(cell.feature_order)
        assert np.isfinite(exp.attributions).all()
        assert exp.runtime_s > 0


def test_same_seed_reproduces_and_different_seed_does_not():
    # LIME is sampling-based, which is the whole substance of H2a. Both halves of
    # this test matter: reproducible under a fixed seed, variable across seeds.
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    a = explain_lime(cell, num_samples=200, seed=7)
    b = explain_lime(cell, num_samples=200, seed=7)
    c = explain_lime(cell, num_samples=200, seed=8)
    for ea, eb in zip(a, b):
        np.testing.assert_allclose(ea.attributions, eb.attributions)
    assert max(np.abs(ea.attributions - ec.attributions).max()
               for ea, ec in zip(a, c)) > 0.0


def test_variance_shrinks_as_num_samples_grows():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    inst = cell.instance_ids[:2]

    def spread(n):
        runs = np.array([
            explain_lime(cell, inst, num_samples=n, seed=1000 + r)[0].attributions
            for r in range(6)])
        return float(runs.std(axis=0).mean())

    assert spread(2000) < spread(100)


def test_attributions_aggregate_to_original_features_on_expanded_data():
    cell = build_cell("heart", "rf", params={}, profile=QUICK)
    assert len(cell.groups) > len(cell.feature_order)
    for exp in explain_lime(cell, num_samples=200, seed=42):
        assert exp.attributions.shape == (len(cell.feature_order),)


def test_config_hash_distinguishes_num_samples_for_provenance():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    a = explain_lime(cell, cell.instance_ids[:1], num_samples=100, seed=1)[0]
    b = explain_lime(cell, cell.instance_ids[:1], num_samples=200, seed=1)[0]
    assert a.config_hash != b.config_hash


def test_explaining_a_subset_returns_only_that_subset():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    subset = cell.instance_ids[:3]
    exps = explain_lime(cell, subset, num_samples=100, seed=1)
    assert [e.instance_id for e in exps] == [int(i) for i in subset]


def test_every_feature_receives_a_weight_so_rankings_are_complete():
    # num_features must cover all columns; a truncated explanation would leave
    # zeros that rank arbitrarily and quietly corrupt every agreement metric.
    cell = build_cell("ckd", "lgbm", params={}, profile=QUICK)
    exp = explain_lime(cell, cell.instance_ids[:1], num_samples=300, seed=3)[0]
    assert np.count_nonzero(exp.attributions) >= len(cell.feature_order) - 2
