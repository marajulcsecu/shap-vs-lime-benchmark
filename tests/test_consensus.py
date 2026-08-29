import numpy as np
import pytest

from xaibench.config import CALIB_FRACTION, QUICK
from xaibench.explain.base import Explanation
from xaibench.explain.consensus import (
    build_consensus,
    calibration_split,
    fidelity_weight,
    fuse_rankavg,
    fuse_rrf,
    fuse_weighted,
    normalised_ranks,
    random_control,
)
from xaibench.metrics.agreement import ranking_from_attributions
from xaibench.models.cells import build_cell

A = np.array([0.9, -0.5, 0.1, 0.05])     # order: 0, 1, 2, 3
B = np.array([0.1, -0.8, 0.7, 0.02])     # order: 1, 2, 0, 3


def test_normalised_ranks_put_the_most_important_feature_at_zero():
    r = normalised_ranks(A)
    assert r[0] == pytest.approx(0.0)
    assert r[3] == pytest.approx(1.0)
    assert np.all((r >= 0) & (r <= 1))


def test_fusing_a_ranking_with_itself_is_the_identity():
    for fuse in (fuse_rankavg, fuse_rrf):
        np.testing.assert_array_equal(ranking_from_attributions(fuse(A, A)),
                                      ranking_from_attributions(A))


def test_extreme_weights_recover_each_base_explainer_exactly():
    np.testing.assert_array_equal(ranking_from_attributions(fuse_weighted(A, B, w=1.0)),
                                  ranking_from_attributions(A))
    np.testing.assert_array_equal(ranking_from_attributions(fuse_weighted(A, B, w=0.0)),
                                  ranking_from_attributions(B))


def test_equal_weight_matches_plain_rank_averaging():
    np.testing.assert_allclose(fuse_weighted(A, B, w=0.5), fuse_rankavg(A, B))


def test_fusion_lands_between_the_two_inputs_on_a_disputed_feature():
    fused = ranking_from_attributions(fuse_rankavg(A, B)).tolist()
    assert fused.index(2) <= max(ranking_from_attributions(A).tolist().index(2),
                                 ranking_from_attributions(B).tolist().index(2))


def test_rrf_is_more_tolerant_of_one_bad_ranking_than_mean_rank():
    # A feature ranked 1st by one method and last by the other: RRF's 1/(k+r) keeps
    # it no lower than the arithmetic mean of ranks does.
    a = np.array([1.0, 0.1, 0.05, 0.01])
    b = np.array([0.01, 0.9, 0.5, 0.2])
    pos_rrf = ranking_from_attributions(fuse_rrf(a, b)).tolist().index(0)
    pos_avg = ranking_from_attributions(fuse_rankavg(a, b)).tolist().index(0)
    assert pos_rrf <= pos_avg


def test_calibration_and_evaluation_sets_are_disjoint_and_exhaustive():
    ids = np.arange(100)
    calib, evaluation = calibration_split(ids, seed=42)
    assert set(calib) | set(evaluation) == set(ids)
    assert not (set(calib) & set(evaluation))
    assert len(calib) == pytest.approx(100 * CALIB_FRACTION, abs=1)


def test_calibration_split_is_reproducible():
    a = calibration_split(np.arange(50), seed=1)
    b = calibration_split(np.arange(50), seed=1)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_fidelity_weight_is_a_proportion_and_handles_degenerate_input():
    assert fidelity_weight(0.3, 0.1) == pytest.approx(0.75)
    assert fidelity_weight(0.0, 0.0) == pytest.approx(0.5)     # no signal -> unweighted
    assert 0.0 <= fidelity_weight(-0.2, 0.4) <= 1.0            # clipped, not negative


def test_build_consensus_preserves_instance_alignment():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    names = tuple(cell.feature_order)
    d = len(names)
    rng = np.random.default_rng(0)
    shap_exps = [Explanation("pima", "rf", "shap", int(i), rng.normal(size=d), names,
                             0.01, 42, "h1") for i in cell.instance_ids]
    lime_exps = [Explanation("pima", "rf", "lime", int(i), rng.normal(size=d), names,
                             0.02, 42, "h2") for i in cell.instance_ids]
    fused = build_consensus(shap_exps, lime_exps, weight=0.6, rule="fidwt")
    assert [e.instance_id for e in fused] == [e.instance_id for e in shap_exps]
    assert all(e.explainer == "fidwt" for e in fused)
    # Cost is the sum of both base explainers: fusion is not free.
    assert fused[0].runtime_s == pytest.approx(0.03)


def test_build_consensus_rejects_misaligned_inputs():
    names = ("a", "b")
    shap_exps = [Explanation("d", "m", "shap", 0, np.array([1.0, 0.0]), names, 0.0, 1, "h")]
    lime_exps = [Explanation("d", "m", "lime", 9, np.array([0.0, 1.0]), names, 0.0, 1, "h")]
    with pytest.raises(ValueError):
        build_consensus(shap_exps, lime_exps, weight=0.5, rule="rankavg")


def test_random_control_is_reproducible_and_uninformative():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    a = random_control(cell, cell.instance_ids, seed=42)
    b = random_control(cell, cell.instance_ids, seed=42)
    for ea, eb in zip(a, b):
        np.testing.assert_allclose(ea.attributions, eb.attributions)
    assert all(e.explainer == "random" for e in a)
    # Across instances the rankings must differ, or it is not a random control.
    assert len({tuple(e.ranking) for e in a}) > 1


def test_unknown_fusion_rule_is_rejected():
    names = ("a", "b")
    exps = [Explanation("d", "m", "shap", 0, np.array([1.0, 0.0]), names, 0.0, 1, "h")]
    with pytest.raises(ValueError):
        build_consensus(exps, exps, weight=0.5, rule="not-a-rule")
