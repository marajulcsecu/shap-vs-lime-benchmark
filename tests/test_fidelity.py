import itertools

import numpy as np
import pytest

from xaibench.metrics.fidelity import (
    aopc_deletion,
    auc_insertion,
    comprehensiveness,
    deletion_curve,
    insertion_curve,
    marginal_baselines,
    median_baseline,
    sufficiency,
)

COEF = np.array([3.0, 2.0, 0.0])


def linear_proba(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    p1 = 1.0 / (1.0 + np.exp(-(X @ COEF)))
    return np.column_stack([1.0 - p1, p1])


X_INSTANCE = np.array([1.0, 1.0, 1.0])
BASELINE = np.zeros(3)


def test_correct_ranking_maximises_aopc_over_all_permutations():
    scores = {perm: aopc_deletion(linear_proba, X_INSTANCE, list(perm), BASELINE, k_max=3)
              for perm in itertools.permutations(range(3))}
    assert max(scores, key=scores.get) == (0, 1, 2)
    assert min(scores, key=scores.get) == (2, 1, 0)


def test_deletion_curve_starts_at_the_undeleted_prediction():
    curve = deletion_curve(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k_max=3)
    assert len(curve) == 4
    assert curve[0] == pytest.approx(linear_proba(X_INSTANCE)[0, 1])


def test_deletion_curve_is_monotone_for_an_additive_positive_model():
    curve = deletion_curve(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k_max=3)
    assert np.all(np.diff(curve) <= 1e-12)


def test_deleting_a_zero_coefficient_feature_changes_nothing():
    curve = deletion_curve(linear_proba, X_INSTANCE, [2, 0, 1], BASELINE, k_max=1)
    assert curve[1] == pytest.approx(curve[0])


def test_insertion_curve_ends_at_the_full_prediction():
    curve = insertion_curve(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k_max=3)
    assert curve[0] == pytest.approx(linear_proba(BASELINE)[0, 1])
    assert curve[-1] == pytest.approx(linear_proba(X_INSTANCE)[0, 1])


def test_auc_insertion_prefers_the_correct_ranking():
    good = auc_insertion(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k_max=3)
    bad = auc_insertion(linear_proba, X_INSTANCE, [2, 1, 0], BASELINE, k_max=3)
    assert good > bad


def test_metrics_track_the_models_own_predicted_class_not_class_one():
    # For this instance the model predicts class 0, so a faithful deletion must
    # reduce p(class 0). Anything hard-coded to column 1 gets this backwards.
    x_neg = np.array([-1.0, -1.0, 0.0])
    assert linear_proba(x_neg)[0].argmax() == 0
    curve = deletion_curve(linear_proba, x_neg, [0, 1, 2], BASELINE, k_max=3)
    assert curve[0] == pytest.approx(linear_proba(x_neg)[0, 0])
    assert curve[-1] < curve[0]


def test_comprehensiveness_and_sufficiency_have_the_expected_signs():
    comp = comprehensiveness(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k=2)
    suff = sufficiency(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k=2)
    assert comp > 0.0                    # removing the top-2 must hurt
    assert abs(suff) < abs(comp)         # the top-2 alone nearly suffice here


def test_median_baseline_ignores_nan_and_matches_numpy():
    X = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]])
    np.testing.assert_allclose(median_baseline(X), [3.0, 20.0])


def test_marginal_baselines_are_drawn_from_observed_values_only():
    X = np.array([[1.0], [2.0], [3.0]])
    draws = marginal_baselines(X, n_draws=8, seed=0)
    assert draws.shape == (8, 1)
    assert set(np.unique(draws)).issubset({1.0, 2.0, 3.0})


def test_marginal_baselines_are_reproducible_from_the_seed():
    X = np.arange(20.0).reshape(-1, 1)
    np.testing.assert_allclose(marginal_baselines(X, 5, seed=7),
                               marginal_baselines(X, 5, seed=7))


def test_k_max_is_clipped_to_the_feature_count():
    curve = deletion_curve(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k_max=99)
    assert len(curve) == 4


def test_random_ranking_scores_lower_aopc_than_the_true_one_on_average():
    # The property the random control relies on. Without it, an AOPC value has no
    # scale and "SHAP achieved 0.31" cannot be interpreted.
    rng = np.random.default_rng(0)
    true_score = aopc_deletion(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k_max=3)
    random_scores = [aopc_deletion(linear_proba, X_INSTANCE, list(rng.permutation(3)),
                                   BASELINE, k_max=3) for _ in range(40)]
    assert true_score > np.mean(random_scores)
