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


def test_grouped_deletion_removes_whole_feature_groups_not_single_columns():
    # With one-hot encoding, "delete the top-3 features" must delete every column
    # belonging to those three features. Deleting 3 *columns* instead would silently
    # under-perturb every categorical dataset, making explanations look less faithful
    # than they are on Heart (13 features spanning 25 columns) but not on PIMA.
    from xaibench.metrics.fidelity import deletion_curve_grouped

    coef = np.array([5.0, 5.0, 1.0, 0.0])   # feature A spans columns 0 and 1

    def proba(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        p1 = 1.0 / (1.0 + np.exp(-(X @ coef)))
        return np.column_stack([1.0 - p1, p1])

    x = np.ones(4)
    groups = [np.array([0, 1]), np.array([2]), np.array([3])]
    curve = deletion_curve_grouped(proba, x, groups, np.zeros(4), k_max=3)
    assert len(curve) == 4
    # Deleting group 0 removes BOTH columns, so the logit drops 10, not 5.
    assert curve[1] == pytest.approx(proba(np.array([0.0, 0.0, 1.0, 1.0]))[0, 1])
    assert np.all(np.diff(curve) <= 1e-12)


def test_grouped_curves_agree_with_ungrouped_when_every_group_is_one_column():
    from xaibench.metrics.fidelity import deletion_curve_grouped, insertion_curve_grouped
    groups = [np.array([i]) for i in [0, 1, 2]]
    np.testing.assert_allclose(
        deletion_curve_grouped(linear_proba, X_INSTANCE, groups, BASELINE, k_max=3),
        deletion_curve(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k_max=3))
    np.testing.assert_allclose(
        insertion_curve_grouped(linear_proba, X_INSTANCE, groups, BASELINE, k_max=3),
        insertion_curve(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k_max=3))


def test_batched_metrics_equal_the_one_call_at_a_time_version():
    # The batched path is a pure optimisation: one predict_proba call for every
    # baseline and both curves instead of 2 per baseline. It must be bit-comparable.
    from xaibench.metrics.fidelity import batched_curve_metrics

    groups = [np.array([i]) for i in [0, 1, 2]]
    baselines = [BASELINE, np.array([0.5, 0.5, 0.5]), np.array([-1.0, 0.0, 1.0])]
    batched = batched_curve_metrics(linear_proba, X_INSTANCE, groups, baselines,
                                   k_max=3, k=2)
    from xaibench.metrics.fidelity import (
        curve_metrics, deletion_curve_grouped, insertion_curve_grouped,
    )
    individually = [
        curve_metrics(
            deletion_curve_grouped(linear_proba, X_INSTANCE, groups, b, 3),
            insertion_curve_grouped(linear_proba, X_INSTANCE, groups, b, 3), k=2)
        for b in baselines]
    expected = {key: float(np.mean([m[key] for m in individually]))
                for key in individually[0]}
    for key, value in expected.items():
        assert batched[key] == pytest.approx(value), key


def test_batched_metrics_make_exactly_one_model_call():
    from xaibench.metrics.fidelity import batched_curve_metrics
    calls = []

    def counting_proba(X):
        calls.append(np.atleast_2d(X).shape[0])
        return linear_proba(X)

    groups = [np.array([i]) for i in [0, 1, 2]]
    baselines = [BASELINE, np.array([0.5, 0.5, 0.5])]
    batched_curve_metrics(counting_proba, X_INSTANCE, groups, baselines, k_max=3, k=2)
    # One call for the target class plus one batched call for every curve.
    assert len(calls) <= 2, calls
    assert max(calls) == 2 * len(baselines) * 4
