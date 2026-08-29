import numpy as np
import pytest

from xaibench.stats.tests import (
    bca_ci,
    cliffs_delta,
    holm,
    mannwhitney_with_effect,
    wilcoxon_with_effect,
)


def test_all_positive_differences_give_effect_size_plus_one():
    result = wilcoxon_with_effect(np.arange(1.0, 11.0))
    assert result.n == 10
    assert result.effect_size == pytest.approx(1.0)
    assert result.effect_name == "rank_biserial"
    assert result.p_value < 0.05


def test_all_negative_differences_give_effect_size_minus_one():
    assert wilcoxon_with_effect(-np.arange(1.0, 11.0)).effect_size == pytest.approx(-1.0)


def test_paired_form_equals_the_difference_form():
    a = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    b = np.array([1.0, 3.0, 2.0, 7.0, 4.0])
    assert (wilcoxon_with_effect(a, b).p_value
            == pytest.approx(wilcoxon_with_effect(a - b).p_value))


def test_mu_shifts_the_null_so_a_threshold_hypothesis_can_be_tested():
    # H1 tests rho > 0.50, i.e. Wilcoxon on (rho_i - 0.50).
    rho = np.array([0.55, 0.60, 0.58, 0.62, 0.57, 0.61])
    assert wilcoxon_with_effect(rho, mu=0.50, alternative="greater").p_value < 0.05
    assert wilcoxon_with_effect(rho, mu=0.90, alternative="greater").p_value > 0.5


def test_all_zero_differences_are_reported_degenerate_not_forced_through_a_test():
    # Expected for tree SHAP determinism. scipy raises here; we must not.
    result = wilcoxon_with_effect(np.zeros(8))
    assert result.degenerate is True
    assert result.p_value == 1.0
    assert result.effect_size == 0.0


def test_holm_matches_a_hand_computed_example():
    # p = [0.01, 0.04, 0.03], n = 3, sorted: 0.01(x3), 0.03(x2), 0.04(x1)
    # -> 0.03, 0.06, 0.06 with monotonicity enforced
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_holm_is_monotone_and_never_exceeds_one():
    p = np.random.default_rng(0).uniform(size=25)
    adjusted = holm(p)
    assert np.all(adjusted <= 1.0)
    assert np.all(np.diff(adjusted[np.argsort(p)]) >= -1e-12)


def test_holm_is_never_more_lenient_than_the_raw_p_value():
    p = [0.001, 0.02, 0.5]
    assert np.all(holm(p) >= np.asarray(p) - 1e-12)


def test_holm_of_a_single_test_is_the_identity():
    np.testing.assert_allclose(holm([0.023]), [0.023])


def test_bca_ci_brackets_a_known_mean():
    data = np.random.default_rng(0).normal(loc=5.0, scale=1.0, size=200)
    low, high = bca_ci(data, np.mean, n_resamples=800, seed=1)
    assert low < 5.0 < high
    assert high - low < 1.0


def test_bca_ci_is_reproducible_from_the_seed():
    data = np.random.default_rng(2).normal(size=80)
    assert bca_ci(data, np.mean, 400, seed=5) == bca_ci(data, np.mean, 400, seed=5)


def test_bca_ci_degenerates_gracefully_on_constant_data():
    low, high = bca_ci(np.full(30, 2.5), np.mean, 300, seed=0)
    assert low == pytest.approx(2.5) and high == pytest.approx(2.5)


def test_cliffs_delta_is_plus_one_when_a_strictly_dominates_b():
    assert cliffs_delta([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert cliffs_delta([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(-1.0)


def test_cliffs_delta_is_zero_for_identical_samples():
    x = [1.0, 2.0, 3.0, 4.0]
    assert cliffs_delta(x, x) == pytest.approx(0.0)


def test_mannwhitney_reports_cliffs_delta_as_its_effect_size():
    result = mannwhitney_with_effect([4.0, 5.0, 6.0, 7.0], [1.0, 2.0, 3.0, 3.5])
    assert result.effect_name == "cliffs_delta"
    assert result.effect_size == pytest.approx(1.0)
