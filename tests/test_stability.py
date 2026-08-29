import numpy as np
import pytest

from xaibench.metrics.stability import (
    local_lipschitz,
    repeat_pair_rho,
    rerun_cv,
    retrain_agreement,
    sample_neighbours,
)


def test_a_deterministic_explainer_scores_exactly_zero_variance():
    # The protocol's a-priori expectation for tree SHAP, asserted here so the claim
    # is a tested property of the metric rather than a hope about the backend.
    repeats = np.tile(np.array([0.4, -0.2, 0.1]), (10, 1))
    assert rerun_cv(repeats) == pytest.approx(0.0)
    assert repeat_pair_rho(repeats) == pytest.approx(1.0)


def test_rerun_cv_is_positive_when_repeats_differ():
    rng = np.random.default_rng(0)
    repeats = np.array([0.4, -0.2, 0.1]) + rng.normal(scale=0.05, size=(10, 3))
    assert rerun_cv(repeats) > 0.0


def test_rerun_cv_grows_with_noise():
    rng = np.random.default_rng(1)
    base = np.array([0.4, -0.2, 0.1])
    low = base + rng.normal(scale=0.01, size=(20, 3))
    high = base + rng.normal(scale=0.20, size=(20, 3))
    assert rerun_cv(low) < rerun_cv(high)


def test_rerun_cv_survives_a_zero_mean_feature_without_dividing_by_zero():
    repeats = np.zeros((5, 3))
    repeats[:, 0] = [1.0, -1.0, 1.0, -1.0, 1.0]
    assert np.isfinite(rerun_cv(repeats))


def test_repeat_pair_rho_falls_when_orderings_disagree():
    stable = np.tile([3.0, 2.0, 1.0], (6, 1))
    shuffled = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]] * 3)
    assert repeat_pair_rho(shuffled) < repeat_pair_rho(stable)


def test_lipschitz_is_zero_for_a_locally_constant_explainer():
    x = np.array([0.0, 0.0])
    nbrs = np.array([[0.1, 0.0], [0.0, 0.1]])
    phi_x = np.array([1.0, 2.0])
    phi_n = np.tile(phi_x, (2, 1))
    assert local_lipschitz(phi_x, x, phi_n, nbrs) == pytest.approx(0.0)


def test_lipschitz_equals_the_hand_computed_worst_case_ratio():
    x = np.array([0.0, 0.0])
    nbrs = np.array([[0.1, 0.0], [0.0, 0.2]])
    phi_x = np.array([0.0, 0.0])
    phi_n = np.array([[0.05, 0.0], [0.0, 0.02]])
    # ratios: 0.05/0.1 = 0.5 and 0.02/0.2 = 0.1 -> worst case 0.5
    assert local_lipschitz(phi_x, x, phi_n, nbrs) == pytest.approx(0.5)


def test_lipschitz_ignores_a_coincident_neighbour_instead_of_dividing_by_zero():
    x = np.array([1.0, 1.0])
    nbrs = np.array([[1.0, 1.0], [1.1, 1.0]])
    phi_x = np.array([1.0, 0.0])
    phi_n = np.array([[1.0, 0.0], [1.05, 0.0]])
    assert local_lipschitz(phi_x, x, phi_n, nbrs) == pytest.approx(0.5, abs=1e-9)


def test_neighbours_scale_with_sigma_and_hold_fixed_columns():
    x = np.array([0.0, 0.0, 5.0])
    sigma = np.array([1.0, 1.0, 1.0])
    fixed = np.array([False, False, True])
    nbrs = sample_neighbours(x, sigma * 0.1, n=200, seed=0, fixed_mask=fixed)
    assert nbrs.shape == (200, 3)
    assert np.all(nbrs[:, 2] == 5.0)                # categorical columns must not move
    assert nbrs[:, 0].std() == pytest.approx(0.1, abs=0.03)


def test_neighbours_are_reproducible_from_the_seed():
    x = np.zeros(4)
    sigma = np.ones(4)
    np.testing.assert_allclose(sample_neighbours(x, sigma, 10, seed=3),
                               sample_neighbours(x, sigma, 10, seed=3))


def test_retrain_agreement_is_one_for_identical_refits_and_lower_when_they_differ():
    same = np.tile([0.5, 0.3, 0.1], (5, 1))
    assert retrain_agreement(same) == pytest.approx(1.0)
    differ = np.array([[0.5, 0.3, 0.1], [0.1, 0.3, 0.5],
                       [0.5, 0.3, 0.1], [0.1, 0.3, 0.5]])
    assert retrain_agreement(differ) < 1.0


def test_bit_identical_repeats_give_exactly_zero_not_float_residue():
    # np.std on identical rows leaves ~1e-17 of cancellation residue, and dividing
    # by a small |mean| amplifies it. Reporting 8.3e-17 in a results table as though
    # it were measured variance would be noise dressed as a finding.
    repeats = np.tile(np.array([1e-9, -2e-9, 5e-10]), (10, 1))
    assert rerun_cv(repeats) == 0.0


def test_a_single_differing_repeat_still_registers_as_nonzero():
    repeats = np.tile(np.array([0.5, 0.2, 0.1]), (10, 1))
    repeats[3, 0] += 1e-6
    assert rerun_cv(repeats) > 0.0
