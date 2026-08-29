import numpy as np
import pytest

from xaibench.metrics.agreement import (
    jaccard_at_k,
    kendall_tau_b,
    rank_vector,
    ranking_from_attributions,
    rbo,
    rbo_floor,
    rbo_normalised,
    spearman,
    topk_set,
)


def test_ranking_uses_absolute_value_so_sign_does_not_reorder():
    attr = np.array([0.1, -0.9, 0.5, -0.2])
    np.testing.assert_array_equal(ranking_from_attributions(attr), [1, 2, 3, 0])


def test_ranking_breaks_ties_by_index_for_determinism():
    attr = np.array([0.5, 0.5, 0.5])
    np.testing.assert_array_equal(ranking_from_attributions(attr), [0, 1, 2])


def test_identical_rankings_score_maximum_on_every_metric():
    order = [3, 0, 2, 1, 4]
    r = rank_vector(order)
    assert spearman(r, r) == pytest.approx(1.0)
    assert kendall_tau_b(r, r) == pytest.approx(1.0)
    assert jaccard_at_k(order, order, 3) == pytest.approx(1.0)
    assert rbo(order, order, p=0.9) == pytest.approx(1.0)


def test_reversed_rankings_are_perfectly_discordant():
    order = [0, 1, 2, 3, 4]
    rev = order[::-1]
    assert spearman(rank_vector(order), rank_vector(rev)) == pytest.approx(-1.0)
    assert kendall_tau_b(rank_vector(order), rank_vector(rev)) == pytest.approx(-1.0)
    assert jaccard_at_k(order, rev, 2) == pytest.approx(0.0)
    # Raw RBO does NOT go near zero for a reversed ranking: both lists are
    # permutations of the same set, so the extrapolation term p**k is a hard floor.
    assert rbo(order, rev, p=0.9) == pytest.approx(rbo_floor(5, 0.9))
    assert rbo_normalised(order, rev, p=0.9) == pytest.approx(0.0)
    assert rbo_normalised(order, order, p=0.9) == pytest.approx(1.0)


def test_jaccard_at_k_on_hand_computed_sets():
    a = [0, 1, 2, 3, 4]
    b = [0, 1, 5, 6, 7]
    # top-3: {0,1,2} vs {0,1,5} -> intersection 2, union 4
    assert jaccard_at_k(a, b, 3) == pytest.approx(2 / 4)
    # top-5: {0,1,2,3,4} vs {0,1,5,6,7} -> intersection 2, union 8
    assert jaccard_at_k(a, b, 5) == pytest.approx(2 / 8)


def test_jaccard_k_larger_than_list_is_clipped_not_an_error():
    assert jaccard_at_k([0, 1], [1, 0], 5) == pytest.approx(1.0)


def test_rbo_matches_a_hand_computed_three_item_case():
    # S=[a,b,c], T=[a,c,b] with p=0.9.
    # A_1 = 1, A_2 = 1/2, A_3 = 1
    # RBO_ext = (1-p)(p^0*1 + p^1*0.5 + p^2*1) + p^3 * A_3
    #         = 0.1*(1 + 0.45 + 0.81) + 0.729 = 0.226 + 0.729 = 0.955
    assert rbo([0, 1, 2], [0, 2, 1], p=0.9) == pytest.approx(0.955, abs=1e-9)


def test_rbo_is_top_weighted_unlike_spearman():
    # Swapping ranks 1 and 2 must cost more than swapping ranks 5 and 6.
    base = [0, 1, 2, 3, 4, 5, 6, 7]
    swap_head = [1, 0, 2, 3, 4, 5, 6, 7]
    swap_tail = [0, 1, 2, 3, 5, 4, 6, 7]
    assert rbo(base, swap_head, p=0.9) < rbo(base, swap_tail, p=0.9)


def test_rbo_is_symmetric_and_bounded():
    rng = np.random.default_rng(3)
    for _ in range(20):
        a = list(rng.permutation(8))
        b = list(rng.permutation(8))
        value = rbo(a, b, p=0.9)
        assert 0.0 <= value <= 1.0
        assert value == pytest.approx(rbo(b, a, p=0.9))


def test_rbo_overlap_counter_tracks_the_true_prefix_intersection():
    # Guards the incremental counter directly: at every depth d it must equal
    # |top-d(a) ∩ top-d(b)|. A drift here silently biases every RBO in the study.
    from xaibench.metrics.agreement import _prefix_overlaps
    rng = np.random.default_rng(11)
    for _ in range(30):
        a = list(rng.permutation(9))
        b = list(rng.permutation(9))
        got = _prefix_overlaps(a, b)
        want = [len(set(a[:d]) & set(b[:d])) for d in range(1, len(a) + 1)]
        assert got == want


def test_spearman_handles_constant_input_without_nan():
    # A degenerate explanation (all attributions equal) must not poison aggregation.
    assert spearman(np.zeros(5), np.arange(5.0)) == pytest.approx(0.0)


def test_topk_set_returns_first_k_indices():
    assert topk_set([4, 2, 7, 1], 2) == {4, 2}


def test_raw_rbo_floor_is_width_dependent_but_normalised_rbo_is_not():
    # The reason rbo_normalised exists: PIMA (8 features) cannot score below 0.589
    # while CKD (24) reaches 0.193, so raw cross-dataset RBO comparison is invalid.
    assert rbo_floor(8, 0.9) > rbo_floor(24, 0.9)
    for d in (8, 13, 24):
        order = list(range(d))
        assert rbo_normalised(order, order[::-1], p=0.9) == pytest.approx(0.0)
        assert rbo_normalised(order, order, p=0.9) == pytest.approx(1.0)


def test_normalised_rbo_stays_top_weighted():
    base = list(range(10))
    head = [1, 0] + base[2:]
    tail = base[:8] + [9, 8]
    assert rbo_normalised(base, head, p=0.9) < rbo_normalised(base, tail, p=0.9)
