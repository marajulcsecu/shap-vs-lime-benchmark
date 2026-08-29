import pytest

from xaibench.config import DATASETS
from xaibench.data.loaders import load_dataset
from xaibench.metrics.plausibility import (
    jaccard_vs_reference,
    load_reference_sets,
    precision_at_k,
)


def test_every_dataset_has_a_reference_set_with_citations():
    refs = load_reference_sets()
    assert set(refs) == set(DATASETS)
    for ds_id, ref in refs.items():
        assert len(ref.features) == 5, f"{ds_id}: protocol fixes |R_D| = 5"
        assert len(ref.citations) >= 2, f"{ds_id}: needs at least two sources"
        assert ref.note.strip()


def test_reference_features_all_exist_in_their_dataset():
    # A typo here would silently score zero overlap for the rest of the study and
    # look like a finding.
    refs = load_reference_sets()
    for ds_id, ref in refs.items():
        columns = set(load_dataset(ds_id).X.columns)
        missing = set(ref.features) - columns
        assert not missing, f"{ds_id}: {missing} not in the dataset columns"


def test_perfect_and_zero_overlap_score_as_expected():
    order = ["a", "b", "c", "d", "e", "f"]
    ranking = list(range(6))
    assert jaccard_vs_reference(ranking, order, ["a", "b", "c", "d", "e"], k=5) == pytest.approx(1.0)
    assert precision_at_k(ranking, order, ["a", "b", "c", "d", "e"], k=5) == pytest.approx(1.0)
    assert jaccard_vs_reference(ranking, order, ["f"], k=5) == pytest.approx(0.0)


def test_partial_overlap_matches_hand_computation():
    order = ["a", "b", "c", "d", "e"]
    ranking = [0, 1, 2, 3, 4]          # top-3 = {a, b, c}
    reference = ["c", "d", "e"]
    # intersection {c} = 1, union {a,b,c,d,e} = 5
    assert jaccard_vs_reference(ranking, order, reference, k=3) == pytest.approx(1 / 5)
    assert precision_at_k(ranking, order, reference, k=3) == pytest.approx(1 / 3)


def test_a_reference_feature_absent_from_the_ranking_is_ignored_not_an_error():
    assert jaccard_vs_reference([0, 1], ["a", "b"], ["a", "zzz"], k=2) == pytest.approx(1 / 2)
