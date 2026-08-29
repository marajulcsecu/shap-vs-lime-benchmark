import numpy as np
import pytest

from xaibench.data.loaders import load_dataset

EXPECTED = {   # (n_rows, n_features, positive_rate) -- NORMATIVE, protocol Table T1
    "pima":  (768,  8, 0.349),
    "heart": (303, 13, 0.459),
    "ckd":   (400, 24, 0.625),
}


@pytest.mark.parametrize("ds_id", ["pima", "heart", "ckd"])
def test_shape_and_balance_match_protocol(ds_id):
    n, d, rate = EXPECTED[ds_id]
    ds = load_dataset(ds_id)
    assert ds.X.shape == (n, d)
    assert len(ds.y) == n
    assert ds.y.isin([0, 1]).all()
    assert ds.positive_rate == pytest.approx(rate, abs=0.005)


@pytest.mark.parametrize("ds_id", ["pima", "heart", "ckd"])
def test_column_lists_partition_the_frame(ds_id):
    ds = load_dataset(ds_id)
    assert set(ds.numeric) | set(ds.categorical) == set(ds.X.columns)
    assert not (set(ds.numeric) & set(ds.categorical))
    for c in ds.numeric:
        assert np.issubdtype(ds.X[c].dtype, np.number), c


def test_pima_impossible_zeros_became_missing():
    ds = load_dataset("pima")
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        assert (ds.X[col] == 0).sum() == 0, f"{col} still has impossible zeros"
        assert ds.X[col].isna().sum() > 0, f"{col} should now carry NaNs"
    # Pregnancies zeros are legitimate and must survive
    assert (ds.X["Pregnancies"] == 0).sum() > 0


def test_heart_target_is_binarised_from_five_levels():
    ds = load_dataset("heart")
    assert "num" not in ds.X.columns
    assert set(ds.y.unique()) == {0, 1}


def test_ckd_object_columns_are_stripped_of_stray_whitespace():
    ds = load_dataset("ckd")
    assert ds.categorical, "ckd must have categorical columns"
    for c in ds.categorical:
        vals = ds.X[c].dropna().astype(str)
        assert (vals == vals.str.strip()).all(), f"{c} carries untrimmed values"
        assert not vals.str.contains("\t").any()


@pytest.mark.parametrize("ds_id", ["pima", "heart", "ckd"])
def test_metadata_is_present_for_the_paper(ds_id):
    ds = load_dataset(ds_id)
    assert ds.source_url.startswith("https://")
    assert ds.licence and ds.citation
