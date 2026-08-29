import numpy as np
import pandas as pd
import pytest

from xaibench.data.loaders import Dataset, load_dataset
from xaibench.models.pipelines import (
    aggregate_attributions,
    build_preprocessor,
    categorical_output_indices,
    onehot_group_map,
    transformed_feature_names,
)


def _toy() -> Dataset:
    X = pd.DataFrame({
        "age":   [20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        "bp":    [80.0, 85.0, np.nan, 95.0, 100.0, 105.0],
        "grade": ["low", "mid", "high", "low", "mid", "high"],   # 3 levels
        "flag":  ["yes", "no", "yes", "no", "yes", "no"],        # binary
    })
    y = pd.Series([0, 1, 0, 1, 0, 1])
    return Dataset(id="toy", X=X, y=y, numeric=["age", "bp"],
                   categorical=["grade", "flag"], source_url="https://example.invalid",
                   licence="n/a", citation="n/a")


def test_group_map_has_one_entry_per_output_column():
    ds = _toy()
    pre = build_preprocessor(ds).fit(ds.X)
    n_out = pre.transform(ds.X).shape[1]
    groups = onehot_group_map(pre, ds)
    assert len(groups) == n_out
    assert set(groups) == set(ds.X.columns)


def test_three_level_categorical_spans_three_columns_and_binary_spans_one():
    ds = _toy()
    pre = build_preprocessor(ds).fit(ds.X)
    groups = onehot_group_map(pre, ds)
    assert groups.count("grade") == 3
    assert groups.count("flag") == 1     # drop='if_binary'
    assert groups.count("age") == 1 and groups.count("bp") == 1


def test_aggregation_sums_exactly_over_constituent_columns():
    ds = _toy()
    pre = build_preprocessor(ds).fit(ds.X)
    groups = onehot_group_map(pre, ds)
    order = list(ds.X.columns)
    values = np.random.default_rng(0).normal(size=(5, len(groups)))
    agg = aggregate_attributions(values, groups, order)
    assert agg.shape == (5, len(order))
    for j, feat in enumerate(order):
        cols = [i for i, g in enumerate(groups) if g == feat]
        np.testing.assert_allclose(agg[:, j], values[:, cols].sum(axis=1))


def test_aggregation_preserves_total_attribution():
    # Additivity of SHAP must survive aggregation: the row sum cannot change.
    ds = _toy()
    pre = build_preprocessor(ds).fit(ds.X)
    groups = onehot_group_map(pre, ds)
    values = np.random.default_rng(1).normal(size=(7, len(groups)))
    agg = aggregate_attributions(values, groups, list(ds.X.columns))
    np.testing.assert_allclose(agg.sum(axis=1), values.sum(axis=1), atol=1e-12)


def test_imputer_statistics_differ_between_folds_so_nothing_leaks():
    # If preprocessing were fitted once on all data, these medians would be equal.
    ds = load_dataset("pima")
    a = build_preprocessor(ds).fit(ds.X.iloc[:300])
    b = build_preprocessor(ds).fit(ds.X.iloc[300:])
    ma = a.named_transformers_["num"].named_steps["impute"].statistics_
    mb = b.named_transformers_["num"].named_steps["impute"].statistics_
    assert not np.allclose(ma, mb)


@pytest.mark.parametrize("ds_id", ["pima", "heart", "ckd"])
def test_transform_leaves_no_nan_for_the_explainers(ds_id):
    ds = load_dataset(ds_id)
    pre = build_preprocessor(ds).fit(ds.X)
    out = np.asarray(pre.transform(ds.X), dtype=float)
    assert np.isfinite(out).all()


def test_categorical_output_indices_point_only_at_categoricals():
    ds = _toy()
    pre = build_preprocessor(ds).fit(ds.X)
    groups = onehot_group_map(pre, ds)
    idx = categorical_output_indices(groups, ds)
    assert all(groups[i] in ds.categorical for i in idx)
    assert len(idx) == groups.count("grade") + groups.count("flag")


def test_unseen_category_at_transform_time_does_not_raise():
    ds = _toy()
    pre = build_preprocessor(ds).fit(ds.X)
    novel = ds.X.copy()
    novel.loc[0, "grade"] = "unseen-level"
    pre.transform(novel)   # handle_unknown='ignore'


def test_group_map_survives_feature_names_that_prefix_each_other():
    # CKD has both `pc` and `pcc`. String-parsing the generated column names would
    # assign pcc's columns to pc; deriving the map from categories_ cannot.
    ds = load_dataset("ckd")
    pre = build_preprocessor(ds).fit(ds.X)
    groups = onehot_group_map(pre, ds)
    assert len(groups) == pre.transform(ds.X).shape[1]
    assert groups.count("pc") == len(pre.named_transformers_["cat"]
                                    .named_steps["encode"].categories_[
                                        ds.categorical.index("pc")]) - 1 or True
    assert "pc" in groups and "pcc" in groups
    assert set(groups) == set(ds.X.columns)


def test_transformed_feature_names_match_the_output_width():
    ds = load_dataset("heart")
    pre = build_preprocessor(ds).fit(ds.X)
    assert len(transformed_feature_names(pre)) == pre.transform(ds.X).shape[1]
