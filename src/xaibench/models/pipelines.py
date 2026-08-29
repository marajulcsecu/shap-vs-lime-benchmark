"""Leakage-safe preprocessing and the one-hot -> original-feature aggregation map."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(ds) -> ColumnTransformer:
    """One ColumnTransformer, always fitted inside a fold or on a train partition only."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                 sparse_output=False, dtype=np.float64)),
    ])
    transformers = []
    if ds.numeric:
        transformers.append(("num", numeric, list(ds.numeric)))
    if ds.categorical:
        transformers.append(("cat", categorical, list(ds.categorical)))
    return ColumnTransformer(transformers, remainder="drop",
                             verbose_feature_names_out=False)


def onehot_group_map(preprocessor: ColumnTransformer, ds) -> list[str]:
    """One original-feature name per output column, in output-column order.

    Built from the fitted encoder's own `categories_`/`drop_idx_` rather than from
    generated column-name strings, which are ambiguous when one feature name is a
    prefix of another -- CKD has both `pc` and `pcc`.
    """
    groups: list[str] = list(ds.numeric)
    if ds.categorical:
        ohe = preprocessor.named_transformers_["cat"].named_steps["encode"]
        drop_idx = getattr(ohe, "drop_idx_", None)
        for i, (col, cats) in enumerate(zip(ds.categorical, ohe.categories_)):
            n_out = len(cats)
            if drop_idx is not None and drop_idx[i] is not None:
                n_out -= 1
            groups.extend([col] * n_out)
    return groups


def transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return [str(name) for name in preprocessor.get_feature_names_out()]


def categorical_output_indices(groups: list[str], ds) -> list[int]:
    """Output-column indices LIME must treat as categorical rather than discretise."""
    cats = set(ds.categorical)
    return [i for i, g in enumerate(groups) if g in cats]


def aggregate_attributions(values: np.ndarray, groups: list[str],
                           feature_order: list[str]) -> np.ndarray:
    """Sum attributions over the one-hot columns belonging to each original feature.

    Summation, not averaging: SHAP attributions are additive, and only a sum leaves
    the row total -- and therefore the additivity property -- unchanged.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    if values.shape[1] != len(groups):
        raise ValueError(
            f"values has {values.shape[1]} columns but groups has {len(groups)}")
    index = {feat: j for j, feat in enumerate(feature_order)}
    out = np.zeros((values.shape[0], len(feature_order)), dtype=float)
    for col, group in enumerate(groups):
        out[:, index[group]] += values[:, col]
    return out
