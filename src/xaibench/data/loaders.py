"""Load the three datasets with their per-dataset semantics fixes applied."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xaibench.config import paths
from xaibench.data.fetch import SOURCES, fetch_all

PIMA_IMPOSSIBLE_ZEROS = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
HEART_CATEGORICAL = ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")


@dataclass(frozen=True)
class Dataset:
    id: str
    X: pd.DataFrame
    y: pd.Series
    numeric: list[str]
    categorical: list[str]
    source_url: str
    licence: str
    citation: str

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean())

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def missing_rate(self) -> float:
        return float(self.X.isna().to_numpy().mean())

    def missing_by_column(self) -> pd.Series:
        return self.X.isna().mean().sort_values(ascending=False)


def _is_texty(series: pd.Series) -> bool:
    """True for string-like columns.

    pandas 3.x gives string columns the `str` dtype rather than `object`, so a bare
    `dtype == object` check silently misses every CKD categorical column.
    """
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def _raw(dataset_id: str) -> pd.DataFrame:
    path = paths().data_raw / f"{dataset_id}.csv"
    if not path.exists():
        fetch_all()
    return pd.read_csv(path)


def _load_pima() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = _raw("pima")
    y = df.pop("Outcome").astype(int)
    # Biologically impossible zeros are missing data, not measurements. This is a
    # data-semantics correction applied at load time, so it is not fold-dependent.
    for col in PIMA_IMPOSSIBLE_ZEROS:
        df[col] = df[col].replace(0, np.nan)
    return df, y, list(df.columns), []


def _load_heart() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = _raw("heart")
    y = (pd.to_numeric(df.pop("num"), errors="coerce").fillna(0) > 0).astype(int)
    df = df.apply(pd.to_numeric, errors="coerce")
    categorical = [c for c in HEART_CATEGORICAL if c in df.columns]
    numeric = [c for c in df.columns if c not in categorical]
    # Nominal codes must not be treated as magnitudes by the linear model, so they
    # are carried as strings and one-hot encoded downstream.
    for col in categorical:
        as_text = df[col].astype("object").map(
            lambda v: np.nan if pd.isna(v) else str(int(v)) if float(v).is_integer() else str(v)
        )
        df[col] = as_text
    return df, y, numeric, categorical


def _load_ckd() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = _raw("ckd")
    for col in df.columns:
        if _is_texty(df[col]):
            cleaned = df[col].astype("object").map(
                lambda v: np.nan if pd.isna(v) else str(v).replace("\t", "").strip()
            )
            df[col] = cleaned.replace({"?": np.nan, "nan": np.nan, "NaN": np.nan, "": np.nan})
    y = df.pop("class").map({"ckd": 1, "notckd": 0}).astype(int)

    numeric, categorical = [], []
    for col in df.columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        # A column is numeric only if coercion loses no observed value.
        if coerced.notna().sum() == df[col].notna().sum():
            df[col] = coerced
            numeric.append(col)
        else:
            categorical.append(col)
    return df, y, numeric, categorical


_LOADERS = {"pima": _load_pima, "heart": _load_heart, "ckd": _load_ckd}


def load_dataset(dataset_id: str) -> Dataset:
    if dataset_id not in _LOADERS:
        raise ValueError(f"unknown dataset {dataset_id!r}; expected one of {sorted(_LOADERS)}")
    X, y, numeric, categorical = _LOADERS[dataset_id]()
    src = SOURCES[dataset_id]
    return Dataset(
        id=dataset_id,
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        numeric=numeric,
        categorical=categorical,
        source_url=src.url,
        licence=src.licence,
        citation=src.citation,
    )
