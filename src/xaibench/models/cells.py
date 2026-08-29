"""One fitted (dataset, model) cell: the fixed object every explainer operates on."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from xaibench.config import SEED, TEST_SIZE, Profile
from xaibench.data.loaders import load_dataset
from xaibench.models.pipelines import (
    build_preprocessor,
    categorical_output_indices,
    onehot_group_map,
)
from xaibench.models.zoo import build_model


@dataclass(frozen=True)
class Cell:
    """Frozen so `dataclasses.replace` can swap the test matrix (RQ2b neighbours) or
    the classifier (RQ2c bootstrap refits) without touching anything else."""

    dataset: str
    model: str
    preprocessor: object
    classifier: object
    X_train_t: np.ndarray
    X_test_t: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    groups: list[str]
    feature_order: list[str]
    categorical_idx: list[int]
    instance_ids: np.ndarray
    X_train_raw: pd.DataFrame = field(repr=False, default=None)
    y_train_raw: pd.Series = field(repr=False, default=None)

    def predict_proba(self, X_t: np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier.predict_proba(np.atleast_2d(X_t)), dtype=float)

    @property
    def fixed_mask(self) -> np.ndarray:
        """True for output columns that must not be jittered (one-hot columns)."""
        mask = np.zeros(len(self.groups), dtype=bool)
        if self.categorical_idx:
            mask[np.asarray(self.categorical_idx, dtype=int)] = True
        return mask

    @property
    def train_sigma(self) -> np.ndarray:
        return self.X_train_t.std(axis=0, ddof=1)

    @property
    def key(self) -> str:
        return f"{self.dataset}|{self.model}"


def _stratified_subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Up to n indices, class-proportional, deterministic. Returns sorted indices."""
    n = int(min(n, y.size))
    if n >= y.size:
        return np.arange(y.size)
    idx, _ = train_test_split(np.arange(y.size), train_size=n, stratify=y,
                              random_state=seed)
    return np.sort(idx)


def build_cell(dataset_id: str, model_id: str, params: dict, profile: Profile,
               seed: int = SEED) -> Cell:
    """Fit the one model per cell that all explanation analysis runs against.

    Track B of the protocol: a single stratified 80/20 split, hyperparameters taken
    from Track A's inner loop, preprocessing fitted on the train partition only.
    """
    ds = load_dataset(dataset_id)
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=TEST_SIZE, stratify=ds.y, random_state=seed)

    pre = build_preprocessor(ds).fit(X_tr)
    X_tr_t = np.asarray(pre.transform(X_tr), dtype=float)
    X_te_t = np.asarray(pre.transform(X_te), dtype=float)

    clf = build_model(model_id, seed=seed)
    if params:
        clf.set_params(**params)
    clf.fit(X_tr_t, np.asarray(y_tr))

    y_te_arr = np.asarray(y_te)
    groups = onehot_group_map(pre, ds)
    return Cell(
        dataset=dataset_id, model=model_id, preprocessor=pre, classifier=clf,
        X_train_t=X_tr_t, X_test_t=X_te_t,
        y_train=np.asarray(y_tr), y_test=y_te_arr,
        groups=groups, feature_order=list(ds.X.columns),
        categorical_idx=categorical_output_indices(groups, ds),
        instance_ids=_stratified_subsample(y_te_arr, profile.n_explain, seed),
        X_train_raw=X_tr.reset_index(drop=True), y_train_raw=y_tr.reset_index(drop=True),
    )
