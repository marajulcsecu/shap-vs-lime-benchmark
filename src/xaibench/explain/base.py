"""The single output contract every explainer satisfies."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from xaibench.metrics.agreement import ranking_from_attributions


def config_hash(**kwargs) -> str:
    """Short stable digest of an explainer configuration, for provenance."""
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Explanation:
    dataset: str
    model: str
    explainer: str
    instance_id: int
    attributions: np.ndarray      # (d,) in ORIGINAL-feature space
    feature_names: tuple[str, ...]
    runtime_s: float
    seed: int
    config_hash: str

    def __post_init__(self) -> None:
        shape = np.asarray(self.attributions).shape
        if shape != (len(self.feature_names),):
            raise ValueError(
                f"attributions shape {shape} does not match {len(self.feature_names)} "
                "feature names; one-hot aggregation was skipped"
            )

    @property
    def ranking(self) -> np.ndarray:
        return ranking_from_attributions(self.attributions)

    @property
    def rank_of_feature(self) -> np.ndarray:
        out = np.empty(len(self.feature_names), dtype=int)
        out[self.ranking] = np.arange(len(self.feature_names))
        return out


def explanations_to_frame(explanations: list[Explanation]) -> pd.DataFrame:
    """Long format: one row per (explanation, feature). Parquet-friendly and tidy."""
    rows = []
    for exp in explanations:
        rank_of = exp.rank_of_feature
        for j, name in enumerate(exp.feature_names):
            rows.append({
                "dataset": exp.dataset, "model": exp.model, "explainer": exp.explainer,
                "instance_id": int(exp.instance_id), "feature": name,
                "attribution": float(exp.attributions[j]), "rank": int(rank_of[j]),
                "runtime_s": float(exp.runtime_s), "seed": int(exp.seed),
                "config_hash": exp.config_hash,
            })
    return pd.DataFrame(rows)
