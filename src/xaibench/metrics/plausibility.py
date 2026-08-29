"""Overlap between top-k explanation features and pre-registered reference sets.

The weakest metric in the study, by design and by declaration: reference sets are
literature-derived, incomplete, and not causal. Reported as corroborating evidence
only, never as a correctness criterion (protocol §5.5).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import yaml

from xaibench.config import paths


@dataclass(frozen=True)
class ReferenceSet:
    dataset: str
    features: list[str]
    citations: list[str]
    note: str


@lru_cache(maxsize=1)
def load_reference_sets() -> dict[str, ReferenceSet]:
    raw = yaml.safe_load((paths().docs / "reference_features.yaml").read_text())
    return {
        key: ReferenceSet(dataset=key, features=list(value["features"]),
                          citations=list(value["citations"]),
                          note=str(value.get("note", "")))
        for key, value in raw.items()
    }


def _topk_names(ranking, feature_order, k: int) -> set[str]:
    order = np.asarray(ranking, dtype=int).ravel()
    k = max(1, min(int(k), order.size))
    return {feature_order[i] for i in order[:k]}


def jaccard_vs_reference(ranking, feature_order, reference, k: int = 5) -> float:
    top = _topk_names(ranking, feature_order, k)
    # Reference entries absent from this dataset can be neither found nor missed.
    ref = set(reference) & set(feature_order)
    union = top | ref
    return float(len(top & ref) / len(union)) if union else 0.0


def precision_at_k(ranking, feature_order, reference, k: int = 5) -> float:
    top = _topk_names(ranking, feature_order, k)
    ref = set(reference) & set(feature_order)
    return float(len(top & ref) / len(top)) if top else 0.0
