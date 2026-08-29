"""Rank-agreement metrics. Rankings are index orderings; rank vectors are positions."""
from __future__ import annotations

import numpy as np
from scipy import stats


def ranking_from_attributions(attr: np.ndarray) -> np.ndarray:
    """Feature indices ordered by descending |attribution|, ties broken by index.

    Ties must break deterministically, or two runs of the same deterministic
    explainer would disagree. lexsort with the index as secondary key does that.
    """
    magnitude = np.abs(np.asarray(attr, dtype=float).ravel())
    return np.lexsort((np.arange(magnitude.size), -magnitude))


def rank_vector(order) -> np.ndarray:
    """Inverse of a ranking: the position occupied by each feature."""
    order = np.asarray(order, dtype=int).ravel()
    out = np.empty(order.size, dtype=float)
    out[order] = np.arange(order.size, dtype=float)
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0   # constant input: correlation undefined, reported as no association
    rho = stats.spearmanr(a, b).statistic
    return 0.0 if not np.isfinite(rho) else float(rho)


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    tau = stats.kendalltau(a, b, variant="b").statistic
    return 0.0 if not np.isfinite(tau) else float(tau)


def topk_set(order, k: int) -> set[int]:
    items = list(np.asarray(order, dtype=int).ravel())
    return set(items[: max(1, min(int(k), len(items)))])


def jaccard_at_k(order_a, order_b, k: int) -> float:
    set_a, set_b = topk_set(order_a, k), topk_set(order_b, k)
    union = set_a | set_b
    return float(len(set_a & set_b) / len(union)) if union else 1.0


def _prefix_overlaps(order_a, order_b) -> list[int]:
    """|top-d(a) ∩ top-d(b)| for every depth d = 1..k, computed incrementally.

    Separated out and unit-tested against a set-intersection reference, because a
    drift in this counter would silently bias every RBO value in the study.
    """
    a = list(np.asarray(order_a, dtype=int).ravel())
    b = list(np.asarray(order_b, dtype=int).ravel())
    seen_a: set[int] = set()
    seen_b: set[int] = set()
    overlap = 0
    out: list[int] = []
    for depth in range(len(a)):
        head_a, head_b = a[depth], b[depth]
        seen_a.add(head_a)
        seen_b.add(head_b)
        if head_a == head_b:
            overlap += 1
        else:
            if head_a in seen_b:
                overlap += 1
            if head_b in seen_a:
                overlap += 1
        out.append(overlap)
    return out


def rbo(order_a, order_b, p: float = 0.90) -> float:
    """Extrapolated Rank-Biased Overlap (Webber, Moffat & Zobel, 2010).

    Both arguments are full permutations of the same feature set, so the tail
    agreement A_k is exactly 1 and the extrapolation term reduces to p**k. Unlike
    Jaccard, RBO needs no choice of k; unlike Spearman, it weights the head of the
    list, which is what a practitioner reading a top-5 explanation actually uses.
    """
    a = np.asarray(order_a, dtype=int).ravel()
    b = np.asarray(order_b, dtype=int).ravel()
    if a.size != b.size:
        raise ValueError("RBO here assumes equal-length permutations of one feature set")
    k = a.size
    if k == 0:
        return 1.0
    overlaps = _prefix_overlaps(a, b)
    agreements = np.array([overlaps[d - 1] / d for d in range(1, k + 1)], dtype=float)
    weights = p ** np.arange(k, dtype=float)
    return float((1.0 - p) * float(np.sum(weights * agreements)) + (p ** k) * agreements[-1])


def rbo_floor(d: int, p: float = 0.90) -> float:
    """RBO of a perfectly reversed ranking of d items -- the minimum attainable value.

    Both rankings compared in this study are permutations of the SAME feature set,
    so the extrapolation term p**k * A_k is always exactly p**k, and for small d it
    dominates the score. The floor is 0.589 at d=8 (PIMA) but 0.193 at d=24 (CKD),
    so raw RBO is not comparable across datasets of different width -- PIMA simply
    cannot score below 0.589. See rbo_normalised.
    """
    order = list(range(int(d)))
    return rbo(order, order[::-1], p=p)


def rbo_normalised(order_a, order_b, p: float = 0.90) -> float:
    """RBO rescaled so 0 is a reversed ranking and 1 is an identical one.

    This is what makes the measure comparable across PIMA (8 features), Heart (13),
    and CKD (24). Logged as protocol amendment 2026-08-29-A1.
    """
    d = np.asarray(order_a, dtype=int).size
    floor = rbo_floor(d, p)
    if floor >= 1.0:
        return 1.0
    return float((rbo(order_a, order_b, p=p) - floor) / (1.0 - floor))
