"""Stability across three separable sources: explainer randomness, input
perturbation, and model retraining.

Conflating them is what single-notion stability studies get wrong, and only the
first is degenerate for tree SHAP.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from xaibench.metrics.agreement import rank_vector, ranking_from_attributions, spearman

EPS = 1e-12


def rerun_cv(repeats: np.ndarray) -> float:
    """Mean per-feature coefficient of variation across R repeats of one instance.

    Unit of analysis for H2a. Exactly 0.0 for a deterministic explainer.
    """
    r = np.atleast_2d(np.asarray(repeats, dtype=float))
    if r.shape[0] < 2:
        return 0.0
    sd = r.std(axis=0, ddof=1)
    mu = np.abs(r.mean(axis=0))
    return float(np.mean(sd / (mu + EPS)))


def repeat_pair_rho(repeats: np.ndarray) -> float:
    """Mean Spearman rho over all R(R-1)/2 pairs of rank vectors from the repeats."""
    r = np.atleast_2d(np.asarray(repeats, dtype=float))
    if r.shape[0] < 2:
        return 1.0
    vectors = [rank_vector(ranking_from_attributions(row)) for row in r]
    pairs = [spearman(a, b) for a, b in combinations(vectors, 2)]
    return float(np.mean(pairs)) if pairs else 1.0


def sample_neighbours(x: np.ndarray, sigma: np.ndarray, n: int, seed: int,
                      fixed_mask: np.ndarray | None = None) -> np.ndarray:
    """Gaussian neighbours of x.

    Columns flagged in fixed_mask are held constant, because jittering a one-hot
    column produces an input the model has never seen.
    """
    x = np.asarray(x, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=1.0, size=(int(n), x.size)) * sigma
    if fixed_mask is not None:
        noise[:, np.asarray(fixed_mask, dtype=bool)] = 0.0
    return x + noise


def local_lipschitz(phi_x: np.ndarray, x: np.ndarray,
                    phi_neighbours: np.ndarray, neighbours: np.ndarray) -> float:
    """max over neighbours of ||phi(x)-phi(x')||_2 / ||x-x'||_2.

    The estimator of Alvarez-Melis & Jaakkola (2018). Lower is more robust. A
    neighbour coinciding with x is skipped rather than dividing by zero.
    """
    phi_x = np.asarray(phi_x, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    phi_n = np.atleast_2d(np.asarray(phi_neighbours, dtype=float))
    nbrs = np.atleast_2d(np.asarray(neighbours, dtype=float))
    input_dist = np.linalg.norm(nbrs - x, axis=1)
    attr_dist = np.linalg.norm(phi_n - phi_x, axis=1)
    usable = input_dist > 1e-12
    return float(np.max(attr_dist[usable] / input_dist[usable])) if usable.any() else 0.0


def retrain_agreement(global_attr_per_refit: np.ndarray) -> float:
    """Mean pairwise Spearman rho between global rankings from B model refits.

    Separates model variance from explainer variance: an explanation can be
    perfectly reproducible (rerun_cv == 0) and still worthless if a marginally
    different training sample reorders its top features.
    """
    return repeat_pair_rho(global_attr_per_refit)
