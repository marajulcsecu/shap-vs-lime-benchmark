"""Three rank-fusion rules plus the random control, on the shared contract.

Explanation aggregation is not new; the contribution is a controlled, leakage-safe
evaluation of these rules against a strict bar (beat max(SHAP, LIME), not their mean).
"""
from __future__ import annotations

import numpy as np

from xaibench.config import CALIB_FRACTION, RANDOM_CONTROL_DRAWS, RRF_K, SEED
from xaibench.explain.base import Explanation, config_hash
from xaibench.metrics.agreement import ranking_from_attributions


def normalised_ranks(attr: np.ndarray) -> np.ndarray:
    """Position of each feature in [0, 1]; 0 = most important."""
    order = ranking_from_attributions(attr)
    d = order.size
    out = np.empty(d, dtype=float)
    out[order] = np.arange(d, dtype=float)
    return out / max(d - 1, 1)


def _as_pseudo_attributions(fused_rank: np.ndarray) -> np.ndarray:
    """1 - normalised rank, so ranking by |value| reproduces the fused order.

    This is what lets a fused ranking travel through the same Explanation contract
    as a real attribution vector, with no special case in any metric.
    """
    return 1.0 - np.asarray(fused_rank, dtype=float)


def fuse_rankavg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _as_pseudo_attributions(0.5 * (normalised_ranks(a) + normalised_ranks(b)))


def fuse_weighted(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    w = float(np.clip(w, 0.0, 1.0))
    return _as_pseudo_attributions(
        w * normalised_ranks(a) + (1.0 - w) * normalised_ranks(b))


def fuse_rrf(a: np.ndarray, b: np.ndarray, k: int = RRF_K) -> np.ndarray:
    """Reciprocal rank fusion (Cormack et al., 2009).

    Each list's contribution is bounded by 1/(k+1), so one badly wrong ranking
    degrades the fusion less than arithmetic rank averaging does.
    """
    d = int(np.asarray(a).size)
    rank_a = normalised_ranks(a) * max(d - 1, 1)
    rank_b = normalised_ranks(b) * max(d - 1, 1)
    score = 1.0 / (k + 1.0 + rank_a) + 1.0 / (k + 1.0 + rank_b)
    order = np.lexsort((np.arange(d), -score))
    fused = np.empty(d, dtype=float)
    fused[order] = np.arange(d, dtype=float) / max(d - 1, 1)
    return _as_pseudo_attributions(fused)


def calibration_split(instance_ids, seed: int,
                      fraction: float = CALIB_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint calibration / evaluation instance sets.

    The fidelity weight is estimated on the calibration set only. Without this
    split the weight would be tuned on the same instances H6 is tested on, and the
    consensus rule would win by construction.
    """
    ids = np.asarray(instance_ids, dtype=int)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    n_calib = max(1, int(round(fraction * ids.size))) if ids.size > 1 else 0
    return np.sort(shuffled[:n_calib]), np.sort(shuffled[n_calib:])


def fidelity_weight(aopc_shap: float, aopc_lime: float) -> float:
    """Share of weight given to SHAP, from calibration-set fidelity."""
    a = max(float(aopc_shap), 0.0)
    b = max(float(aopc_lime), 0.0)
    return 0.5 if (a + b) <= 0.0 else float(a / (a + b))


_RULES = {
    "rankavg": lambda a, b, w: fuse_rankavg(a, b),
    "rrf": lambda a, b, w: fuse_rrf(a, b),
    "fidwt": lambda a, b, w: fuse_weighted(a, b, w),
}

CONSENSUS_RULES = tuple(_RULES)


def build_consensus(shap_exps: list[Explanation], lime_exps: list[Explanation],
                    weight: float, rule: str) -> list[Explanation]:
    if rule not in _RULES:
        raise ValueError(f"unknown fusion rule {rule!r}; expected one of {sorted(_RULES)}")
    if len(shap_exps) != len(lime_exps):
        raise ValueError("consensus needs one SHAP and one LIME explanation per instance")

    cfg = config_hash(explainer=rule, weight=round(float(weight), 6))
    out: list[Explanation] = []
    for s, l in zip(shap_exps, lime_exps):
        if s.instance_id != l.instance_id:
            raise ValueError(
                f"instance misalignment: shap={s.instance_id} lime={l.instance_id}")
        out.append(Explanation(
            dataset=s.dataset, model=s.model, explainer=rule, instance_id=s.instance_id,
            attributions=_RULES[rule](s.attributions, l.attributions, weight),
            feature_names=s.feature_names,
            # Fusion consumes both base explanations, so it inherits both costs.
            runtime_s=float(s.runtime_s + l.runtime_s), seed=s.seed, config_hash=cfg))
    return out


def random_control(cell, instance_ids, seed: int = SEED,
                   n_draws: int = RANDOM_CONTROL_DRAWS) -> list[Explanation]:
    """Uniformly random rankings.

    Without this control an AOPC value has no scale: deleting any five of eight
    features moves a prediction, so "SHAP achieved 0.31" cannot be interpreted.
    """
    ids = np.asarray(instance_ids, dtype=int)
    d = len(cell.feature_order)
    rng = np.random.default_rng(seed)
    cfg = config_hash(explainer="random", n_draws=int(n_draws))
    return [Explanation(
        dataset=cell.dataset, model=cell.model, explainer="random", instance_id=int(i),
        attributions=rng.permutation(d).astype(float),
        feature_names=tuple(cell.feature_order), runtime_s=1e-6,
        seed=int(seed), config_hash=cfg) for i in ids]
