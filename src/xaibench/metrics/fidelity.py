"""Deletion/insertion fidelity metrics, evaluated against the model's own class."""
from __future__ import annotations

import numpy as np

# numpy<2 is pinned (shap's numba chain breaks on 2.x), where the trapezoid rule is
# still spelled trapz.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def median_baseline(X: np.ndarray) -> np.ndarray:
    return np.nanmedian(np.asarray(X, dtype=float), axis=0)


def marginal_baselines(X: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    """Independent draws from each feature's observed training marginal.

    Median replacement pushes instances off the data manifold, which can inflate
    AOPC. This is the protocol's sensitivity variant (§5.3): sample real observed
    values instead, so perturbed rows stay marginally plausible.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty((int(n_draws), X.shape[1]), dtype=float)
    for j in range(X.shape[1]):
        column = X[np.isfinite(X[:, j]), j]
        column = column if column.size else np.zeros(1)
        out[:, j] = rng.choice(column, size=int(n_draws), replace=True)
    return out


def _target_class(predict_proba, x: np.ndarray) -> int:
    return int(np.asarray(predict_proba(np.atleast_2d(x)), dtype=float)[0].argmax())


def _probabilities(predict_proba, rows: np.ndarray, cls: int) -> np.ndarray:
    return np.asarray(predict_proba(rows), dtype=float)[:, cls]


def deletion_curve(predict_proba, x, ranking, baseline, k_max: int) -> np.ndarray:
    """p(own class) after deleting the top-0, top-1, ..., top-k ranked features.

    One batched predict_proba call over all k_max+1 perturbed rows: the study
    evaluates on the order of a million curve points, so a call per row would
    dominate the runtime.
    """
    x = np.asarray(x, dtype=float).ravel()
    baseline = np.asarray(baseline, dtype=float).ravel()
    order = np.asarray(ranking, dtype=int).ravel()
    k_max = int(min(k_max, order.size))
    cls = _target_class(predict_proba, x)
    rows = np.tile(x, (k_max + 1, 1))
    for k in range(1, k_max + 1):
        rows[k:, order[k - 1]] = baseline[order[k - 1]]
    return _probabilities(predict_proba, rows, cls)


def insertion_curve(predict_proba, x, ranking, baseline, k_max: int) -> np.ndarray:
    """p(own class) as the top-k features are restored onto a fully-baseline row."""
    x = np.asarray(x, dtype=float).ravel()
    baseline = np.asarray(baseline, dtype=float).ravel()
    order = np.asarray(ranking, dtype=int).ravel()
    k_max = int(min(k_max, order.size))
    cls = _target_class(predict_proba, x)
    rows = np.tile(baseline, (k_max + 1, 1))
    for k in range(1, k_max + 1):
        rows[k:, order[k - 1]] = x[order[k - 1]]
    return _probabilities(predict_proba, rows, cls)


def aopc_deletion(predict_proba, x, ranking, baseline, k_max: int) -> float:
    """Mean probability drop over k = 1..K. Higher means a more faithful ranking."""
    curve = deletion_curve(predict_proba, x, ranking, baseline, k_max)
    return float(np.mean(curve[0] - curve[1:])) if curve.size > 1 else 0.0


def auc_insertion(predict_proba, x, ranking, baseline, k_max: int) -> float:
    """Normalised area under the insertion curve. Higher is more faithful."""
    curve = insertion_curve(predict_proba, x, ranking, baseline, k_max)
    return float(_trapezoid(curve, dx=1.0) / max(len(curve) - 1, 1))


def comprehensiveness(predict_proba, x, ranking, baseline, k: int = 5) -> float:
    """p(own class) lost by removing the top-k (DeYoung et al., 2020). Higher is better."""
    curve = deletion_curve(predict_proba, x, ranking, baseline, k)
    return float(curve[0] - curve[-1])


def sufficiency(predict_proba, x, ranking, baseline, k: int = 5) -> float:
    """p(own class) lost by keeping ONLY the top-k. Nearer zero is better."""
    curve = insertion_curve(predict_proba, x, ranking, baseline, k)
    cls = _target_class(predict_proba, x)
    full = _probabilities(predict_proba, np.atleast_2d(np.asarray(x, dtype=float)), cls)[0]
    return float(full - curve[-1])


def _grouped_rows(x, baseline, groups, k_max, start_from_baseline):
    x = np.asarray(x, dtype=float).ravel()
    baseline = np.asarray(baseline, dtype=float).ravel()
    k_max = int(min(k_max, len(groups)))
    source, target = (baseline, x) if start_from_baseline else (x, baseline)
    rows = np.tile(source, (k_max + 1, 1))
    for k in range(1, k_max + 1):
        columns = np.asarray(groups[k - 1], dtype=int)
        rows[k:, columns] = target[columns]
    return rows


def deletion_curve_grouped(predict_proba, x, groups, baseline, k_max: int) -> np.ndarray:
    """Deletion curve at ORIGINAL-feature granularity.

    `groups[i]` holds every transformed column belonging to the i-th ranked original
    feature. Deleting "the top-k features" must remove all of their columns: deleting
    k *columns* instead would under-perturb Heart (13 features across 25 columns) while
    leaving PIMA unaffected, so the two datasets would not be measured on the same scale.
    """
    rows = _grouped_rows(x, baseline, groups, k_max, start_from_baseline=False)
    return _probabilities(predict_proba, rows, _target_class(predict_proba, x))


def insertion_curve_grouped(predict_proba, x, groups, baseline, k_max: int) -> np.ndarray:
    """Insertion curve at ORIGINAL-feature granularity."""
    rows = _grouped_rows(x, baseline, groups, k_max, start_from_baseline=True)
    return _probabilities(predict_proba, rows, _target_class(predict_proba, x))


def curve_metrics(deletion: np.ndarray, insertion: np.ndarray, k: int = 5) -> dict:
    """All four fidelity metrics read off two already-computed curves.

    Exact, not approximate: deletion[0] is p(own class) on the unperturbed instance,
    so comprehensiveness and sufficiency need no extra model calls. That halves the
    predict_proba traffic across ~14k curve evaluations.
    """
    deletion = np.asarray(deletion, dtype=float)
    insertion = np.asarray(insertion, dtype=float)
    k = int(min(k, deletion.size - 1, insertion.size - 1))
    return {
        "aopc": float(np.mean(deletion[0] - deletion[1:])) if deletion.size > 1 else 0.0,
        "auc_ins": float(_trapezoid(insertion, dx=1.0) / max(insertion.size - 1, 1)),
        "comprehensiveness": float(deletion[0] - deletion[k]),
        "sufficiency": float(deletion[0] - insertion[k]),
    }


def batched_curve_metrics(predict_proba, x, groups, baselines, k_max: int,
                          k: int = 5) -> dict:
    """Fidelity metrics averaged over several baselines, in ONE model call.

    The naive loop costs two `predict_proba` calls per baseline, which across the
    study is ~28k calls and dominates Stage 3. Stacking every deletion and insertion
    row for every baseline into a single matrix cuts that to one call per
    (explainer, instance) - the same numbers, ~20x less traffic.
    """
    x = np.asarray(x, dtype=float).ravel()
    k_max = int(min(k_max, len(groups)))
    width = k_max + 1
    blocks = []
    for baseline in baselines:
        blocks.append(_grouped_rows(x, baseline, groups, k_max, start_from_baseline=False))
        blocks.append(_grouped_rows(x, baseline, groups, k_max, start_from_baseline=True))

    cls = _target_class(predict_proba, x)
    flat = _probabilities(predict_proba, np.vstack(blocks), cls)

    scored = []
    for i in range(len(baselines)):
        deletion = flat[(2 * i) * width:(2 * i + 1) * width]
        insertion = flat[(2 * i + 1) * width:(2 * i + 2) * width]
        scored.append(curve_metrics(deletion, insertion, k=k))
    return {key: float(np.mean([s[key] for s in scored])) for key in scored[0]}
