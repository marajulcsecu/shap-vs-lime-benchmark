"""Feature-dependence-aware fidelity validation (Post-Review Amendment A8).

Addresses Reviewer #1 Point 4: median and independent-marginal perturbation both
assume feature independence and can push perturbed rows off the data manifold.
This module provides two complementary dependence-aware validations:

1. Local Conditional k-NN Replacement: when deleting feature set S, replace x_S
   jointly with the S-columns of the M nearest training rows matched on the
   RETAINED columns (complement of S). This keeps perturbed inputs on the empirical
   joint manifold and preserves valid one-hot categorical encodings.

2. Global ROAR (Remove And Retrain; Hooker et al., NeurIPS 2019): remove the
   top-ranked original features from both train and test partitions and retrain
   the full preprocessing + classifier pipeline from scratch under 5-fold
   stratified cross-validation, eliminating off-manifold perturbation bias.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from xaibench.config import AOPC_MAX_K, COMP_SUFF_K, SEED
from xaibench.metrics.fidelity import _probabilities, _target_class
from xaibench.models.pipelines import build_preprocessor
from xaibench.models.zoo import build_model


def conditional_knn_deletion_curve(
    predict_proba,
    x: np.ndarray,
    groups: list[np.ndarray],
    X_train_t: np.ndarray,
    k_max: int = AOPC_MAX_K,
    n_neighbors: int = 10,
    seed: int = SEED,
) -> np.ndarray:
    """Conditional deletion curve via k-NN matching on retained columns.

    At step k = 1..k_max, columns belonging to the top-k ranked original features
    are replaced jointly by the values of the `n_neighbors` nearest training rows
    whose Euclidean distance is computed ONLY on the retained (undeleted) columns.
    All k_max * n_neighbors perturbed rows are evaluated in ONE batched predict_proba
    call per explanation.
    """
    x = np.asarray(x, dtype=float).ravel()
    X_tr = np.asarray(X_train_t, dtype=float)
    n_train, n_cols = X_tr.shape
    k_max = int(min(k_max, len(groups)))
    m = int(min(max(1, n_neighbors), n_train))
    cls = _target_class(predict_proba, x)

    if k_max == 0:
        p0 = _probabilities(predict_proba, x[None, :], cls)[0]
        return np.array([float(p0)], dtype=float)

    rng = np.random.default_rng(seed)
    all_cols = np.arange(n_cols, dtype=int)
    deleted_mask = np.zeros(n_cols, dtype=bool)

    blocks = [x[None, :]]
    for k in range(1, k_max + 1):
        cols = np.asarray(groups[k - 1], dtype=int)
        deleted_mask[cols] = True
        del_idx = all_cols[deleted_mask]
        ret_idx = all_cols[~deleted_mask]

        if ret_idx.size > 0:
            diff = X_tr[:, ret_idx] - x[ret_idx]
            sq_dist = np.einsum("ij,ij->i", diff, diff)
            if m < n_train:
                # Deterministic tie-break via mergesort on top candidates.
                nn_idx = np.argsort(sq_dist, kind="mergesort")[:m]
            else:
                nn_idx = np.arange(n_train, dtype=int)
        else:
            # All features deleted: draw M real joint training rows.
            nn_idx = rng.choice(n_train, size=m, replace=False)

        perturbed = np.tile(x, (m, 1))
        perturbed[:, del_idx] = X_tr[nn_idx][:, del_idx]
        blocks.append(perturbed)

    stacked = np.vstack(blocks)
    probs = _probabilities(predict_proba, stacked, cls)

    curve = np.empty(k_max + 1, dtype=float)
    curve[0] = float(probs[0])
    for k in range(1, k_max + 1):
        start = 1 + (k - 1) * m
        curve[k] = float(np.mean(probs[start : start + m]))
    return curve


def conditional_knn_metrics(
    predict_proba,
    x: np.ndarray,
    groups: list[np.ndarray],
    X_train_t: np.ndarray,
    k_max: int = AOPC_MAX_K,
    k: int = COMP_SUFF_K,
    n_neighbors: int = 10,
    seed: int = SEED,
) -> dict[str, float]:
    """Conditional AOPC and comprehensiveness from the conditional k-NN deletion curve."""
    curve = conditional_knn_deletion_curve(
        predict_proba, x, groups, X_train_t,
        k_max=k_max, n_neighbors=n_neighbors, seed=seed,
    )
    k_eff = int(min(k, curve.size - 1))
    return {
        "aopc_cond": float(np.mean(curve[0] - curve[1:])) if curve.size > 1 else 0.0,
        "comp_cond": float(curve[0] - curve[k_eff]) if k_eff > 0 else 0.0,
    }


def _subset_dataset(ds, keep_cols: list[str]):
    """Build a reduced Dataset carrying only `keep_cols`."""
    keep_set = set(keep_cols)
    return replace(
        ds,
        X=ds.X[keep_cols].copy(),
        numeric=[c for c in ds.numeric if c in keep_set],
        categorical=[c for c in ds.categorical if c in keep_set],
    )


def _eval_subset_cv(ds, model_id: str, params: dict, keep_cols: list[str],
                    splits: list[tuple[np.ndarray, np.ndarray]], seed: int) -> list[dict]:
    """Fit preprocessor + classifier inside each CV fold on `keep_cols` only."""
    sub_ds = _subset_dataset(ds, keep_cols)
    y_arr = np.asarray(sub_ds.y, dtype=int)
    out = []
    for fold, (tr_idx, te_idx) in enumerate(splits):
        X_tr, X_te = sub_ds.X.iloc[tr_idx], sub_ds.X.iloc[te_idx]
        y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]
        pre = build_preprocessor(sub_ds).fit(X_tr)
        X_tr_t = np.asarray(pre.transform(X_tr), dtype=float)
        X_te_t = np.asarray(pre.transform(X_te), dtype=float)
        clf = build_model(model_id, seed=seed + fold)
        if params:
            clf.set_params(**params)
        clf.fit(X_tr_t, y_tr)
        probs = np.clip(clf.predict_proba(X_te_t)[:, 1], 1e-7, 1.0 - 1e-7)
        preds = (probs >= 0.5).astype(int)
        out.append({
            "fold": fold,
            "roc_auc": float(roc_auc_score(y_te, probs)),
            "log_loss": float(log_loss(y_te, probs, labels=[0, 1])),
            "accuracy": float(accuracy_score(y_te, preds)),
        })
    return out


def roar_for_cell(
    ds,
    model_id: str,
    params: dict,
    global_rankings: dict[str, list[str]],
    fractions: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50),
    n_splits: int = 5,
    n_random_draws: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    """Global ROAR (Remove And Retrain) for one (dataset, model) cell.

    Parameters
    ----------
    ds : Dataset
        Full dataset object (raw features + target).
    model_id : str
        Classifier identifier from `xaibench.models.zoo`.
    params : dict
        Fixed Track A hyperparameters for this cell.
    global_rankings : dict[str, list[str]]
        Mapping explainer -> ordered list of original feature names (most important first),
        e.g. {"shap": [...], "lime": [...]}.
    fractions : tuple[float, ...]
        Proportions of features to remove and retrain on (Hooker et al., 2019).
    n_splits : int
        Stratified K-fold splits for retraining evaluation.
    n_random_draws : int
        Number of random feature permutations averaged for the random control.
    """
    all_features = list(ds.X.columns)
    d = len(all_features)
    y_arr = np.asarray(ds.y, dtype=int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(ds.X, y_arr))

    # Baseline (fraction = 0.0, all features retained)
    base_folds = {r["fold"]: r for r in _eval_subset_cv(ds, model_id, params, all_features, splits, seed)}

    rows: list[dict] = []
    # Record baseline rows for each explainer + random
    for explainer in (*global_rankings.keys(), "random"):
        for fold, base in base_folds.items():
            rows.append({
                "dataset": ds.id, "model": model_id, "explainer": explainer,
                "fraction": 0.0, "k_removed": 0, "fold": fold,
                "roc_auc": base["roc_auc"], "log_loss": base["log_loss"],
                "accuracy": base["accuracy"],
                "auc_drop": 0.0, "logloss_inc": 0.0,
            })

    # Pre-generate deterministic random permutations
    rng = np.random.default_rng(seed + 777)
    random_orders = [list(rng.permutation(all_features)) for _ in range(n_random_draws)]

    # Cache CV evaluations by frozenset of retained columns to avoid redundant refits
    # when two explainers or two rounding fractions drop the identical feature subset!
    cache: dict[frozenset[str], list[dict]] = {}

    def eval_cached(keep_cols: list[str]) -> list[dict]:
        key = frozenset(keep_cols)
        if key not in cache:
            cache[key] = _eval_subset_cv(ds, model_id, params, keep_cols, splits, seed)
        return cache[key]

    for frac in fractions:
        k_rem = int(max(1, min(d - 1, round(frac * d))))

        for explainer, ranking in global_rankings.items():
            removed = set(ranking[:k_rem])
            keep = [f for f in all_features if f not in removed]
            fold_res = eval_cached(keep)
            for r in fold_res:
                base = base_folds[r["fold"]]
                rows.append({
                    "dataset": ds.id, "model": model_id, "explainer": explainer,
                    "fraction": float(frac), "k_removed": k_rem, "fold": r["fold"],
                    "roc_auc": r["roc_auc"], "log_loss": r["log_loss"],
                    "accuracy": r["accuracy"],
                    "auc_drop": float(base["roc_auc"] - r["roc_auc"]),
                    "logloss_inc": float(r["log_loss"] - base["log_loss"]),
                })

        # Random control averaged over n_random_draws per fold
        fold_accum = {f: {"roc_auc": [], "log_loss": [], "accuracy": []} for f in range(n_splits)}
        for rand_rank in random_orders:
            removed = set(rand_rank[:k_rem])
            keep = [f for f in all_features if f not in removed]
            for r in eval_cached(keep):
                f = r["fold"]
                fold_accum[f]["roc_auc"].append(r["roc_auc"])
                fold_accum[f]["log_loss"].append(r["log_loss"])
                fold_accum[f]["accuracy"].append(r["accuracy"])
        for f in range(n_splits):
            base = base_folds[f]
            mean_auc = float(np.mean(fold_accum[f]["roc_auc"]))
            mean_ll = float(np.mean(fold_accum[f]["log_loss"]))
            mean_acc = float(np.mean(fold_accum[f]["accuracy"]))
            rows.append({
                "dataset": ds.id, "model": model_id, "explainer": "random",
                "fraction": float(frac), "k_removed": k_rem, "fold": f,
                "roc_auc": mean_auc, "log_loss": mean_ll, "accuracy": mean_acc,
                "auc_drop": float(base["roc_auc"] - mean_auc),
                "logloss_inc": float(mean_ll - base["log_loss"]),
            })

    return pd.DataFrame(rows)
