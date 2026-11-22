"""Deep-dive analysis of rank-level explanation fusion (Post-Review Amendment A8).

Addresses Reviewer #1 Point 2: proportional AOPC weighting
w = AOPC_SHAP / (AOPC_SHAP + AOPC_LIME) naturally compresses near 0.50 (observed
0.504--0.652 across our 18 cells) and behaves similarly to equal-rank averaging.

This module tests stronger, non-proportional, and instance-adaptive fusion rules
under the exact same leakage-safe 30%/70% calibration/evaluation split:
  1. `grid_opt`: cell-level weight w* in {0.0, 0.05, ..., 1.0} maximizing
     calibration AOPC (allowed to collapse to 1.0 or 0.0).
  2. `grid_interior`: cell-level weight w* in {0.1, 0.2, ..., 0.9} (strict mixing).
  3. `r2_adaptive`: instance-level adaptive weight modulated by LIME's local
     surrogate R^2 (`exp.score`), available at inference time with zero labels:
       w(x) = clip(w0 - gamma * (R^2(x) - R^2_cal_mean), 0, 1)
     with (w0, gamma) tuned strictly on the 30% calibration split.
  4. `r2_gate`: instance-level selective routing (use LIME if R^2(x) >= tau*,
     else SHAP), with tau* tuned strictly on the 30% calibration split.

Evaluates every rule against BOTH:
  - Realistic deployable bar: the single explainer selected on the calibration split
    (`cal_best_single`), and
  - Theoretical upper bound: the per-instance oracle max(AOPC_SHAP(x), AOPC_LIME(x)),
    reporting instance win-rates and oracle headroom.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from xaibench.config import AOPC_MAX_K, COMP_SUFF_K, SEED
from xaibench.explain.consensus import fuse_rankavg, fuse_rrf, fuse_weighted
from xaibench.explain.lime_backend import DISCRETIZE_CONTINUOUS, build_lime_explainer
from xaibench.metrics.agreement import ranking_from_attributions
from xaibench.metrics.fidelity import batched_curve_metrics, median_baseline
from xaibench.models.pipelines import aggregate_attributions


def extract_lime_local_r2(
    cell,
    instance_ids=None,
    num_samples: int = 5000,
    seed: int = SEED,
    discretize: bool = DISCRETIZE_CONTINUOUS,
) -> dict[int, float]:
    """Compute LIME's local surrogate R^2 (`exp.score`) per instance.

    Uses the exact same explainer construction, instance iteration order, and RNG
    seed as Stage 2 (`explain_lime`) so the surrogate fits match identically.
    """
    ids = np.asarray(cell.instance_ids if instance_ids is None else instance_ids, dtype=int)
    n_out = int(cell.X_train_t.shape[1])
    explainer = build_lime_explainer(cell, seed, discretize)
    X_test = np.asarray(cell.X_test_t, dtype=float)
    r2_map: dict[int, float] = {}
    for inst in ids:
        x = X_test[int(inst)]
        exp = explainer.explain_instance(
            data_row=x, predict_fn=cell.classifier.predict_proba,
            num_features=n_out, num_samples=int(num_samples), labels=(1,),
        )
        score = getattr(exp, "score", np.nan)
        r2_map[int(inst)] = float(score) if score is not None and np.isfinite(score) else 0.0
    return r2_map


def _aopc_for_attr(cell, instance_id: int, attr: np.ndarray, baseline: np.ndarray) -> float:
    ranking = ranking_from_attributions(attr)
    groups = np.asarray(cell.groups)
    col_groups = [np.flatnonzero(groups == cell.feature_order[int(i)]) for i in ranking]
    return float(batched_curve_metrics(
        cell.predict_proba, cell.X_test_t[int(instance_id)],
        col_groups, [baseline], AOPC_MAX_K, COMP_SUFF_K,
    )["aopc"])


def evaluate_cell_fusion_deepdive(
    cell,
    shap_vecs: dict[int, np.ndarray],
    lime_vecs: dict[int, np.ndarray],
    lime_r2: dict[int, float],
    calib_ids: np.ndarray,
    eval_ids: np.ndarray,
    w_prop: float,
) -> tuple[pd.DataFrame, dict]:
    """Tune advanced fusion rules on `calib_ids` and evaluate on `eval_ids`."""
    baseline = median_baseline(cell.X_train_t)
    calib_ids = [int(i) for i in calib_ids if int(i) in shap_vecs and int(i) in lime_vecs]
    eval_ids = [int(i) for i in eval_ids if int(i) in shap_vecs and int(i) in lime_vecs]

    # Precompute base AOPCs on calibration split
    cal_shap = {i: _aopc_for_attr(cell, i, shap_vecs[i], baseline) for i in calib_ids}
    cal_lime = {i: _aopc_for_attr(cell, i, lime_vecs[i], baseline) for i in calib_ids}
    cal_best_single = "shap" if np.mean(list(cal_shap.values())) >= np.mean(list(cal_lime.values())) else "lime"

    # 1. Calibration grid search over w in [0.0, 1.0] (step 0.05)
    weight_grid = [round(w, 2) for w in np.linspace(0.0, 1.0, 21)]
    cal_scores_by_w: dict[float, float] = {}
    for w in weight_grid:
        if w == 1.0:
            cal_scores_by_w[w] = float(np.mean(list(cal_shap.values())))
        elif w == 0.0:
            cal_scores_by_w[w] = float(np.mean(list(cal_lime.values())))
        else:
            cal_scores_by_w[w] = float(np.mean([
                _aopc_for_attr(cell, i, fuse_weighted(shap_vecs[i], lime_vecs[i], w), baseline)
                for i in calib_ids
            ]))
    w_opt = max(weight_grid, key=lambda w: cal_scores_by_w[w])
    interior_grid = [w for w in weight_grid if 0.1 <= w <= 0.9]
    w_interior = max(interior_grid, key=lambda w: cal_scores_by_w[w])

    # 2. Instance-level R^2-adaptive weighted fusion tuned on calibration split
    r2_cal_vals = np.array([lime_r2.get(i, 0.0) for i in calib_ids], dtype=float)
    r2_mean = float(np.mean(r2_cal_vals)) if r2_cal_vals.size else 0.0
    r2_std = float(np.std(r2_cal_vals)) if r2_cal_vals.size else 1.0
    r2_scale = r2_std if r2_std > 1e-8 else 1.0

    best_adapt_score = -np.inf
    best_w0, best_gamma = w_opt, 0.0
    for w0 in (0.5, 0.65, 0.8, 0.9, 1.0):
        for gamma in (0.0, 0.15, 0.30, 0.50):
            vals = []
            for i in calib_ids:
                z = (lime_r2.get(i, 0.0) - r2_mean) / r2_scale
                wi = float(np.clip(w0 - gamma * z, 0.0, 1.0))
                vals.append(_aopc_for_attr(
                    cell, i, fuse_weighted(shap_vecs[i], lime_vecs[i], wi), baseline
                ))
            sc = float(np.mean(vals))
            if sc > best_adapt_score:
                best_adapt_score = sc
                best_w0, best_gamma = w0, gamma

    # 3. Instance-level R^2 selective routing (gate) tuned on calibration split
    # Candidate thresholds: quantiles of calibration R^2 + +inf (always SHAP) + -inf (always LIME)
    quantiles = np.unique(np.quantile(r2_cal_vals, [0.25, 0.50, 0.75, 0.90])) if r2_cal_vals.size else np.array([])
    tau_candidates = [float("inf"), float("-inf")] + [float(q) for q in quantiles]
    best_gate_score = -np.inf
    best_tau = float("inf")
    for tau in tau_candidates:
        vals = [cal_lime[i] if lime_r2.get(i, 0.0) >= tau else cal_shap[i] for i in calib_ids]
        sc = float(np.mean(vals))
        if sc > best_gate_score:
            best_gate_score = sc
            best_tau = tau

    # Evaluate every method on the disjoint 70% evaluation split
    eval_rows: list[dict] = []
    for i in eval_ids:
        s_vec, l_vec = shap_vecs[i], lime_vecs[i]
        r2_val = float(lime_r2.get(i, 0.0))
        z = (r2_val - r2_mean) / r2_scale
        w_adapt = float(np.clip(best_w0 - best_gamma * z, 0.0, 1.0))

        aopc_s = _aopc_for_attr(cell, i, s_vec, baseline)
        aopc_l = _aopc_for_attr(cell, i, l_vec, baseline)
        aopc_rankavg = _aopc_for_attr(cell, i, fuse_rankavg(s_vec, l_vec), baseline)
        aopc_rrf = _aopc_for_attr(cell, i, fuse_rrf(s_vec, l_vec), baseline)
        aopc_fidwt = _aopc_for_attr(cell, i, fuse_weighted(s_vec, l_vec, w_prop), baseline)
        aopc_grid = aopc_s if w_opt == 1.0 else (
            aopc_l if w_opt == 0.0 else _aopc_for_attr(cell, i, fuse_weighted(s_vec, l_vec, w_opt), baseline)
        )
        aopc_interior = _aopc_for_attr(cell, i, fuse_weighted(s_vec, l_vec, w_interior), baseline)
        aopc_adapt = _aopc_for_attr(cell, i, fuse_weighted(s_vec, l_vec, w_adapt), baseline)
        aopc_gate = aopc_l if r2_val >= best_tau else aopc_s
        aopc_cal_best = aopc_s if cal_best_single == "shap" else aopc_l

        eval_rows.append({
            "dataset": cell.dataset, "model": cell.model, "instance_id": int(i),
            "lime_r2": r2_val,
            "shap": aopc_s, "lime": aopc_l,
            "cal_best_single": aopc_cal_best,
            "oracle": max(aopc_s, aopc_l),
            "rankavg": aopc_rankavg, "rrf": aopc_rrf, "fidwt": aopc_fidwt,
            "grid_opt": aopc_grid, "grid_interior": aopc_interior,
            "r2_adaptive": aopc_adapt, "r2_gate": aopc_gate,
        })

    df_eval = pd.DataFrame(eval_rows)
    diff_ls = (df_eval["lime"] - df_eval["shap"]).to_numpy(float)
    r2_arr = df_eval["lime_r2"].to_numpy(float)
    if np.ptp(r2_arr) > 0 and np.ptp(diff_ls) > 0:
        r2_corr = float(sps.spearmanr(r2_arr, diff_ls).statistic)
    else:
        r2_corr = float("nan")

    summary = {
        "dataset": cell.dataset, "model": cell.model,
        "n_calib": len(calib_ids), "n_eval": len(eval_ids),
        "cal_best_single": cal_best_single,
        "w_prop": float(w_prop), "w_opt": float(w_opt),
        "w_interior": float(w_interior),
        "adapt_w0": float(best_w0), "adapt_gamma": float(best_gamma),
        "gate_tau": float(best_tau),
        "lime_r2_median": float(np.median(r2_arr)),
        "r2_vs_lime_adv_rho": r2_corr,
        "shap_win_rate": float((df_eval["shap"] > df_eval["lime"]).mean()),
        "lime_win_rate": float((df_eval["lime"] > df_eval["shap"]).mean()),
        "tie_rate": float((df_eval["shap"] == df_eval["lime"]).mean()),
        "oracle_headroom": float(df_eval["oracle"].mean() - df_eval["cal_best_single"].mean()),
    }
    return df_eval, summary
