"""Stage 6: Post-Review Amendment A8 experiments (ICCIT 2026 conditional acceptance).

Never modifies any pre-registered output in results/ (Stages 1--5 stay intact and
auditable). Writes additive revision artefacts to results/revision_*.csv and
checkpoints every part independently so any single part can be re-run cleanly.

Parts:
  A. redundancy          -- feature collinearity, effective rank, univariate AUC count
  B. conditional_fidelity -- local k-NN conditional deletion fidelity (all 18 cells)
  C. roar                -- global Remove And Retrain 5-fold CV (all 18 cells)
  D. lime_budget         -- 5-budget LIME sweep: 500, 1000, 2500, 5000, 10000
  E. fusion_deepdive     -- calibration-tuned w*, LIME R^2 adaptive/gate, oracle gap
  F. scalability         -- external validation on BCWD (569x30) + CDC Diabetes (5000x21)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample

from xaibench.config import (
    ALPHA,
    AOPC_MAX_K,
    CELLS,
    COMP_SUFF_K,
    DATASETS,
    ENSEMBLE_MODELS,
    JACCARD_KS,
    LIPSCHITZ_SIGMA,
    MODELS,
    RBO_P,
    SEED,
    TEST_SIZE,
    Profile,
    get_profile,
    paths,
)
from xaibench.data.loaders import Dataset, load_dataset
from xaibench.explain.consensus import (
    calibration_split,
    fidelity_weight,
    fuse_weighted,
    random_control,
)
from xaibench.explain.lime_backend import explain_lime
from xaibench.explain.revision_fusion import (
    evaluate_cell_fusion_deepdive,
    extract_lime_local_r2,
)
from xaibench.explain.shap_backend import build_background, explain_shap
from xaibench.metrics.agreement import (
    jaccard_at_k,
    rank_vector,
    ranking_from_attributions,
    rbo_normalised,
    spearman,
)
from xaibench.metrics.conditional_fidelity import (
    conditional_knn_metrics,
    roar_for_cell,
)
from xaibench.metrics.fidelity import batched_curve_metrics, median_baseline
from xaibench.metrics.stability import (
    local_lipschitz,
    repeat_pair_rho,
    rerun_cv,
    retrain_agreement,
    sample_neighbours,
)
from xaibench.models.cells import Cell, _stratified_subsample, build_cell
from xaibench.models.pipelines import (
    build_preprocessor,
    categorical_output_indices,
    onehot_group_map,
)
from xaibench.models.zoo import build_model, shap_backend_kind
from xaibench.pipeline.stage2_explain import transformed_ranking
from xaibench.pipeline.stage3_metrics import (
    _FakeExplanation,
    _attribution_matrix,
    _load_cell_attributions,
    _primary,
)
from xaibench.stats.tests import holm, wilcoxon_with_effect

LIME_BUDGET_GRID = (500, 1000, 2500, 5000, 10000)
ROAR_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50)
COND_KNN_NEIGHBORS = 10
CDC_COHORT_N = 5000


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ external datasets
def load_external_dataset(dataset_id: str) -> Dataset:
    """Load and vendor external scalability datasets with SHA-256 verification."""
    p = paths()
    p.mkdirs()
    checksum_path = p.data_raw / "checksums_revision.json"
    checksums = json.loads(checksum_path.read_text()) if checksum_path.exists() else {}

    if dataset_id == "bcwd":
        csv_path = p.data_raw / "bcwd.csv"
        if not csv_path.exists():
            bc = load_breast_cancer(as_frame=True)
            df = bc.data.copy()
            # Malignant = 1 (disease positive), consistent with PIMA/Heart/CKD.
            df["target"] = (bc.target == 0).astype(int)
            csv_path.write_text(df.to_csv(index=False))
        raw_bytes = csv_path.read_bytes()
        digest = _sha256_bytes(raw_bytes)
        if "bcwd.csv" in checksums:
            if checksums["bcwd.csv"] != digest:
                raise RuntimeError("SHA-256 mismatch for data/raw/bcwd.csv")
        else:
            checksums["bcwd.csv"] = digest
            checksum_path.write_text(json.dumps(checksums, indent=2) + "\n")
        df = pd.read_csv(csv_path)
        y = df.pop("target").astype(int)
        return Dataset(
            id="bcwd", X=df.reset_index(drop=True), y=y.reset_index(drop=True),
            numeric=list(df.columns), categorical=[],
            source_url="https://archive.ics.uci.edu/static/public/17/data.csv",
            licence="CC BY 4.0 (UCI #17 / scikit-learn)",
            citation="Street, Wolberg, and Mangasarian (1993)",
        )

    if dataset_id == "cdc_diabetes":
        csv_path = p.data_raw / "cdc_diabetes_5k.csv"
        if not csv_path.exists():
            url = "https://archive.ics.uci.edu/static/public/891/data.csv"
            full = pd.read_csv(url)
            target_col = "Diabetes_binary" if "Diabetes_binary" in full.columns else full.columns[-1]
            y_full = full[target_col].astype(int)
            X_full = full.drop(columns=[target_col, "ID"], errors="ignore")
            X_sub, _, y_sub, _ = train_test_split(
                X_full, y_full, train_size=CDC_COHORT_N, stratify=y_full, random_state=SEED,
            )
            sub = X_sub.copy()
            sub["Diabetes_binary"] = y_sub.to_numpy(int)
            csv_path.write_text(sub.reset_index(drop=True).to_csv(index=False))
        raw_bytes = csv_path.read_bytes()
        digest = _sha256_bytes(raw_bytes)
        if "cdc_diabetes_5k.csv" in checksums:
            if checksums["cdc_diabetes_5k.csv"] != digest:
                raise RuntimeError("SHA-256 mismatch for data/raw/cdc_diabetes_5k.csv")
        else:
            checksums["cdc_diabetes_5k.csv"] = digest
            checksum_path.write_text(json.dumps(checksums, indent=2) + "\n")
        df = pd.read_csv(csv_path)
        y = df.pop("Diabetes_binary").astype(int)
        return Dataset(
            id="cdc_diabetes", X=df.reset_index(drop=True), y=y.reset_index(drop=True),
            numeric=list(df.columns), categorical=[],
            source_url="https://archive.ics.uci.edu/static/public/891/data.csv",
            licence="CC0 1.0 (CDC BRFSS 2015 / UCI #891)",
            citation="CDC Diabetes Health Indicators (UCI #891)",
        )

    raise ValueError(f"unknown external dataset {dataset_id!r}")


def _build_cell_from_dataset(ds: Dataset, model_id: str, profile: Profile,
                             seed: int = SEED) -> Cell:
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=TEST_SIZE, stratify=ds.y, random_state=seed,
    )
    pre = build_preprocessor(ds).fit(X_tr)
    X_tr_t = np.asarray(pre.transform(X_tr), dtype=float)
    X_te_t = np.asarray(pre.transform(X_te), dtype=float)

    clf = build_model(model_id, seed=seed)
    clf.fit(X_tr_t, np.asarray(y_tr))
    y_te_arr = np.asarray(y_te)
    groups = onehot_group_map(pre, ds)
    return Cell(
        dataset=ds.id, model=model_id, preprocessor=pre, classifier=clf,
        X_train_t=X_tr_t, X_test_t=X_te_t,
        y_train=np.asarray(y_tr), y_test=y_te_arr,
        groups=groups, feature_order=list(ds.X.columns),
        categorical_idx=categorical_output_indices(groups, ds),
        instance_ids=_stratified_subsample(y_te_arr, profile.n_explain, seed),
        X_train_raw=X_tr.reset_index(drop=True), y_train_raw=y_tr.reset_index(drop=True),
    )


# ------------------------------------------------------------------ Part A: Redundancy
def compute_redundancy_metrics(ds: Dataset) -> dict:
    """Quantify inter-feature collinearity, effective rank, and univariate sufficiency."""
    X = ds.X
    y = np.asarray(ds.y, dtype=int)
    num = X.select_dtypes(include=[np.number]).copy()
    for col in X.columns:
        if col not in num.columns:
            codes = pd.Categorical(X[col]).codes.astype(float)
            codes[X[col].isna().to_numpy()] = np.nan
            num[col] = codes
    num = num.fillna(num.median())

    corr_sp = num.corr(method="spearman").abs().to_numpy(copy=True)
    np.fill_diagonal(corr_sp, np.nan)
    off_diag = corr_sp[np.isfinite(corr_sp)]

    corr_p = np.nan_to_num(num.corr().to_numpy(copy=True), nan=0.0)
    np.fill_diagonal(corr_p, 1.0)
    eig = np.maximum(np.linalg.eigvalsh(corr_p), 0.0)
    probs = eig / eig.sum()
    probs = probs[probs > 1e-12]
    eff_rank = float(np.exp(-np.sum(probs * np.log(probs))))

    uni_aucs = []
    for col in num.columns:
        vals = num[col].to_numpy(dtype=float)
        if np.ptp(vals) > 0:
            auc = roc_auc_score(y, vals)
            uni_aucs.append(float(max(auc, 1.0 - auc)))

    return {
        "dataset": ds.id,
        "n_rows": int(len(y)),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "mean_abs_spearman": float(off_diag.mean()) if off_diag.size else 0.0,
        "max_abs_spearman": float(off_diag.max()) if off_diag.size else 0.0,
        "frac_pairs_gt_05": float((off_diag > 0.5).mean()) if off_diag.size else 0.0,
        "effective_rank": eff_rank,
        "effective_rank_ratio": float(eff_rank / max(X.shape[1], 1)),
        "n_uni_auc_ge_080": int(sum(a >= 0.80 for a in uni_aucs)),
        "n_uni_auc_ge_090": int(sum(a >= 0.90 for a in uni_aucs)),
        "max_uni_auc": float(max(uni_aucs)) if uni_aucs else 0.5,
    }


def run_part_redundancy(force: bool = False) -> pd.DataFrame:
    p = paths()
    target = p.results / "revision_redundancy.csv"
    if target.exists() and not force:
        print("[stage6:A] redundancy cached")
        return pd.read_csv(target)
    print("[stage6:A] computing feature redundancy metrics", flush=True)
    rows = [compute_redundancy_metrics(load_dataset(d)) for d in DATASETS]
    for ext in ("bcwd", "cdc_diabetes"):
        rows.append(compute_redundancy_metrics(load_external_dataset(ext)))
    df = pd.DataFrame(rows)
    df.to_csv(target, index=False)
    return df


# ------------------------------------------------------------------ Part B: Conditional k-NN
def run_part_conditional_fidelity(profile: Profile, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = paths()
    out_raw = p.results / "revision_conditional_fidelity.csv"
    out_stats = p.results / "revision_conditional_stats.csv"
    if out_raw.exists() and out_stats.exists() and not force:
        print("[stage6:B] conditional k-NN fidelity cached")
        return pd.read_csv(out_raw), pd.read_csv(out_stats)

    print("[stage6:B] computing local conditional k-NN deletion fidelity (18 cells)", flush=True)
    t0 = time.perf_counter()
    best = json.loads((p.results / "best_params.json").read_text())
    primary_n = profile.lime_num_samples[0]
    rows: list[dict] = []

    for dataset_id, model_id in CELLS:
        cell = build_cell(dataset_id, model_id, best.get(f"{dataset_id}|{model_id}", {}), profile)
        frame = _load_cell_attributions(p, dataset_id, model_id)
        for explainer in ("shap", "lime", "random"):
            vectors = _attribution_matrix(_primary(frame, explainer, primary_n), cell.feature_order)
            for inst_id, attr in vectors.items():
                groups = transformed_ranking(cell, _FakeExplanation(attr, cell.feature_order))
                metrics = conditional_knn_metrics(
                    cell.predict_proba, cell.X_test_t[inst_id], groups, cell.X_train_t,
                    k_max=AOPC_MAX_K, k=COMP_SUFF_K, n_neighbors=COND_KNN_NEIGHBORS,
                    seed=SEED + int(inst_id),
                )
                rows.append({
                    "dataset": dataset_id, "model": model_id,
                    "explainer": explainer, "instance_id": int(inst_id),
                    **metrics,
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_raw, index=False)

    # Paired Wilcoxon tests per cell, Holm-corrected within each comparison family
    stat_rows = []
    comparisons = [
        ("cond_shap_vs_lime", "shap", "lime", "two-sided"),
        ("cond_shap_vs_random", "shap", "random", "greater"),
        ("cond_lime_vs_random", "lime", "random", "greater"),
    ]
    for comp_name, left, right, alt in comparisons:
        for dataset_id, model_id in CELLS:
            sub = df.query("dataset == @dataset_id and model == @model_id")
            wide = sub.pivot_table(index="instance_id", columns="explainer", values="aopc_cond", aggfunc="first")
            both = wide[[left, right]].dropna()
            a, b = both[left].to_numpy(float), both[right].to_numpy(float)
            res = wilcoxon_with_effect(a, b, alternative=alt)
            stat_rows.append({
                "comparison": comp_name, "dataset": dataset_id, "model": model_id,
                "alternative": res.alternative, "n": res.n,
                "median_left": float(np.median(a)), "median_right": float(np.median(b)),
                "median_diff": float(np.median(a - b)),
                "p_value": res.p_value, "effect_size": res.effect_size,
                "degenerate": bool(res.degenerate),
            })
    sdf = pd.DataFrame(stat_rows)
    sdf["p_adj"] = 1.0
    for comp_name, grp in sdf.groupby("comparison"):
        live = grp.index[~grp.degenerate]
        if len(live):
            sdf.loc[live, "p_adj"] = holm(sdf.loc[live, "p_value"].to_numpy(float))
    sdf["decision"] = np.where(sdf.p_adj < ALPHA, "reject H0", "fail to reject H0")
    sdf.to_csv(out_stats, index=False)
    print(f"[stage6:B] done in {time.perf_counter()-t0:.1f}s", flush=True)
    return df, sdf


# ------------------------------------------------------------------ Part C: Global ROAR
def run_part_roar(profile: Profile, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = paths()
    out_raw = p.results / "revision_roar.csv"
    out_stats = p.results / "revision_roar_stats.csv"
    if out_raw.exists() and out_stats.exists() and not force:
        print("[stage6:C] ROAR cached")
        return pd.read_csv(out_raw), pd.read_csv(out_stats)

    print("[stage6:C] computing global ROAR (Remove And Retrain, 5-fold CV, 18 cells)", flush=True)
    t0 = time.perf_counter()
    best = json.loads((p.results / "best_params.json").read_text())
    primary_n = profile.lime_num_samples[0]
    parts = []

    for dataset_id, model_id in CELLS:
        ds = load_dataset(dataset_id)
        frame = _load_cell_attributions(p, dataset_id, model_id)
        rankings: dict[str, list[str]] = {}
        for exp_name in ("shap", "lime"):
            vecs = _attribution_matrix(_primary(frame, exp_name, primary_n), list(ds.X.columns))
            global_imp = np.mean([np.abs(v) for v in vecs.values()], axis=0)
            order = ranking_from_attributions(global_imp)
            rankings[exp_name] = [ds.X.columns[int(idx)] for idx in order]
        part = roar_for_cell(
            ds, model_id, best.get(f"{dataset_id}|{model_id}", {}),
            rankings, fractions=ROAR_FRACTIONS, n_splits=5, n_random_draws=5, seed=SEED,
        )
        parts.append(part)
        print(f"  [roar] {dataset_id:5s} {model_id:6s} done", flush=True)

    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out_raw, index=False)

    # Summarize mean auc_drop and logloss_inc across fractions > 0 per (dataset, model, fold, explainer)
    pos = df.query("fraction > 0")
    fold_summary = (pos.groupby(["dataset", "model", "fold", "explainer"], as_index=False)
                    [["auc_drop", "logloss_inc"]].mean())
    cell_summary = (fold_summary.groupby(["dataset", "model", "explainer"], as_index=False)
                    [["auc_drop", "logloss_inc"]].mean())

    stat_rows = []
    for metric in ("auc_drop", "logloss_inc"):
        for level_name, tbl, idx_cols in (
            ("cell_mean", cell_summary, ["dataset", "model"]),
            ("cell_fold", fold_summary, ["dataset", "model", "fold"]),
        ):
            wide = tbl.pivot_table(index=idx_cols, columns="explainer", values=metric, aggfunc="first")
            for comp, left, right, alt in (
                ("shap_vs_lime", "shap", "lime", "two-sided"),
                ("shap_vs_random", "shap", "random", "greater"),
                ("lime_vs_random", "lime", "random", "greater"),
            ):
                both = wide[[left, right]].dropna()
                a, b = both[left].to_numpy(float), both[right].to_numpy(float)
                res = wilcoxon_with_effect(a, b, alternative=alt)
                stat_rows.append({
                    "metric": metric, "level": level_name, "comparison": comp,
                    "alternative": res.alternative, "n": res.n,
                    "median_left": float(np.median(a)), "median_right": float(np.median(b)),
                    "mean_left": float(np.mean(a)), "mean_right": float(np.mean(b)),
                    "p_value": res.p_value, "effect_size": res.effect_size,
                    "degenerate": bool(res.degenerate),
                })
    sdf = pd.DataFrame(stat_rows)
    sdf["p_adj"] = 1.0
    for (metric, level_name), grp in sdf.groupby(["metric", "level"]):
        live = grp.index[~grp.degenerate]
        if len(live):
            sdf.loc[live, "p_adj"] = holm(sdf.loc[live, "p_value"].to_numpy(float))
    sdf["decision"] = np.where(sdf.p_adj < ALPHA, "reject H0", "fail to reject H0")
    sdf.to_csv(out_stats, index=False)
    print(f"[stage6:C] done in {time.perf_counter()-t0:.1f}s", flush=True)
    return df, sdf


# ------------------------------------------------------------------ Part D: LIME budget sweep
def run_part_lime_budget(profile: Profile, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = paths()
    out_main = p.results / "revision_lime_budget.csv"
    out_rerun = p.results / "revision_lime_budget_rerun.csv"
    if out_main.exists() and out_rerun.exists() and not force:
        print("[stage6:D] LIME budget sweep cached")
        return pd.read_csv(out_main), pd.read_csv(out_rerun)

    print(f"[stage6:D] running LIME sampling budget sweep {LIME_BUDGET_GRID} (18 cells)", flush=True)
    t0 = time.perf_counter()
    best = json.loads((p.results / "best_params.json").read_text())
    existing_rerun = pd.read_parquet(p.results / "stability_rerun.parquet")
    existing_budgets = set(profile.lime_num_samples)

    main_rows: list[dict] = []
    rerun_rows: list[dict] = []

    for dataset_id, model_id in CELLS:
        cell = build_cell(dataset_id, model_id, best.get(f"{dataset_id}|{model_id}", {}), profile)
        baseline = median_baseline(cell.X_train_t)
        frame = _load_cell_attributions(p, dataset_id, model_id)
        shap_vecs = _attribution_matrix(_primary(frame, "shap", profile.lime_num_samples[0]), cell.feature_order)
        shap_rt = float(frame[frame.explainer == "shap"].groupby("instance_id").runtime_s.first().median())
        subset_rerun = cell.instance_ids[: profile.rerun_instances]

        for n_samples in LIME_BUDGET_GRID:
            if n_samples in existing_budgets:
                # Reuse exact Stage 2 attributions, runtimes, and rerun stability!
                sub_f = frame[(frame.explainer == "lime") & (frame.num_samples == n_samples)]
                lime_vecs = _attribution_matrix(sub_f, cell.feature_order)
                rt_series = sub_f.groupby("instance_id").runtime_s.first()
                rt_map = {int(k): float(v) for k, v in rt_series.items()}

                rr = existing_rerun.query(
                    "dataset == @dataset_id and model == @model_id and explainer == 'lime' and num_samples == @n_samples"
                )
                for row in rr.itertuples():
                    rerun_rows.append({
                        "dataset": dataset_id, "model": model_id,
                        "num_samples": int(n_samples), "instance_id": int(row.instance_id),
                        "rerun_cv": float(row.rerun_cv),
                        "repeat_pair_rho": float(row.repeat_pair_rho),
                    })
            else:
                exps = explain_lime(cell, cell.instance_ids, num_samples=n_samples, seed=SEED)
                lime_vecs = {int(e.instance_id): np.asarray(e.attributions, dtype=float) for e in exps}
                rt_map = {int(e.instance_id): float(e.runtime_s) for e in exps}

                runs = np.array([
                    [e.attributions for e in explain_lime(cell, subset_rerun, num_samples=n_samples, seed=1000 + r)]
                    for r in range(profile.rerun_repeats)
                ])
                for i, inst in enumerate(subset_rerun):
                    rerun_rows.append({
                        "dataset": dataset_id, "model": model_id,
                        "num_samples": int(n_samples), "instance_id": int(inst),
                        "rerun_cv": float(rerun_cv(runs[:, i, :])),
                        "repeat_pair_rho": float(repeat_pair_rho(runs[:, i, :])),
                    })

            # Per-instance runtime, AOPC, and Spearman agreement with SHAP
            for inst_id in cell.instance_ids:
                iid = int(inst_id)
                l_vec = lime_vecs[iid]
                s_vec = shap_vecs[iid]
                r_s = ranking_from_attributions(s_vec)
                r_l = ranking_from_attributions(l_vec)
                rho = spearman(rank_vector(r_s), rank_vector(r_l))
                j5 = jaccard_at_k(r_s, r_l, 5)
                groups = transformed_ranking(cell, _FakeExplanation(l_vec, cell.feature_order))
                aopc = batched_curve_metrics(
                    cell.predict_proba, cell.X_test_t[iid], groups, [baseline], AOPC_MAX_K, COMP_SUFF_K,
                )["aopc"]
                main_rows.append({
                    "dataset": dataset_id, "model": model_id,
                    "shap_backend": shap_backend_kind(model_id),
                    "num_samples": int(n_samples), "instance_id": iid,
                    "runtime_s": rt_map[iid], "shap_median_s": shap_rt,
                    "aopc": float(aopc), "spearman_vs_shap": float(rho),
                    "jaccard5_vs_shap": float(j5),
                })

        print(f"  [budget] {dataset_id:5s} {model_id:6s} done ({time.perf_counter()-t0:.0f}s elapsed)", flush=True)

    df_main = pd.DataFrame(main_rows)
    df_rerun = pd.DataFrame(rerun_rows)
    df_main.to_csv(out_main, index=False)
    df_rerun.to_csv(out_rerun, index=False)
    print(f"[stage6:D] done in {time.perf_counter()-t0:.1f}s", flush=True)
    return df_main, df_rerun


# ------------------------------------------------------------------ Part E: Fusion deep-dive
def run_part_fusion_deepdive(profile: Profile, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = paths()
    out_eval = p.results / "revision_fusion_eval.csv"
    out_sum = p.results / "revision_fusion_summary.csv"
    out_stats = p.results / "revision_fusion_stats.csv"
    if out_eval.exists() and out_sum.exists() and out_stats.exists() and not force:
        print("[stage6:E] fusion deep-dive cached")
        return pd.read_csv(out_eval), pd.read_csv(out_sum), pd.read_csv(out_stats)

    print("[stage6:E] running fusion deep-dive (calibration-tuned w*, LIME R^2 adaptive/gate, oracle)", flush=True)
    t0 = time.perf_counter()
    best = json.loads((p.results / "best_params.json").read_text())
    cfg = pd.read_csv(p.results / "explainer_config.csv")
    splits = pd.read_csv(p.results / "consensus_splits.csv")
    primary_n = profile.lime_num_samples[0]

    eval_parts = []
    sum_rows = []
    for dataset_id, model_id in CELLS:
        cell = build_cell(dataset_id, model_id, best.get(f"{dataset_id}|{model_id}", {}), profile)
        frame = _load_cell_attributions(p, dataset_id, model_id)
        shap_vecs = _attribution_matrix(_primary(frame, "shap", primary_n), cell.feature_order)
        lime_vecs = _attribution_matrix(_primary(frame, "lime", primary_n), cell.feature_order)
        lime_r2 = extract_lime_local_r2(cell, cell.instance_ids, num_samples=primary_n, seed=SEED)

        sp_row = splits.query("dataset == @dataset_id and model == @model_id").iloc[0]
        calib_ids = np.array([int(v) for v in str(sp_row.calibration_ids).split()], dtype=int)
        eval_ids = np.array([int(v) for v in str(sp_row.evaluation_ids).split()], dtype=int)
        w_prop = float(cfg.query("dataset == @dataset_id and model == @model_id").iloc[0].fidelity_weight_shap)

        df_ev, summary = evaluate_cell_fusion_deepdive(
            cell, shap_vecs, lime_vecs, lime_r2, calib_ids, eval_ids, w_prop,
        )
        eval_parts.append(df_ev)
        sum_rows.append(summary)
        print(f"  [fusion] {dataset_id:5s} {model_id:6s} w_prop={w_prop:.3f} w_opt={summary['w_opt']:.2f} "
              f"lime_win={summary['lime_win_rate']:.2f} oracle_gap={summary['oracle_headroom']:.4f}", flush=True)

    df_eval = pd.concat(eval_parts, ignore_index=True)
    df_sum = pd.DataFrame(sum_rows)
    df_eval.to_csv(out_eval, index=False)
    df_sum.to_csv(out_sum, index=False)

    # Hypothesis tests per rule vs (1) cal_best_single (deployable bar) and (2) oracle (instance max)
    stat_rows = []
    rules = ("rankavg", "rrf", "fidwt", "grid_interior", "grid_opt", "r2_adaptive", "r2_gate")
    for bar in ("cal_best_single", "oracle"):
        for rule in rules:
            for dataset_id, model_id in CELLS:
                sub = df_eval.query("dataset == @dataset_id and model == @model_id")
                a = sub[rule].to_numpy(float)
                b = sub[bar].to_numpy(float)
                res = wilcoxon_with_effect(a, b, alternative="greater")
                stat_rows.append({
                    "bar": bar, "rule": rule, "dataset": dataset_id, "model": model_id,
                    "n": res.n, "median_rule": float(np.median(a)), "median_bar": float(np.median(b)),
                    "mean_diff": float(np.mean(a - b)), "median_diff": float(np.median(a - b)),
                    "p_value": res.p_value, "effect_size": res.effect_size,
                    "degenerate": bool(res.degenerate),
                })
    sdf = pd.DataFrame(stat_rows)
    sdf["p_adj"] = 1.0
    for (bar, rule), grp in sdf.groupby(["bar", "rule"]):
        live = grp.index[~grp.degenerate]
        if len(live):
            sdf.loc[live, "p_adj"] = holm(sdf.loc[live, "p_value"].to_numpy(float))
    sdf["decision"] = np.where(sdf.p_adj < ALPHA, "reject H0", "fail to reject H0")
    sdf.to_csv(out_stats, index=False)
    print(f"[stage6:E] done in {time.perf_counter()-t0:.1f}s", flush=True)
    return df_eval, df_sum, sdf


# ------------------------------------------------------------------ Part F: External scalability
def run_part_scalability(profile: Profile, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = paths()
    out_cells = p.results / "revision_scalability_cells.csv"
    out_stats = p.results / "revision_scalability_stats.csv"
    if out_cells.exists() and out_stats.exists() and not force:
        print("[stage6:F] external scalability cached")
        return pd.read_csv(out_cells), pd.read_csv(out_stats)

    print("[stage6:F] running external scalability validation (BCWD 569x30 + CDC Diabetes 5000x21)", flush=True)
    t0 = time.perf_counter()
    primary_n = profile.lime_num_samples[0]
    ext_datasets = ("bcwd", "cdc_diabetes")
    cell_rows = []
    test_rows = []

    for ds_id in ext_datasets:
        ds = load_external_dataset(ds_id)
        for model_id in MODELS:
            cell_t0 = time.perf_counter()
            cell = _build_cell_from_dataset(ds, model_id, profile, seed=SEED)
            ids = cell.instance_ids
            test_auc = float(roc_auc_score(cell.y_test, cell.predict_proba(cell.X_test_t)[:, 1]))

            bg = build_background(cell, profile.shap_background, SEED)
            baseline = median_baseline(cell.X_train_t)

            shap_exps = explain_shap(cell, ids, seed=SEED, background=bg)
            lime_exps = explain_lime(cell, ids, num_samples=primary_n, seed=SEED)
            rand_exps = random_control(cell, ids, seed=SEED)

            # Agreement per instance
            rhos, j5s, rbos = [], [], []
            for s, l in zip(shap_exps, lime_exps):
                rs = ranking_from_attributions(s.attributions)
                rl = ranking_from_attributions(l.attributions)
                rhos.append(spearman(rank_vector(rs), rank_vector(rl)))
                j5s.append(jaccard_at_k(rs, rl, 5))
                rbos.append(rbo_normalised(rs, rl, p=RBO_P))
            rhos_arr = np.asarray(rhos, dtype=float)

            # Global agreement
            g_s = np.mean([np.abs(e.attributions) for e in shap_exps], axis=0)
            g_l = np.mean([np.abs(e.attributions) for e in lime_exps], axis=0)
            global_rho = float(spearman(rank_vector(ranking_from_attributions(g_s)),
                                        rank_vector(ranking_from_attributions(g_l))))

            # Fidelity (median AND conditional k-NN)
            aopc_med = {"shap": [], "lime": [], "random": []}
            aopc_cond = {"shap": [], "lime": [], "random": []}
            for name, exps in (("shap", shap_exps), ("lime", lime_exps), ("random", rand_exps)):
                for e in exps:
                    groups = transformed_ranking(cell, e)
                    x = cell.X_test_t[e.instance_id]
                    aopc_med[name].append(batched_curve_metrics(
                        cell.predict_proba, x, groups, [baseline], AOPC_MAX_K, COMP_SUFF_K,
                    )["aopc"])
                    aopc_cond[name].append(conditional_knn_metrics(
                        cell.predict_proba, x, groups, cell.X_train_t,
                        k_max=AOPC_MAX_K, k=COMP_SUFF_K, n_neighbors=COND_KNN_NEIGHBORS,
                        seed=SEED + int(e.instance_id),
                    )["aopc_cond"])

            s_med, l_med, r_med = (np.asarray(aopc_med[k], float) for k in ("shap", "lime", "random"))
            s_cnd, l_cnd, r_cnd = (np.asarray(aopc_cond[k], float) for k in ("shap", "lime", "random"))

            # Fusion on disjoint 30/70 split
            calib_ids, eval_ids = calibration_split(ids, seed=SEED)
            pos = {int(iid): idx for idx, iid in enumerate(ids)}
            cal_idx = [pos[int(i)] for i in calib_ids]
            ev_idx = [pos[int(i)] for i in eval_ids]
            w_prop = fidelity_weight(float(np.mean(s_med[cal_idx])), float(np.mean(l_med[cal_idx])))
            fidwt_ev = []
            for idx in ev_idx:
                fused_attr = fuse_weighted(shap_exps[idx].attributions, lime_exps[idx].attributions, w_prop)
                groups = transformed_ranking(cell, _FakeExplanation(fused_attr, cell.feature_order))
                fidwt_ev.append(batched_curve_metrics(
                    cell.predict_proba, cell.X_test_t[int(ids[idx])], groups, [baseline],
                    AOPC_MAX_K, COMP_SUFF_K,
                )["aopc"])
            fidwt_arr = np.asarray(fidwt_ev, float)
            best_single_ev = s_med[ev_idx] if np.mean(s_med[cal_idx]) >= np.mean(l_med[cal_idx]) else l_med[ev_idx]
            oracle_ev = np.maximum(s_med[ev_idx], l_med[ev_idx])

            # Stability: re-run CV (subset=15, R=5 for speed on external datasets) + local Lipschitz (subset=15, m=5)
            n_stab = min(15, ids.size)
            sub_stab = ids[:n_stab]
            lime_runs = np.array([
                [e.attributions for e in explain_lime(cell, sub_stab, num_samples=primary_n, seed=1000 + r)]
                for r in range(5)
            ])
            lime_cvs = np.array([rerun_cv(lime_runs[:, i, :]) for i in range(n_stab)], float)
            shap_runs = np.array([
                [e.attributions for e in explain_shap(cell, sub_stab, seed=1000 + r, background=bg)]
                for r in range(5)
            ])
            shap_cvs = np.array([rerun_cv(shap_runs[:, i, :]) for i in range(n_stab)], float)

            # Local Lipschitz
            sigma = cell.train_sigma * LIPSCHITZ_SIGMA
            n_test = cell.X_test_t.shape[0]
            lip_s, lip_l = [], []
            for inst in sub_stab:
                x = cell.X_test_t[int(inst)]
                nbrs = sample_neighbours(x, sigma, profile.robust_neighbours,
                                         seed=SEED + int(inst), fixed_mask=cell.fixed_mask)
                probe = replace(cell, X_test_t=np.vstack([cell.X_test_t, nbrs]))
                n_ids = np.arange(n_test, n_test + nbrs.shape[0])
                ps_x = explain_shap(probe, np.array([int(inst)]), seed=SEED, background=bg)[0].attributions
                ps_n = np.array([e.attributions for e in explain_shap(probe, n_ids, seed=SEED, background=bg)])
                lip_s.append(local_lipschitz(ps_x, x, ps_n, nbrs))
                pl_x = explain_lime(probe, np.array([int(inst)]), seed=SEED, num_samples=primary_n)[0].attributions
                pl_n = np.array([e.attributions for e in explain_lime(probe, n_ids, seed=SEED, num_samples=primary_n)])
                lip_l.append(local_lipschitz(pl_x, x, pl_n, nbrs))
            lip_s_arr, lip_l_arr = np.asarray(lip_s, float), np.asarray(lip_l, float)

            # Retraining stability (ensembles only, B=5 resamples for speed)
            retrain_s = float("nan")
            retrain_l = float("nan")
            if model_id in ENSEMBLE_MODELS:
                mat_s, mat_l = [], []
                for b in range(5):
                    ridx = resample(np.arange(cell.X_train_t.shape[0]), replace=True,
                                    n_samples=cell.X_train_t.shape[0], stratify=cell.y_train,
                                    random_state=2000 + b)
                    clf_b = build_model(model_id, seed=SEED).fit(cell.X_train_t[ridx], cell.y_train[ridx])
                    refit = replace(cell, classifier=clf_b)
                    bg_b = build_background(refit, profile.shap_background, SEED)
                    es = explain_shap(refit, sub_stab, seed=SEED, background=bg_b)
                    el = explain_lime(refit, sub_stab, seed=SEED, num_samples=primary_n)
                    mat_s.append(np.mean([np.abs(e.attributions) for e in es], axis=0))
                    mat_l.append(np.mean([np.abs(e.attributions) for e in el], axis=0))
                retrain_s = float(retrain_agreement(np.asarray(mat_s)))
                retrain_l = float(retrain_agreement(np.asarray(mat_l)))

            rt_s = np.array([e.runtime_s for e in shap_exps], float)
            rt_l = np.array([e.runtime_s for e in lime_exps], float)

            cell_rows.append({
                "dataset": ds_id, "model": model_id, "n_rows": len(ds.y),
                "n_features": ds.n_features, "n_explained": int(ids.size),
                "test_roc_auc": test_auc, "shap_backend": shap_backend_kind(model_id),
                "rho_median": float(np.median(rhos_arr)), "rho_global": global_rho,
                "jaccard5_median": float(np.median(j5s)), "rbo_norm_median": float(np.median(rbos)),
                "aopc_med_shap": float(np.median(s_med)), "aopc_med_lime": float(np.median(l_med)),
                "aopc_med_random": float(np.median(r_med)),
                "aopc_cond_shap": float(np.median(s_cnd)), "aopc_cond_lime": float(np.median(l_cnd)),
                "aopc_cond_random": float(np.median(r_cnd)),
                "shap_s_median": float(np.median(rt_s)), "lime_s_median": float(np.median(rt_l)),
                "speed_factor": float(np.median(rt_l) / max(np.median(rt_s), 1e-12)),
                "rerun_cv_shap": float(np.median(shap_cvs)), "rerun_cv_lime": float(np.median(lime_cvs)),
                "lipschitz_shap": float(np.median(lip_s_arr)), "lipschitz_lime": float(np.median(lip_l_arr)),
                "retrain_shap": retrain_s, "retrain_lime": retrain_l,
                "w_prop": float(w_prop),
                "fidwt_eval_median": float(np.median(fidwt_arr)),
                "best_single_eval_median": float(np.median(best_single_ev)),
                "oracle_headroom": float(np.mean(oracle_ev) - np.mean(best_single_ev)),
            })

            # Paired tests per external cell
            for hname, res in (
                ("H1_rho_gt_050", wilcoxon_with_effect(rhos_arr, mu=0.50, alternative="greater")),
                ("H2b_lipschitz", wilcoxon_with_effect(lip_s_arr, lip_l_arr, alternative="two-sided")),
                ("H3a_med_aopc", wilcoxon_with_effect(s_med, l_med, alternative="two-sided")),
                ("H3a_cond_aopc", wilcoxon_with_effect(s_cnd, l_cnd, alternative="two-sided")),
                ("H3b_shap_vs_rand", wilcoxon_with_effect(s_med, r_med, alternative="greater")),
                ("H3b_lime_vs_rand", wilcoxon_with_effect(l_med, r_med, alternative="greater")),
                ("H4_cost", wilcoxon_with_effect(rt_s, rt_l, alternative="less")),
                ("H6_fidwt_vs_best", wilcoxon_with_effect(fidwt_arr, best_single_ev, alternative="greater")),
            ):
                test_rows.append({
                    "hypothesis": hname, "dataset": ds_id, "model": model_id,
                    "alternative": res.alternative, "n": res.n,
                    "p_value": res.p_value, "effect_size": res.effect_size,
                    "median_diff": res.median_diff, "degenerate": bool(res.degenerate),
                })

            print(f"  [scale] {ds_id:12s} {model_id:6s} auc={test_auc:.3f} rho={np.median(rhos_arr):.2f} "
                  f"aopc(S/L)={np.median(s_med):.3f}/{np.median(l_med):.3f} "
                  f"cond(S/L)={np.median(s_cnd):.3f}/{np.median(l_cnd):.3f} "
                  f"({time.perf_counter()-cell_t0:.1f}s)", flush=True)

    cdf = pd.DataFrame(cell_rows)
    tdf = pd.DataFrame(test_rows)
    tdf["p_adj"] = 1.0
    for hname, grp in tdf.groupby("hypothesis"):
        live = grp.index[~grp.degenerate]
        if len(live):
            tdf.loc[live, "p_adj"] = holm(tdf.loc[live, "p_value"].to_numpy(float))
    tdf["decision"] = np.where(tdf.p_adj < ALPHA, "reject H0", "fail to reject H0")
    cdf.to_csv(out_cells, index=False)
    tdf.to_csv(out_stats, index=False)
    print(f"[stage6:F] done in {time.perf_counter()-t0:.1f}s", flush=True)
    return cdf, tdf


def main(parts: list[str] | None = None, force: bool = False) -> None:
    profile = get_profile("FULL")
    all_parts = ["redundancy", "conditional", "roar", "fusion", "budget", "scalability"]
    selected = parts if parts else all_parts
    for part in selected:
        if part == "redundancy":
            run_part_redundancy(force=force)
        elif part == "conditional":
            run_part_conditional_fidelity(profile, force=force)
        elif part == "roar":
            run_part_roar(profile, force=force)
        elif part == "fusion":
            run_part_fusion_deepdive(profile, force=force)
        elif part == "budget":
            run_part_lime_budget(profile, force=force)
        elif part == "scalability":
            run_part_scalability(profile, force=force)
        else:
            raise ValueError(f"unknown part {part!r}; expected subset of {all_parts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(parts=args.parts, force=args.force)
