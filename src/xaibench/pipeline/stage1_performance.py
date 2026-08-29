"""Track A: nested cross-validation for unbiased performance estimation.

Outer 5-fold estimates performance; inner 3-fold selects hyperparameters. Every
transformer lives inside the Pipeline, so imputation and scaling are fitted on the
inner training rows only. This is the leak the two-track design exists to prevent.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from xaibench.config import CELLS, SEED, Profile, get_profile, paths
from xaibench.data.loaders import load_dataset
from xaibench.models.pipelines import build_preprocessor
from xaibench.models.zoo import build_model, param_grid

OUTER_FOLDS, INNER_FOLDS = 5, 3       # NORMATIVE (protocol §3)


def _score(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
    }


def evaluate_cell(dataset_id: str, model_id: str, profile: Profile,
                  seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    ds = load_dataset(dataset_id)
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
    grid = {f"clf__{k}": v for k, v in param_grid(model_id).items()}

    rows, winners = [], []
    for fold, (train_idx, test_idx) in enumerate(outer.split(ds.X, ds.y)):
        pipe = Pipeline([("pre", build_preprocessor(ds)),
                         ("clf", build_model(model_id, seed=seed))])
        search = GridSearchCV(
            pipe, grid,
            cv=StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed),
            scoring="roc_auc", n_jobs=-1, refit=True, error_score="raise")
        search.fit(ds.X.iloc[train_idx], ds.y.iloc[train_idx])

        y_true = ds.y.iloc[test_idx]
        y_prob = search.predict_proba(ds.X.iloc[test_idx])[:, 1]
        rows.append({"dataset": dataset_id, "model": model_id, "fold": fold,
                     **_score(y_true, search.predict(ds.X.iloc[test_idx]), y_prob)})
        winners.append(tuple(sorted(
            (k.removeprefix("clf__"), v) for k, v in search.best_params_.items())))

    # Modal winner across outer folds: the single configuration Track B refits with.
    best = dict(Counter(winners).most_common(1)[0][0]) if winners else {}
    return pd.DataFrame(rows), best


def main(force: bool = False) -> None:
    profile, p = get_profile(), paths()
    p.mkdirs()
    out_csv = p.results / "performance.csv"
    out_json = p.results / "best_params.json"
    if out_csv.exists() and out_json.exists() and not force:
        print(f"[stage1] {out_csv.name} exists; skipping (use --force to recompute)")
        return

    frames, best_params = [], {}
    for dataset_id, model_id in CELLS:
        print(f"[stage1] {dataset_id:6s} {model_id:7s} nested CV "
              f"({OUTER_FOLDS}x{INNER_FOLDS})", flush=True)
        df, best = evaluate_cell(dataset_id, model_id, profile)
        frames.append(df)
        best_params[f"{dataset_id}|{model_id}"] = best

    pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(best_params, indent=2, sort_keys=True) + "\n")
    print(f"[stage1] wrote {out_csv} and {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    main(**vars(parser.parse_args()))
