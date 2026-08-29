import json

import pandas as pd
import pytest

from xaibench.config import CELLS, QUICK, paths
from xaibench.pipeline.stage1_performance import evaluate_cell, main

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "brier"]


def test_evaluate_cell_returns_one_row_per_outer_fold():
    df, params = evaluate_cell("heart", "logreg", QUICK, seed=42)
    assert len(df) == 5                     # 5 outer folds
    assert set(METRICS).issubset(df.columns)
    assert isinstance(params, dict)


def test_metrics_are_in_their_valid_ranges():
    df, _ = evaluate_cell("heart", "logreg", QUICK, seed=42)
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert df[metric].between(0.0, 1.0).all(), metric
    assert df["mcc"].between(-1.0, 1.0).all()
    assert df["brier"].between(0.0, 1.0).all()


def test_the_difficulty_gradient_the_study_is_built_on_actually_holds():
    # The three datasets are chosen to span easy -> hard. If this ordering does not
    # hold, H1c (agreement vs difficulty) has no gradient to measure and the framing
    # in the paper must change. Better to learn it here than in Stage 4.
    auc = {ds: evaluate_cell(ds, "rf", QUICK, seed=42)[0]["roc_auc"].mean()
           for ds in ["pima", "heart", "ckd"]}
    assert auc["pima"] < auc["ckd"]
    assert auc["heart"] < auc["ckd"]
    assert auc["ckd"] > 0.95        # CKD is near-separable, protocol risk R2


def test_best_params_keys_are_real_grid_parameters():
    from xaibench.models.zoo import param_grid
    _, params = evaluate_cell("heart", "xgb", QUICK, seed=42)
    assert set(params).issubset(set(param_grid("xgb")))


@pytest.mark.slow
def test_main_writes_every_cell_and_is_idempotent():
    main(force=True)
    p = paths()
    df = pd.read_csv(p.results / "performance.csv")
    assert set(zip(df.dataset, df.model)) == set(CELLS)
    assert len(df) == len(CELLS) * 5
    best = json.loads((p.results / "best_params.json").read_text())
    assert set(best) == {f"{d}|{m}" for d, m in CELLS}
    before = (p.results / "performance.csv").read_text()
    main(force=False)                      # must skip, not recompute
    assert (p.results / "performance.csv").read_text() == before
