"""Stage 3 metric outputs. Planned in the implementation plan but not written during
execution, so the metric aggregation layer had no test of its own."""
import pandas as pd
import pytest

from xaibench.config import ALL_EXPLAINERS, CELLS, paths

METRIC_FILES = ["agreement", "fidelity", "plausibility", "runtime", "stability", "curves"]


@pytest.fixture(scope="module")
def outputs():
    p = paths()
    missing = [n for n in METRIC_FILES if not (p.results / f"metrics_{n}.csv").exists()]
    if missing:
        pytest.skip(f"stage3 outputs absent: {missing}")
    return {n: pd.read_csv(p.results / f"metrics_{n}.csv") for n in METRIC_FILES}


def test_agreement_covers_every_cell_and_stays_in_range(outputs):
    df = outputs["agreement"]
    assert set(zip(df.dataset, df.model)) == set(CELLS)
    assert df.spearman.between(-1, 1).all()
    assert df.kendall.between(-1, 1).all()
    for column in ("jaccard3", "jaccard5", "rbo", "rbo_norm"):
        assert df[column].between(0, 1).all(), column


def test_a_global_row_exists_per_cell_alongside_the_per_instance_rows(outputs):
    df = outputs["agreement"]
    globals_ = df.query("instance_id == -1")
    assert set(zip(globals_.dataset, globals_.model)) == set(CELLS)


def test_fidelity_covers_every_explainer_including_the_control(outputs):
    df = outputs["fidelity"]
    assert set(df.explainer) == set(ALL_EXPLAINERS)
    assert set(df.baseline) == {"median", "marginal"}
    assert df.is_evaluation.dtype == bool


def test_real_explainers_beat_the_random_control_on_average(outputs):
    # A sanity check, not a hypothesis test. If SHAP does not beat a random ranking the
    # bug is in the ranking lift or the perturbation loop, not in SHAP.
    means = outputs["fidelity"].query("baseline == 'median'").groupby("explainer").aopc.mean()
    assert means["shap"] > means["random"]
    assert means["lime"] > means["random"]


def test_runtime_is_positive_and_lime_costs_more_than_tree_shap(outputs):
    df = outputs["runtime"]
    assert (df.median_seconds > 0).all()
    tree = df.query("model in ['rf','xgb','lgbm','dtree'] and explainer == 'shap'")
    lime = df.query("model in ['rf','xgb','lgbm','dtree'] and explainer == 'lime'")
    assert lime.median_seconds.median() > tree.median_seconds.median()


def test_plausibility_scores_are_bounded(outputs):
    df = outputs["plausibility"]
    assert df.jaccard5.between(0, 1).all()
    assert df.precision5.between(0, 1).all()
    assert {"shap", "lime", "random"}.issubset(set(df.explainer))


def test_stability_reports_all_three_components(outputs):
    df = outputs["stability"]
    assert {"rerun_cv", "repeat_pair_rho", "lipschitz", "retrain_agreement"} <= set(df.columns)
    tree_shap = df.query("explainer == 'shap' and model in ['rf','xgb','lgbm','dtree']")
    assert (tree_shap.rerun_cv.fillna(0) == 0).all(), "tree SHAP must be deterministic"


def test_all_deletion_curves_start_from_the_same_prediction(outputs):
    # k=0 is the unperturbed instance, so the ranking cannot have had any effect yet.
    df = outputs["curves"]
    for (dataset, model), group in df.groupby(["dataset", "model"]):
        first = group.query("k == 0").mean_prob
        assert first.max() - first.min() < 1e-9, f"{dataset}/{model}: k=0 differs"


def test_curves_converge_only_where_the_depth_reaches_every_feature(outputs):
    """Convergence is expected only when k_max equals the feature count.

    AOPC_MAX_K is 10, so on PIMA (8 features) the curve deletes everything and the
    ranking stops mattering; on Heart (13) and CKD (24) it stops at 10 of d, where
    different rankings have deleted different subsets and divergence is correct.
    """
    from xaibench.data.loaders import load_dataset
    df = outputs["curves"]
    for (dataset, model), group in df.groupby(["dataset", "model"]):
        depth = int(group.k.max())
        if depth < load_dataset(dataset).n_features:
            continue
        final = group.query("k == @depth").mean_prob
        assert final.max() - final.min() < 1e-9, (
            f"{dataset}/{model}: every feature is deleted at k={depth}, so the curves "
            "must meet")


def test_random_control_curve_falls_slower_than_shap(outputs):
    df = outputs["curves"]
    at_k1 = df.query("k == 1").pivot_table(index=["dataset", "model"],
                                           columns="explainer", values="mean_prob")
    assert (at_k1["random"] > at_k1["shap"]).mean() > 0.8
