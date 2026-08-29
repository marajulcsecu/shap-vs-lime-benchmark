import numpy as np
import pytest

from xaibench.config import ALL_EXPLAINERS, ENSEMBLE_MODELS, QUICK, paths
from xaibench.pipeline.stage2_explain import explain_cell, main, transformed_ranking


@pytest.fixture(scope="module")
def artifacts():
    return explain_cell("pima", "rf", params={}, profile=QUICK)


def test_every_explainer_produced_the_same_instance_coverage(artifacts):
    df = artifacts.attributions
    assert set(df.explainer) == set(ALL_EXPLAINERS)
    counts = df.groupby("explainer").instance_id.nunique()
    assert counts.nunique() == 1, f"uneven coverage: {counts.to_dict()}"


def test_attributions_are_in_original_feature_space(artifacts):
    df = artifacts.attributions
    assert df.feature.nunique() == 8                      # pima has 8 features
    per_explanation = df.groupby(["explainer", "instance_id", "num_samples"]).size()
    assert (per_explanation == 8).all()


def test_ranks_are_a_permutation_per_explanation(artifacts):
    grouped = artifacts.attributions.groupby(["explainer", "instance_id", "num_samples"])
    for _, group in grouped:
        assert sorted(group["rank"]) == list(range(8))


def test_config_row_records_the_probability_space_check(artifacts):
    row = artifacts.config_row
    assert row["shap_backend"] in {"tree", "tree_prob", "permutation"}
    assert row["additivity_error"] < 5e-3
    assert row["space"] == "probability"


def test_calibration_and_evaluation_ids_are_disjoint(artifacts):
    assert not (set(artifacts.calib_ids) & set(artifacts.eval_ids))
    assert (set(artifacts.calib_ids) | set(artifacts.eval_ids)
            == set(artifacts.attributions.instance_id.unique()))


def test_rerun_records_tree_shap_as_exactly_zero_and_lime_as_positive(artifacts):
    rerun = artifacts.rerun
    shap_cv = rerun.loc[rerun.explainer == "shap", "rerun_cv"]
    lime_cv = rerun.loc[rerun.explainer == "lime", "rerun_cv"]
    assert (shap_cv == 0.0).all()          # tree backend, deterministic a priori
    assert (lime_cv > 0.0).any()


def test_robustness_frame_covers_both_base_explainers(artifacts):
    assert set(artifacts.robust.explainer) == {"shap", "lime"}
    assert (artifacts.robust.lipschitz >= 0).all()


def test_retrain_runs_only_for_ensembles_per_protocol_r3():
    ensemble = explain_cell("pima", "rf", params={}, profile=QUICK)
    other = explain_cell("pima", "logreg", params={}, profile=QUICK)
    assert "rf" in ENSEMBLE_MODELS and not ensemble.retrain.empty
    assert "logreg" not in ENSEMBLE_MODELS and other.retrain.empty


def test_transformed_ranking_expands_a_feature_into_all_its_onehot_columns():
    # Fidelity perturbs the model's real inputs, which are transformed columns,
    # while rankings live in original-feature space. Deleting a categorical feature
    # must delete every column belonging to it.
    from xaibench.explain.shap_backend import explain_shap
    from xaibench.models.cells import build_cell
    cell = build_cell("heart", "rf", params={}, profile=QUICK)
    exp = explain_shap(cell, cell.instance_ids[:1])[0]
    order = transformed_ranking(cell, exp)
    assert order.size == len(cell.groups)
    assert sorted(order.tolist()) == list(range(len(cell.groups)))
    # The first original feature's columns must all come before the second's.
    first = cell.feature_order[int(exp.ranking[0])]
    n_first = cell.groups.count(first)
    assert {cell.groups[i] for i in order[:n_first]} == {first}


@pytest.mark.slow
def test_main_writes_one_parquet_per_cell_and_resumes():
    main(force=True)
    p = paths()
    files = sorted(p.attributions.glob("*_attr.parquet"))
    assert len(files) == 18
    stamp = files[0].stat().st_mtime
    main(force=False)
    assert files[0].stat().st_mtime == stamp, "resume did not skip a completed cell"
    assert (p.results / "explainer_config.csv").exists()
    assert (p.results / "instance_budget.csv").exists()
