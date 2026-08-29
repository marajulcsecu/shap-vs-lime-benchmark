import numpy as np
import pytest
from sklearn.base import is_classifier

from xaibench.config import MODELS
from xaibench.models.zoo import MODEL_LABELS, build_model, param_grid, shap_backend_kind


@pytest.mark.parametrize("model_id", MODELS)
def test_every_model_is_a_classifier_with_predict_proba(model_id):
    model = build_model(model_id)
    assert is_classifier(model)
    assert hasattr(model, "predict_proba")


@pytest.mark.parametrize("model_id", MODELS)
def test_every_model_fits_and_predicts_probabilities_in_range(model_id):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    proba = build_model(model_id).fit(X, y).predict_proba(X)
    assert proba.shape == (120, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("model_id", MODELS)
def test_grids_are_small_enough_for_the_compute_budget(model_id):
    grid = param_grid(model_id)
    combos = int(np.prod([len(v) for v in grid.values()])) if grid else 1
    assert 1 <= combos <= 12, f"{model_id} grid has {combos} combinations, budget is 12"


@pytest.mark.parametrize("model_id", MODELS)
def test_grid_keys_are_real_estimator_parameters(model_id):
    valid = build_model(model_id).get_params().keys()
    for key in param_grid(model_id):
        assert key in valid, f"{model_id}: {key} is not a parameter of the estimator"


def test_backend_assignment_matches_the_protocol():
    # dtree/rf output class proportions, so TreeExplainer's raw output is already
    # probability space. xgb/lgbm output log-odds margins and need model_output.
    # logreg/stack have no tree backend, so they use the model-agnostic sampler.
    assert shap_backend_kind("dtree") == "tree"
    assert shap_backend_kind("rf") == "tree"
    assert shap_backend_kind("xgb") == "tree_prob"
    assert shap_backend_kind("lgbm") == "tree_prob"
    assert shap_backend_kind("logreg") == "permutation"
    assert shap_backend_kind("stack") == "permutation"


@pytest.mark.parametrize("model_id", MODELS)
def test_models_are_reproducible_from_the_seed(model_id):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 5))
    y = (X[:, 0] > 0).astype(int)
    a = build_model(model_id, seed=42).fit(X, y).predict_proba(X)
    b = build_model(model_id, seed=42).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b)


def test_labels_exist_for_every_model():
    assert set(MODEL_LABELS) == set(MODELS)
