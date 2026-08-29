"""Model factory, parameter grids, and SHAP-backend assignment."""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from xaibench.config import SEED

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "dtree": "Decision Tree",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "stack": "Stacking",
}

# NORMATIVE (protocol §2.2). dtree/rf produce class proportions, so TreeExplainer's
# raw output is already probability space. xgb/lgbm produce log-odds margins and need
# model_output="probability" with an interventional background. logreg/stack have no
# tree structure, so they use the model-agnostic permutation sampler on predict_proba.
_BACKEND = {
    "logreg": "permutation", "dtree": "tree", "rf": "tree",
    "xgb": "tree_prob", "lgbm": "tree_prob", "stack": "permutation",
}


def shap_backend_kind(model_id: str) -> str:
    if model_id not in _BACKEND:
        raise ValueError(f"unknown model {model_id!r}")
    return _BACKEND[model_id]


def _base_learners(seed: int):
    return [
        ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                      class_weight="balanced", random_state=seed,
                                      n_jobs=1)),
        ("xgb", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                              subsample=0.9, colsample_bytree=0.9,
                              tree_method="hist", eval_metric="logloss",
                              random_state=seed, n_jobs=1)),
        ("lgbm", LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                min_child_samples=5, class_weight="balanced",
                                random_state=seed, n_jobs=1, verbose=-1)),
    ]


def build_model(model_id: str, seed: int = SEED):
    if model_id == "logreg":
        return LogisticRegression(max_iter=5000, class_weight="balanced",
                                  random_state=seed)
    if model_id == "dtree":
        return DecisionTreeClassifier(max_depth=5, min_samples_leaf=5,
                                      class_weight="balanced", random_state=seed)
    if model_id == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      class_weight="balanced", random_state=seed,
                                      n_jobs=1)
    if model_id == "xgb":
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1,
                             subsample=0.9, colsample_bytree=0.9,
                             tree_method="hist", eval_metric="logloss",
                             random_state=seed, n_jobs=1)
    if model_id == "lgbm":
        # num_leaves and min_child_samples are deliberately small: these datasets
        # have 303-768 rows and LightGBM's defaults overfit and warn at that scale.
        return LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                              min_child_samples=5, class_weight="balanced",
                              random_state=seed, n_jobs=1, verbose=-1)
    if model_id == "stack":
        return StackingClassifier(
            estimators=_base_learners(seed),
            final_estimator=LogisticRegression(max_iter=5000,
                                               class_weight="balanced",
                                               random_state=seed),
            cv=3, stack_method="predict_proba", n_jobs=1, passthrough=False,
        )
    raise ValueError(f"unknown model {model_id!r}")


# Grids are capped at 12 combinations per model: with 18 cells x 5 outer x 3 inner
# folds, anything larger stops fitting inside the two-hour budget.
_GRIDS = {
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0]},
    "dtree":  {"max_depth": [3, 5, 8], "min_samples_leaf": [2, 5, 10]},
    "rf":     {"max_depth": [None, 8], "min_samples_leaf": [1, 2, 5]},
    "xgb":    {"max_depth": [2, 3, 5], "learning_rate": [0.05, 0.1]},
    "lgbm":   {"num_leaves": [7, 15, 31], "learning_rate": [0.05, 0.1]},
    "stack":  {"final_estimator__C": [0.1, 1.0, 10.0]},
}


def param_grid(model_id: str) -> dict[str, list]:
    if model_id not in _GRIDS:
        raise ValueError(f"unknown model {model_id!r}")
    return {k: list(v) for k, v in _GRIDS[model_id].items()}
