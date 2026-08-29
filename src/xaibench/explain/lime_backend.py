"""LIME attributions, returned through the same contract as SHAP."""
from __future__ import annotations

import time

import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from xaibench.config import SEED
from xaibench.explain.base import Explanation, config_hash
from xaibench.models.pipelines import aggregate_attributions

# NORMATIVE primary (protocol §2.3); False is the sensitivity variant.
DISCRETIZE_CONTINUOUS = True


def build_lime_explainer(cell, seed: int, discretize: bool = DISCRETIZE_CONTINUOUS):
    """One explainer per cell, built on the transformed training rows.

    categorical_features is passed so LIME does not discretise one-hot columns into
    bins, which would produce neighbourhood samples the model has never seen.
    """
    return LimeTabularExplainer(
        training_data=np.asarray(cell.X_train_t, dtype=float),
        feature_names=[f"c{i}" for i in range(cell.X_train_t.shape[1])],
        class_names=["negative", "positive"],
        categorical_features=list(cell.categorical_idx),
        discretize_continuous=bool(discretize),
        mode="classification",
        random_state=int(seed),
    )


def explain_lime(cell, instance_ids=None, num_samples: int = 5000,
                 seed: int = SEED,
                 discretize: bool = DISCRETIZE_CONTINUOUS) -> list[Explanation]:
    ids = np.asarray(cell.instance_ids if instance_ids is None else instance_ids, dtype=int)
    n_out = int(cell.X_train_t.shape[1])
    explainer = build_lime_explainer(cell, seed, discretize)
    cfg = config_hash(explainer="lime", num_samples=int(num_samples),
                      discretize=bool(discretize), space="probability")

    X_test = np.asarray(cell.X_test_t, dtype=float)
    out: list[Explanation] = []
    for inst in ids:
        x = X_test[int(inst)]
        start = time.perf_counter()
        exp = explainer.explain_instance(
            data_row=x, predict_fn=cell.classifier.predict_proba,
            num_features=n_out, num_samples=int(num_samples), labels=(1,))
        elapsed = time.perf_counter() - start

        raw = np.zeros(n_out, dtype=float)
        for column, weight in exp.as_map()[1]:
            raw[int(column)] = float(weight)
        aggregated = aggregate_attributions(raw[None, :], cell.groups,
                                           cell.feature_order)[0]

        out.append(Explanation(
            dataset=cell.dataset, model=cell.model, explainer="lime",
            instance_id=int(inst), attributions=aggregated,
            feature_names=tuple(cell.feature_order), runtime_s=float(elapsed),
            seed=int(seed), config_hash=cfg))
    return out
