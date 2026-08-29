import pandas as pd
import pytest

from xaibench.config import ALPHA, paths
from xaibench.pipeline.stage4_stats import main

EXPECTED = {"H1", "H1b", "H1c", "H2a", "H2b", "H2c", "H3a", "H3b", "H4", "H5", "H6"}


@pytest.fixture(scope="module")
def stats_frame():
    main(force=True)
    return pd.read_csv(paths().results / "stats_hypotheses.csv")


def test_all_eleven_pre_registered_hypotheses_are_tested(stats_frame):
    assert set(stats_frame.hypothesis) == EXPECTED


def test_family_sizes_match_the_protocol(stats_frame):
    counts = stats_frame.hypothesis.value_counts().to_dict()
    assert counts["H1"] == 18
    assert counts["H2a"] == 18            # LIME only; SHAP determinism is a priori
    assert counts["H2b"] == 18
    assert counts["H2c"] == 9             # ensembles only, protocol R3
    assert counts["H3a"] == 18
    assert counts["H3b"] == 36            # each of SHAP and LIME against the control
    assert counts["H4"] == 18
    assert counts["H6"] == 18
    assert counts["H1b"] == 1 and counts["H1c"] == 1 and counts["H5"] == 1


def test_holm_adjustment_is_never_more_lenient_than_the_raw_p(stats_frame):
    assert (stats_frame.p_adj >= stats_frame.p_value - 1e-12).all()


def test_holm_is_applied_within_family_not_across_all_tests(stats_frame):
    # A family of 18 must not be corrected as if it were a family of 156.
    family = stats_frame.query("hypothesis == 'H1' and not degenerate")
    if len(family) > 1:
        ratio = (family.p_adj / family.p_value.clip(lower=1e-300)).max()
        assert ratio <= len(family) + 1e-9


def test_degenerate_rows_are_excluded_from_the_correction_divisor(stats_frame):
    # Correcting a test that was never performed only inflates the divisor and
    # throws away power. Degenerate rows carry p_adj = 1 by construction.
    degenerate = stats_frame.query("degenerate")
    if len(degenerate):
        assert (degenerate.p_adj == 1.0).all()


def test_decision_follows_the_adjusted_p_value(stats_frame):
    assert (stats_frame.query("decision == 'reject H0'").p_adj < ALPHA).all()
    assert (stats_frame.query("decision == 'fail to reject H0'").p_adj >= ALPHA).all()


def test_every_non_degenerate_test_reports_an_effect_size(stats_frame):
    live = stats_frame.query("not degenerate")
    assert live.effect_size.notna().all()
    assert live.effect_name.notna().all()


def test_shap_determinism_is_recorded_as_a_measurement_not_a_hypothesis():
    # The protocol declares tree SHAP deterministic a priori, so it is reported in
    # the stability metrics rather than tested; H2a covers LIME, where the question
    # is real. This guards against quietly resurrecting the vacuous test.
    stability = pd.read_csv(paths().results / "metrics_stability.csv")
    tree = stability.query("explainer == 'shap' and model in ['rf','xgb','lgbm','dtree']")
    assert len(tree) > 0
    assert (tree.rerun_cv.fillna(0) == 0).all()


def test_estimates_carry_confidence_intervals_that_bracket_the_point():
    estimates = pd.read_csv(paths().results / "stats_estimates.csv")
    usable = estimates.dropna(subset=["ci_low", "ci_high", "point"])
    assert len(usable) > 0
    assert (usable.ci_low <= usable.point + 1e-9).all()
    assert (usable.point <= usable.ci_high + 1e-9).all()


def test_h6_is_tested_on_evaluation_instances_only(stats_frame):
    # The fidelity weight is fitted on the calibration split, so testing H6 on those
    # instances would let the fused rule win by construction.
    budget = pd.read_csv(paths().results / "instance_budget.csv")
    h6 = stats_frame.query("hypothesis == 'H6'")
    merged = h6.merge(budget, on=["dataset", "model"])
    assert (merged.n <= merged.n_evaluation).all()
