import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from xaibench.config import QUICK, paths
from xaibench.viz.figures import FIGURES, build_all
from xaibench.viz.theme import PALETTE, apply_theme


def test_manifest_covers_every_planned_figure():
    assert set(FIGURES) == {f"F{i}" for i in range(1, 15)}


def test_the_two_series_the_paper_contrasts_are_separated_in_luminance():
    # SHAP versus LIME is the comparison every results figure turns on, so that one
    # pair must survive a black-and-white print on colour alone.
    from xaibench.viz.theme import relative_luminance
    gap = abs(relative_luminance(PALETTE["shap"]) - relative_luminance(PALETTE["lime"]))
    assert gap > 0.05, f"SHAP/LIME luminance gap is only {gap:.3f}"


def test_every_series_carries_redundant_non_colour_encoding():
    # Six series cannot be separated by luminance alone inside a validated
    # colourblind-safe palette, so line style and marker carry the difference too.
    from xaibench.viz.theme import LINESTYLES, MARKERS
    assert set(LINESTYLES) == set(PALETTE)
    assert set(MARKERS) == set(PALETTE)
    combinations = {(str(LINESTYLES[k]), MARKERS[k]) for k in PALETTE}
    assert len(combinations) == len(PALETTE), "two series share style and marker"


def test_style_helper_returns_all_three_channels():
    from xaibench.viz.theme import style
    for explainer in PALETTE:
        spec = style(explainer)
        assert set(spec) == {"color", "linestyle", "marker"}


def test_every_figure_writes_both_formats():
    apply_theme()
    names = build_all(QUICK)
    folder = paths().figures
    for name in names:
        assert (folder / f"{name}.pdf").exists(), name
        assert (folder / f"{name}.png").exists(), name
        assert (folder / f"{name}.pdf").stat().st_size > 1000, name


def test_quick_profile_watermarks_so_a_draft_cannot_be_mistaken_for_results():
    apply_theme()
    fig = FIGURES["F3"](QUICK)
    from xaibench.viz.theme import watermark
    watermark(fig, QUICK)
    assert any("QUICK" in text.get_text() for text in fig.texts)


@pytest.mark.parametrize("name", ["F3", "F4", "F6", "F10", "F11", "F13"])
def test_axes_carry_labels_because_captions_must_stand_alone(name):
    apply_theme()
    fig = FIGURES[name](QUICK)
    labelled = [ax for ax in fig.get_axes()
                if ax.has_data() and (ax.get_xlabel() or ax.get_ylabel())]
    assert labelled, f"{name}: no axis carries a label"


def test_shap_determinism_annotation_covers_only_the_tree_backends():
    """The figure claiming SHAP is deterministic must not quote a stochastic backend.

    Taking the max over all six models mixes in the permutation backend, whose
    coefficient of variation reached 139, producing a label that read
    "tree backends: exactly 0, max observed 1e+02" -- self-contradictory and wrong.
    """
    import pandas as pd

    from xaibench.config import paths
    path = paths().results / "stability_rerun.parquet"
    if not path.exists():
        pytest.skip("stage2 output not present")
    rerun = pd.read_parquet(path)
    shap_rows = rerun.query("explainer == 'shap'")
    tree = shap_rows[shap_rows.model.isin(["dtree", "rf", "xgb", "lgbm"])]
    agnostic = shap_rows[~shap_rows.model.isin(["dtree", "rf", "xgb", "lgbm"])]
    assert tree.rerun_cv.max() == 0.0, "tree SHAP is not deterministic in this run"
    if len(agnostic):
        assert agnostic.rerun_cv.max() > 0.0, (
            "the model-agnostic backend should be stochastic; if it is not, the "
            "figure's two-backend framing needs revisiting")


@pytest.mark.parametrize("name", ["F2", "F3", "F7", "F9", "F10", "F11", "F12"])
def test_legends_do_not_sit_on_top_of_axis_labels(name):
    """Guards the failure this test was written for.

    constrained_layout does not reserve space for a legend placed with a negative
    bbox_to_anchor, so several legends printed straight over the x-axis label. The
    supported fix is matplotlib's "outside ..." locations, which the layout engine
    accounts for. This asserts no legend box overlaps an axis label box.
    """
    apply_theme()
    fig = FIGURES[name](QUICK)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legends = [fig.legends, *[[ax.get_legend()] for ax in fig.get_axes()]]
    boxes = [lg.get_window_extent(renderer)
             for group in legends for lg in group if lg is not None]
    labels = []
    for ax in fig.get_axes():
        for artist in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if artist.get_text():
                labels.append(artist.get_window_extent(renderer))
    for i, legend in enumerate(boxes):
        for label in labels:
            overlap = (min(legend.x1, label.x1) - max(legend.x0, label.x0)) * \
                      (min(legend.y1, label.y1) - max(legend.y0, label.y0))
            assert not (legend.overlaps(label) and overlap > 40), (
                f"{name}: legend {i} overlaps an axis label")


@pytest.mark.parametrize("name", [f"F{i}" for i in [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]])
def test_no_axis_label_overflows_the_figure(name):
    """F3's rotated y-label was taller than its panel, so it ran off the canvas and the
    legend was drawn on top of it. A label that does not fit belongs in the caption."""
    apply_theme()
    fig = FIGURES[name](QUICK)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    for ax in fig.get_axes():
        for kind, artist in (("x", ax.xaxis.label), ("y", ax.yaxis.label),
                             ("title", ax.title)):
            if not artist.get_text():
                continue
            box = artist.get_window_extent(renderer)
            assert box.y0 >= -2 and box.y1 <= canvas.y1 + 2, (
                f"{name}: {kind} label overflows vertically "
                f"({artist.get_text()[:40]!r})")
