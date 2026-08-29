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
