import pandas as pd
import pytest

from xaibench.stats.report import escape, fmt, fmt_ci, fmt_p, to_booktabs


def test_booktabs_output_is_compilable_latex():
    df = pd.DataFrame({"model": ["RF"], "value": ["0.500"]})
    tex = to_booktabs(df, caption="Cap", label="tbl:x")
    for token in [r"\begin{table}", r"\toprule", r"\midrule", r"\bottomrule",
                  r"\end{table}", r"\label{tbl:x}"]:
        assert token in tex
    assert tex.count(r"\begin{tabular}") == tex.count(r"\end{tabular}") == 1


def test_latex_special_characters_are_escaped():
    escaped = escape("a_b & c% 50$")
    assert r"a\_b" in escaped
    assert r"\&" in escaped
    assert r"\%" in escaped
    assert r"\$" in escaped


def test_preformatted_cells_pass_through_so_p_value_maths_survives():
    # to_booktabs takes cells LaTeX-ready. A marker type cannot survive a DataFrame
    # round trip (pandas coerces a str subclass back to str), so escaping by default
    # would turn fmt_p's "$<$0.001" into a literal dollar sign.
    df = pd.DataFrame({"p": [fmt_p(1e-9)]})
    assert r"$<$0.001" in to_booktabs(df, caption="c", label="l")


def test_builders_escape_text_columns_explicitly():
    # CKD and Heart feature names carry underscores; the builder escapes, not the writer.
    df = pd.DataFrame({"feature": [escape("some_feature")], "v": [fmt(0.5)]})
    assert r"some\_feature" in to_booktabs(df, caption="c", label="l")


def test_headers_are_always_escaped_even_when_cells_are_not():
    assert r"a\_b" in to_booktabs(pd.DataFrame({"a_b": ["1"]}), caption="c", label="l")


def test_p_values_are_rendered_with_a_floor_not_as_zero():
    assert fmt_p(1e-7) == r"$<$0.001"
    assert fmt_p(0.0234) == "0.023"
    assert fmt_p(float("nan")) == "--"


def test_missing_values_render_as_a_dash_not_the_word_nan():
    assert fmt(float("nan")) == "--"
    assert fmt(None) == "--"
    assert fmt_ci(float("nan"), 1.0, 2.0) == "--"
    assert fmt(0.12345, digits=3) == "0.123"


def test_tiny_magnitudes_do_not_collapse_to_zero():
    # A rerun CV of 8e-17 must print as neither "0.000" (implying exact determinism)
    # nor "nan". Scientific notation keeps it honest.
    assert "e-" in fmt(8.3e-17)
    assert fmt(0.0) == "0.000"


def test_ci_format_is_compact_and_readable():
    assert fmt_ci(0.5, 0.4, 0.6) == "0.500 [0.400, 0.600]"
