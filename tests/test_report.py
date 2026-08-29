import re

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


def test_headers_follow_the_same_ready_to_typeset_contract_as_cells():
    # Headers legitimately carry maths, so the writer must not escape them. Builders
    # call escape() on free text and leave intentional maths alone.
    tex = to_booktabs(pd.DataFrame({r"Spearman $\rho$": ["1"]}), caption="c", label="l")
    assert r"Spearman $\rho$" in tex
    assert r"\$" not in tex


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


def test_manifest_covers_every_planned_table():
    from xaibench.stats.report import TABLES
    assert set(TABLES) == {f"T{i}" for i in range(1, 13)}


def _generated_tables():
    from xaibench.config import paths
    # findings.tex holds \newcommand macros, not a table environment.
    return [p for p in sorted(paths().tables.glob("T*.tex")) if p.stem != "findings"]


def test_no_generated_table_contains_a_leaked_missing_value():
    # A "nan" in a submitted table is the most embarrassing possible defect. The check
    # is for a leaked *missing value*, so it matches how Python renders one -- word
    # boundaries and exact case -- not the English word "none", which T11 uses
    # legitimately in "none reported".
    files = _generated_tables()
    assert files, "no tables generated yet; run stage5 first"
    for path in files:
        text = path.read_text()
        assert not re.search(r"\bnan\b", text, re.IGNORECASE), f"{path.name}: nan"
        assert not re.search(r"\bNone\b", text), f"{path.name}: None"
        assert "inf" not in text.lower().replace("information", ""), f"{path.name}: inf"


def test_every_generated_table_is_balanced_latex():
    for path in _generated_tables():
        text = path.read_text()
        assert text.count(r"\begin{table}") == text.count(r"\end{table}") == 1, path.name
        assert text.count(r"\begin{tabular}") == text.count(r"\end{tabular}") == 1, path.name
        assert r"\label{" in text, path.name
        assert path.stat().st_size > 60, path.name


@pytest.mark.parametrize("name", [f"T{i}" for i in range(1, 13)])
def test_every_generated_table_compiles_standalone(name, tmp_path):
    """The only reliable check that a builder's escaping is right.

    Headers legitimately contain maths ("Spearman $\\rho$", "$p_{\\text{adj}}$") while
    cells legitimately contain escaped text (CKD feature names carry underscores).
    No escaping rule distinguishes them, so the table is compiled instead of inspected.
    """
    import shutil
    import subprocess

    from xaibench.config import paths

    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex not available")
    source = paths().tables / f"{name}.tex"
    if not source.exists():
        pytest.skip("tables not generated yet; run stage5 first")

    document = tmp_path / "probe.tex"
    document.write_text(
        "\\documentclass{article}\n"
        "\\usepackage[T1]{fontenc}\\usepackage{booktabs}\\usepackage{amsmath,amssymb}\n"
        "\\usepackage{multirow}\\usepackage{array}\\usepackage{siunitx}\n"
        "\\begin{document}\n" + source.read_text() + "\n\\end{document}\n")
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", document.name],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"{name}.tex does not compile:\n" + result.stdout[-2500:])
