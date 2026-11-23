"""Document build guards.

These were in the implementation plan but never written during execution; the document
builds were only checked by verify.sh, which is not run by CI or by `make test`. The
6-page conference limit in particular is a hard external requirement and belongs in the
suite.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ("paper", "report", "slides")

pytestmark = pytest.mark.skipif(shutil.which("latexmk") is None,
                                reason="latexmk not available")


def _pages(document: str) -> int:
    from pypdf import PdfReader
    pdf = ROOT / document / "main.pdf"
    if not pdf.exists():
        pytest.skip(f"{document}/main.pdf not built; run `make docs`")
    return len(PdfReader(str(pdf)).pages)


@pytest.mark.slow
@pytest.mark.parametrize("document", DOCUMENTS)
def test_document_compiles_from_a_clean_state(document):
    result = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=ROOT / document, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-3000:]


def test_paper_respects_the_six_page_conference_limit():
    """A hard limit set by the supervisor, including references -- not a soft target."""
    pages = _pages("paper")
    assert pages <= 6, f"paper is {pages} pages; the IEEE conference limit is 6"
    assert pages >= 5, f"paper is only {pages} pages; content is probably missing"


def test_report_is_the_full_length_treatment():
    assert _pages("report") >= 28


def test_slide_count_fits_a_fifteen_minute_talk():
    assert 18 <= _pages("slides") <= 40


def test_the_report_does_not_depend_on_the_papers_prose():
    """The report used to \\input the paper's sections, so cutting the paper to 6 pages
    would have silently gutted the 48-page report. They are independent sources now."""
    directive = re.compile(r"^[^%]*\\input\{\.\./paper/sections/", re.MULTILINE)
    for path in sorted((ROOT / "report").rglob("*.tex")):
        found = directive.search(path.read_text())
        assert not found, f"{path.relative_to(ROOT)} still inputs the paper's prose"


@pytest.mark.parametrize("document", ("paper", "report"))
def test_no_undefined_references_or_overfull_lines(document):
    log = ROOT / document / "main.log"
    if not log.exists():
        pytest.skip(f"{document}/main.log absent; run `make docs`")
    text = log.read_text(errors="ignore")
    assert not re.search(r"Reference .* undefined|Citation .* undefined", text)
    assert "Float(s) lost" not in text
    assert text.count("Overfull \\hbox") == 0, "a line runs past the text block"


@pytest.mark.parametrize("document", ("paper", "report"))
def test_no_placeholder_markers_in_the_rendered_pdf(document):
    if shutil.which("pdftotext") is None:
        pytest.skip("pdftotext not available")
    pdf = ROOT / document / "main.pdf"
    if not pdf.exists():
        pytest.skip(f"{document}/main.pdf not built")
    rendered = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                              text=True, timeout=120).stdout
    assert not re.search(r"Table \?\?|Fig\. \?\?|Section \?\?|Chapter \?\?|\[\?\]",
                         rendered)
    assert "PROFILE=QUICK" not in rendered
    assert "STUB" not in rendered


def test_every_number_enters_through_a_generated_file():
    """The project's load-bearing constraint: no result is typed by hand.

    Scans section bodies for bare multi-decimal numbers. A number is allowed inside an
    \\input'd table, a figure, or a whitelisted context (thresholds and years already
    fixed in the protocol).
    """
    allowed = re.compile(
        r"(\\cite|\\ref|\\label|\\input|\\includegraphics|^%|0\.50|0\.05|1e-|e-0"
        r"|5\\times10|9\.9\\times|7\\times|1\.1\\times|0\.1\\sigma|\\times10"
        r"|2024|2019|34\.9|62\.5|80/20)")
    offenders = []
    for folder in ("paper/sections", "report/sections", "report/appendices"):
        for path in sorted((ROOT / folder).glob("*.tex")):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("%") or allowed.search(stripped):
                    continue
                if re.search(r"(?<![\\{A-Za-z0-9.])\d+\.\d{3,}", stripped):
                    offenders.append(f"{folder}/{path.name}:{number}: {stripped[:80]}")
    assert not offenders, "hand-typed results:\n" + "\n".join(offenders)


def test_the_paper_inputs_generated_tables_rather_than_inlining_them():
    body = "\n".join(p.read_text() for p in (ROOT / "paper" / "sections").glob("*.tex"))
    assert "tables/T1" in body and "tables/T10" in body
    assert "\\begin{tabular}" not in body, "a table is inlined instead of generated"


def test_bibliography_is_sized_for_a_six_page_paper():
    """Post-review revision (ICCIT 2026, R1.5) targets a concise 20-22 reference list
    covering recent disagreement, LIME stability, conditional/ROAR fidelity, and clinical XAI."""
    bbl = ROOT / "paper" / "main.bbl"
    if not bbl.exists():
        pytest.skip("main.bbl absent; run `make paper`")
    cited = bbl.read_text().count("\\bibitem")
    assert 18 <= cited <= 24, f"{cited} references cited; expected 18-24 for a 6-page IEEE paper"
