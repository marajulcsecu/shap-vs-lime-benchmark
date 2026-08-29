"""Guard the guard: the test session must not be able to write into the repository."""
from pathlib import Path

from xaibench.config import paths

REPO = Path(__file__).resolve().parents[1]


def test_generated_paths_point_outside_the_repository():
    p = paths()
    for folder in (p.results, p.figures, p.tables, p.attributions):
        assert REPO not in folder.parents and folder != REPO, (
            f"{folder} is inside the repository; a test run could overwrite "
            "reported results"
        )


def test_source_data_and_protocol_still_resolve_to_the_repository():
    # These are inputs, never written, and must not be redirected.
    p = paths()
    assert p.data_raw == REPO / "data" / "raw"
    assert p.docs == REPO / "docs"
    assert (p.data_raw / "checksums.json").exists()
