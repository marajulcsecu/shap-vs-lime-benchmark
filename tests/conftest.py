import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XAIB_PROFILE", "QUICK")


import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_outputs():
    """Redirect every generated artefact into a temp tree for the whole test session.

    The stage tests call main(force=True). Without this, running pytest would
    overwrite the FULL-profile results the paper is built from -- two hours of compute
    destroyed by a test run. Source data and the committed protocol still resolve to
    the real repository, since those are inputs and are never written.
    """
    root = Path(__file__).resolve().parents[1]
    scratch = Path(tempfile.mkdtemp(prefix="xaibench-tests-"))
    (scratch / "data").mkdir(parents=True, exist_ok=True)
    os.environ["XAIB_OUTPUT_ROOT"] = str(scratch)
    # Stage 3/4 read what a prior FULL or QUICK run produced, so seed the scratch tree
    # with a copy when one exists; otherwise the stage tests build it themselves.
    for name in ("results", "figures", "tables"):
        source = root / name
        if source.exists():
            shutil.copytree(source, scratch / name, dirs_exist_ok=True)
    yield scratch
    os.environ.pop("XAIB_OUTPUT_ROOT", None)
    shutil.rmtree(scratch, ignore_errors=True)
