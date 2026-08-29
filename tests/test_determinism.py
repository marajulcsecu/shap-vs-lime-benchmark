"""Two runs from the same seed must agree on every non-timing column.

This is the test that keeps the reproducibility claim honest rather than aspirational.
It runs the QUICK profile twice inside the isolated output tree and compares digests of
every result table with wall-clock columns removed.
"""
from __future__ import annotations

import hashlib
import shutil

import pandas as pd
import pytest

from xaibench.config import paths

# Wall-clock varies by machine load and is legitimately non-deterministic.
VOLATILE = {"runtime_s", "median_seconds", "iqr_seconds", "q1_seconds", "q3_seconds",
            "fidelity_per_second", "duration_s"}


def _digest(path) -> str:
    frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    frame = frame.drop(columns=[c for c in frame.columns if c in VOLATILE],
                       errors="ignore")
    if frame.empty:
        return "empty"
    frame = frame.sort_index(axis=1)
    frame = frame.sort_values(list(frame.columns), kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


@pytest.mark.slow
def test_two_quick_runs_from_the_same_seed_produce_identical_results():
    from xaibench.pipeline import (
        stage1_performance, stage2_explain, stage3_metrics, stage4_stats,
    )

    def run_once():
        stage1_performance.main(force=True)
        stage2_explain.main(force=True)
        stage3_metrics.main(force=True)
        stage4_stats.main(force=True)
        p = paths()
        return {path.name: _digest(path)
                for path in sorted(p.results.glob("*.csv"))
                if path.name != "MANIFEST.json"}

    first = run_once()
    # Clear the checkpoints so the second pass genuinely recomputes rather than resumes.
    shutil.rmtree(paths().attributions, ignore_errors=True)
    second = run_once()

    assert set(first) == set(second)
    differing = [name for name in first if first[name] != second[name]]
    assert not differing, f"non-deterministic outputs: {differing}"
