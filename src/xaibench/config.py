"""Central configuration. Values marked NORMATIVE are fixed by docs/PROTOCOL.md."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SEED = 42                     # NORMATIVE
ALPHA = 0.05                  # NORMATIVE
RRF_K = 60                    # NORMATIVE
AOPC_MAX_K = 10               # NORMATIVE
CALIB_FRACTION = 0.30         # NORMATIVE
TEST_SIZE = 0.20              # NORMATIVE
LIPSCHITZ_SIGMA = 0.10        # NORMATIVE: neighbour sd, in units of train sd
RANDOM_CONTROL_DRAWS = 20     # NORMATIVE
BASELINE_MARGINAL_DRAWS = 5   # NORMATIVE (see amendment A4)
JACCARD_KS = (3, 5)           # NORMATIVE
RBO_P = 0.90                  # NORMATIVE
COMP_SUFF_K = 5               # NORMATIVE: k for comprehensiveness / sufficiency

DATASETS = ("pima", "heart", "ckd")
MODELS = ("logreg", "dtree", "rf", "xgb", "lgbm", "stack")
ENSEMBLE_MODELS = ("rf", "xgb", "lgbm")     # NORMATIVE: RQ2c scope (protocol R3)
SIMPLE_MODELS = ("logreg", "dtree")         # for H1b's grouping
CELLS = tuple((d, m) for d in DATASETS for m in MODELS)

BASE_EXPLAINERS = ("shap", "lime")
CONSENSUS_EXPLAINERS = ("rankavg", "rrf", "fidwt")
CONTROL_EXPLAINERS = ("random",)
ALL_EXPLAINERS = BASE_EXPLAINERS + CONSENSUS_EXPLAINERS + CONTROL_EXPLAINERS


@dataclass(frozen=True)
class Profile:
    name: str
    n_explain: int
    lime_num_samples: tuple[int, ...]   # first entry is primary
    rerun_repeats: int
    robust_neighbours: int
    robust_instances: int
    rerun_instances: int
    bootstrap_refits: int
    bootstrap_instances: int
    ci_resamples: int
    shap_background: int
    marginal_draws: int

    @property
    def is_reportable(self) -> bool:
        return self.name == "FULL"


QUICK = Profile(
    name="QUICK", n_explain=12, lime_num_samples=(500,), rerun_repeats=3,
    robust_neighbours=2, robust_instances=4, rerun_instances=4,
    bootstrap_refits=3, bootstrap_instances=4, ci_resamples=200,
    shap_background=20, marginal_draws=2,
)

FULL = Profile(                                     # all values NORMATIVE
    name="FULL", n_explain=150, lime_num_samples=(5000, 1000), rerun_repeats=10,
    robust_neighbours=5, robust_instances=25, rerun_instances=30,
    bootstrap_refits=10, bootstrap_instances=25, ci_resamples=2000,
    shap_background=50, marginal_draws=5,
)

_PROFILES = {"QUICK": QUICK, "FULL": FULL}


def get_profile(name: str | None = None) -> Profile:
    key = (name or os.environ.get("XAIB_PROFILE") or "QUICK").upper()
    if key not in _PROFILES:
        raise ValueError(f"unknown profile {key!r}; expected one of {sorted(_PROFILES)}")
    return _PROFILES[key]


@dataclass(frozen=True)
class Paths:
    root: Path
    data_raw: Path
    data_processed: Path
    results: Path
    attributions: Path
    figures: Path
    tables: Path
    docs: Path
    logs: Path

    def mkdirs(self) -> None:
        for folder in (self.data_raw, self.data_processed, self.results,
                       self.attributions, self.figures, self.tables, self.logs):
            folder.mkdir(parents=True, exist_ok=True)


def paths() -> Paths:
    """Project paths, with an env override so tests cannot clobber a reported run.

    The stage tests call main(force=True), which regenerates results/. Without
    XAIB_OUTPUT_ROOT a `pytest` invocation would overwrite the FULL-profile artefacts
    the documents are built from -- a footgun that costs two hours to undo.
    """
    root = Path(__file__).resolve().parents[2]
    out = Path(os.environ["XAIB_OUTPUT_ROOT"]) if os.environ.get("XAIB_OUTPUT_ROOT") else root
    return Paths(
        root=root,
        data_raw=root / "data" / "raw",          # always the vendored source data
        data_processed=out / "data" / "processed",
        results=out / "results",
        attributions=out / "results" / "attributions",
        figures=out / "figures",
        tables=out / "tables",
        docs=root / "docs",                      # always the committed protocol
        logs=out / "logs",
    )
