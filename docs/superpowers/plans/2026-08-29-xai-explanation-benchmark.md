# XAI Explanation Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible benchmark comparing SHAP and LIME across 6 classifiers × 3 clinical tabular datasets on agreement, stability, fidelity, cost, and plausibility, plus three rank-fusion rules, and generate an IEEE conference paper, a thesis report, and a Beamer deck in which every number is machine-generated.

**Architecture:** A five-stage pipeline (`performance → explain → metrics → statistics → outputs`) writes tidy CSV/Parquet into `results/`. Three LaTeX documents consume `figures/` and `tables/` only, via `\input`, so no value is ever transcribed by hand. Every explainer — SHAP, LIME, three fusion rules, and a random control — returns the same frozen `Explanation` dataclass, so metrics are written once and adding an explainer touches one file.

**Tech Stack:** Python 3.12, scikit-learn, XGBoost, LightGBM, `shap`, `lime`, statsmodels, pandas/pyarrow, matplotlib, pytest; TeX Live 2023 with IEEEtran and Beamer; `uv` for the virtualenv; GNU Make for orchestration.

**Spec:** `docs/superpowers/specs/2026-08-29-xai-explanation-benchmark-design.md`
**Frozen protocol (normative for every metric, hypothesis, and test):** `docs/PROTOCOL.md`
**Document outlines and figure/table manifest:** `docs/PAPER_PLAN.md`

## Global Constraints

- Python 3.12.3, virtualenv at `.venv/`, created with `uv venv`. Never install into system Python.
- `numpy<2` — pinned. numpy 2.x breaks the numba/llvmlite chain that `shap` depends on; a two-day schedule has no room for that debugging.
- Exact resolved versions captured to `requirements.lock.txt` via `uv pip freeze` after install. That file, not `pyproject.toml`, is the reproducibility artefact.
- Global seed `42`. Repeat *r* uses seed `1000 + r`. Bootstrap resample *b* uses seed `2000 + b`. Every seed used is recorded in `results/MANIFEST.json`.
- Two profiles, one code path: `QUICK` (≈5 min, validates the whole chain including LaTeX) and `FULL` (≈2 h, the reported results). Profile is selected by `XAIB_PROFILE` env var or `--profile`; it defaults to `QUICK` so an accidental run is never the expensive one.
- Every figure produced under `QUICK` carries a visible `PROFILE=QUICK — NOT FOR REPORTING` watermark.
- **No number is typed by hand into any `.tex` file.** Documents `\input` generated fragments from `tables/` and include PDFs from `figures/`. A hand-typed result is a defect, reviewable as such.
- All 18 cells are `(dataset, model)` pairs over `datasets = [pima, heart, ckd]` and `models = [logreg, dtree, rf, xgb, lgbm, stack]`.
- Attributions are always aggregated from one-hot columns back to **original-feature space** before any ranking, comparison, or metric.
- All explainers must produce attributions in **predicted-probability space**. Enforced by an additivity assertion, not by assumption (Task 14).
- Thread count pinned to 1 (`OMP_NUM_THREADS=1`, `n_jobs=1`) for any timed explainer call, so RQ4 is not confounded by parallelism.
- Non-parametric statistics only; Holm correction within each hypothesis family; every point estimate carries a 95 % BCa bootstrap CI; every test carries an effect size.
- Amendments to `docs/PROTOCOL.md` after its freeze commit go in its §10 amendment log with a reason. Never by silent edit.

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `pyproject.toml`, `requirements.lock.txt` | dependency declaration and exact pin |
| `Makefile` | `setup data test smoke all figures tables paper report slides clean` |
| `src/xaibench/config.py` | `Profile` dataclass, `QUICK`/`FULL`, all paths, seeds, budgets, cell enumeration |
| `src/xaibench/data/fetch.py` | download → SHA-256 verify → vendor to `data/raw/` |
| `src/xaibench/data/loaders.py` | `load_dataset(id) -> Dataset`; per-dataset semantics fixes |
| `src/xaibench/models/pipelines.py` | leakage-safe `ColumnTransformer`; `onehot_group_map`; `aggregate_attributions` |
| `src/xaibench/models/zoo.py` | model factory, parameter grids, SHAP-backend selection |
| `src/xaibench/explain/base.py` | frozen `Explanation` dataclass; `ranks_from_attributions` |
| `src/xaibench/explain/shap_backend.py` | per-model SHAP backend + background summarisation |
| `src/xaibench/explain/lime_backend.py` | `LimeTabularExplainer` wrapper, same contract |
| `src/xaibench/explain/consensus.py` | `rankavg`, `rrf`, `fidelity_weighted`, `random_control` |
| `src/xaibench/metrics/agreement.py` | `spearman`, `kendall_tau_b`, `jaccard_at_k`, `rbo` |
| `src/xaibench/metrics/fidelity.py` | `aopc_deletion`, `auc_insertion`, `comprehensiveness`, `sufficiency` |
| `src/xaibench/metrics/stability.py` | `rerun_cv`, `repeat_pair_rho`, `local_lipschitz`, `retrain_agreement` |
| `src/xaibench/metrics/plausibility.py` | reference-set overlap from `docs/reference_features.yaml` |
| `src/xaibench/stats/tests.py` | `wilcoxon_with_effect`, `holm`, `bca_ci`, `cliffs_delta` |
| `src/xaibench/stats/report.py` | tidy DataFrame → `booktabs` LaTeX fragment |
| `src/xaibench/viz/theme.py` | single matplotlib style; colourblind-safe; greyscale-legible |
| `src/xaibench/viz/figures.py` | one function per figure F1–F14, each saving `.pdf` + `.png` |
| `src/xaibench/pipeline/stage1_performance.py` … `stage5_outputs.py` | CLI stages |
| `tests/` | pytest; metric correctness against analytically known cases |
| `paper/`, `report/`, `slides/` | LaTeX sources; `sections/` shared between paper and report |
| `data/raw/`, `results/`, `figures/`, `tables/` | artefacts; `data/raw/checksums.json`, `results/MANIFEST.json` |

---

## Phase 0 — Foundation

### Task 1: Repository scaffold, environment, configuration profiles

**Files:**
- Create: `pyproject.toml`, `Makefile`, `.gitignore`, `src/xaibench/__init__.py`, `src/xaibench/config.py`
- Create: `tests/test_config.py`, `tests/conftest.py`
- Create: empty package dirs with `__init__.py`: `src/xaibench/{data,models,explain,metrics,stats,viz,pipeline}/`

**Interfaces:**
- Consumes: nothing.
- Produces: `Profile` dataclass with fields `name: str`, `n_explain: int`, `lime_num_samples: tuple[int, ...]`, `rerun_repeats: int`, `robust_neighbours: int`, `bootstrap_refits: int`, `ci_resamples: int`; module constants `QUICK: Profile`, `FULL: Profile`, `DATASETS: tuple[str,...]`, `MODELS: tuple[str,...]`, `CELLS: tuple[tuple[str,str],...]`, `SEED: int = 42`, `ALPHA: float = 0.05`, `RRF_K: int = 60`, `AOPC_MAX_K: int = 10`, `CALIB_FRACTION: float = 0.30`, `ENSEMBLE_MODELS: tuple[str,...] = ("rf","xgb","lgbm")`; functions `get_profile(name: str | None = None) -> Profile`, `paths() -> Paths` where `Paths` is a frozen dataclass of `root, data_raw, data_processed, results, figures, tables, docs`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from xaibench import config

def test_cells_are_18_unique_pairs():
    assert len(config.CELLS) == 18
    assert len(set(config.CELLS)) == 18
    assert set(d for d, _ in config.CELLS) == set(config.DATASETS)
    assert set(m for _, m in config.CELLS) == set(config.MODELS)

def test_quick_is_cheaper_than_full_on_every_budget():
    q, f = config.QUICK, config.FULL
    assert q.n_explain < f.n_explain
    assert q.rerun_repeats < f.rerun_repeats
    assert q.robust_neighbours < f.robust_neighbours
    assert q.bootstrap_refits < f.bootstrap_refits
    assert q.ci_resamples < f.ci_resamples
    assert max(q.lime_num_samples) < max(f.lime_num_samples)

def test_default_profile_is_quick_so_accidents_are_cheap(monkeypatch):
    monkeypatch.delenv("XAIB_PROFILE", raising=False)
    assert config.get_profile().name == "QUICK"
    monkeypatch.setenv("XAIB_PROFILE", "FULL")
    assert config.get_profile().name == "FULL"

def test_protocol_constants_match_frozen_protocol():
    # These values are normative in docs/PROTOCOL.md. If a test here fails,
    # either the code drifted or the protocol was amended without logging it.
    assert config.SEED == 42
    assert config.ALPHA == 0.05
    assert config.RRF_K == 60
    assert config.AOPC_MAX_K == 10
    assert config.CALIB_FRACTION == 0.30
    assert config.FULL.rerun_repeats == 10
    assert config.FULL.robust_neighbours == 5
    assert config.FULL.bootstrap_refits == 10
    assert config.FULL.n_explain == 150
    assert config.FULL.ci_resamples == 2000
    assert config.FULL.lime_num_samples == (5000, 1000)
    assert config.ENSEMBLE_MODELS == ("rf", "xgb", "lgbm")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench'`

- [ ] **Step 3: Create the environment and pin dependencies**

```bash
cd /home/marajul/AI_WorkShop/Undergrade_Thesis
uv venv --python 3.12
uv pip install --python .venv/bin/python \
  "numpy<2" "pandas>=2.2" "scikit-learn>=1.5" "scipy>=1.11" \
  "matplotlib>=3.8" "seaborn>=0.13" "shap>=0.45" "lime>=0.2.0.1" \
  "xgboost>=2.1" "lightgbm>=4.5" "statsmodels>=0.14" \
  "pyarrow>=16" "pyyaml>=6" "joblib>=1.4" "tqdm>=4.66" "pytest>=8.3"
uv pip freeze --python .venv/bin/python > requirements.lock.txt
```

`numpy<2` is not optional — see Global Constraints.

- [ ] **Step 4: Write `pyproject.toml`**

```toml
[project]
name = "xaibench"
version = "1.0.0"
description = "SHAP vs LIME explanation benchmark on clinical tabular data"
requires-python = ">=3.12"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
filterwarnings = ["ignore::FutureWarning", "ignore::DeprecationWarning"]
```

`pythonpath = ["src"]` means pytest imports `xaibench` without an editable install, which keeps the venv reproducible from `requirements.lock.txt` alone.

- [ ] **Step 5: Write `src/xaibench/config.py`**

```python
"""Central configuration. Values marked NORMATIVE are fixed by docs/PROTOCOL.md."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

SEED = 42                    # NORMATIVE
ALPHA = 0.05                 # NORMATIVE
RRF_K = 60                   # NORMATIVE
AOPC_MAX_K = 10              # NORMATIVE
CALIB_FRACTION = 0.30        # NORMATIVE
TEST_SIZE = 0.20             # NORMATIVE
LIPSCHITZ_SIGMA = 0.10       # NORMATIVE: neighbour sd, in units of train sd
RANDOM_CONTROL_DRAWS = 20    # NORMATIVE
BASELINE_MARGINAL_DRAWS = 10 # NORMATIVE
JACCARD_KS = (3, 5)          # NORMATIVE
RBO_P = 0.90                 # NORMATIVE

DATASETS = ("pima", "heart", "ckd")
MODELS = ("logreg", "dtree", "rf", "xgb", "lgbm", "stack")
ENSEMBLE_MODELS = ("rf", "xgb", "lgbm")     # NORMATIVE: RQ2c scope (protocol R3)
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

    @property
    def is_reportable(self) -> bool:
        return self.name == "FULL"


QUICK = Profile(
    name="QUICK", n_explain=12, lime_num_samples=(500,), rerun_repeats=3,
    robust_neighbours=2, robust_instances=4, rerun_instances=4,
    bootstrap_refits=3, bootstrap_instances=4, ci_resamples=200, shap_background=20,
)

FULL = Profile(                                     # all values NORMATIVE
    name="FULL", n_explain=150, lime_num_samples=(5000, 1000), rerun_repeats=10,
    robust_neighbours=5, robust_instances=25, rerun_instances=30,
    bootstrap_refits=10, bootstrap_instances=25, ci_resamples=2000, shap_background=50,
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

    def mkdirs(self) -> None:
        for f in (self.data_raw, self.data_processed, self.results,
                  self.attributions, self.figures, self.tables):
            f.mkdir(parents=True, exist_ok=True)


def paths() -> Paths:
    root = Path(__file__).resolve().parents[2]
    return Paths(
        root=root, data_raw=root / "data" / "raw", data_processed=root / "data" / "processed",
        results=root / "results", attributions=root / "results" / "attributions",
        figures=root / "figures", tables=root / "tables", docs=root / "docs",
    )
```

- [ ] **Step 6: Write `tests/conftest.py`**

```python
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XAIB_PROFILE", "QUICK")
```

- [ ] **Step 7: Write `.gitignore`**

```
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
data/processed/
results/
figures/
tables/
*.aux
*.log
*.out
*.bbl
*.blg
*.nav
*.snm
*.toc
*.fdb_latexmk
*.fls
*.synctex.gz
```

`data/raw/` is deliberately **not** ignored: vendoring the source CSVs is what makes the package runnable offline. `results/`, `figures/`, and `tables/` are ignored because they are regenerable and `make all` is the contract; the compiled PDFs are committed separately in Task 24.

- [ ] **Step 8: Write the `Makefile`**

```makefile
PY := .venv/bin/python
PROFILE ?= QUICK
export OMP_NUM_THREADS = 1
export PYTHONHASHSEED = 42

.PHONY: setup data test smoke full all figures tables paper report slides docs clean

setup:
	uv venv --python 3.12
	uv pip install --python .venv/bin/python -r requirements.lock.txt

data:
	$(PY) -m xaibench.data.fetch

test:
	.venv/bin/pytest -q

stage1: ; XAIB_PROFILE=$(PROFILE) $(PY) -m xaibench.pipeline.stage1_performance
stage2: ; XAIB_PROFILE=$(PROFILE) $(PY) -m xaibench.pipeline.stage2_explain
stage3: ; XAIB_PROFILE=$(PROFILE) $(PY) -m xaibench.pipeline.stage3_metrics
stage4: ; XAIB_PROFILE=$(PROFILE) $(PY) -m xaibench.pipeline.stage4_stats
stage5: ; XAIB_PROFILE=$(PROFILE) $(PY) -m xaibench.pipeline.stage5_outputs

pipeline: stage1 stage2 stage3 stage4 stage5

smoke: ; $(MAKE) data && $(MAKE) test && $(MAKE) pipeline PROFILE=QUICK
full:  ; $(MAKE) data && $(MAKE) test && $(MAKE) pipeline PROFILE=FULL

figures tables: stage5

paper:  ; cd paper  && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
report: ; cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
slides: ; cd slides && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
docs: paper report slides

all: full docs

clean:
	rm -rf results figures tables
	cd paper && latexmk -C; cd ../report && latexmk -C; cd ../slides && latexmk -C
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml requirements.lock.txt Makefile .gitignore src tests
git commit -m "feat: scaffold package, pinned environment, and configuration profiles

QUICK is the default profile so an accidental run is never the two-hour one.
Protocol constants are asserted in tests, so drift between code and
docs/PROTOCOL.md fails the suite rather than passing silently."
```

---

### Task 2: Dataset acquisition and loading

**Files:**
- Create: `src/xaibench/data/fetch.py`, `src/xaibench/data/loaders.py`
- Create: `tests/test_loaders.py`
- Produces: `data/raw/{pima,heart,ckd}.csv`, `data/raw/checksums.json`

**Interfaces:**
- Consumes: `config.paths()`, `config.DATASETS`.
- Produces: `Dataset` frozen dataclass with fields `id: str`, `X: pd.DataFrame`, `y: pd.Series`, `numeric: list[str]`, `categorical: list[str]`, `positive_rate: float`, `source_url: str`, `licence: str`, `citation: str`; function `load_dataset(dataset_id: str) -> Dataset`; function `fetch_all(force: bool = False) -> dict[str, str]` returning `{dataset_id: sha256}`; constant `SOURCES: dict[str, Source]`.

- [ ] **Step 1: Write the failing test**

The expected shapes come straight from `docs/PROTOCOL.md` §2.1. If a source silently changes, this test fails rather than the study reporting wrong *n*.

```python
# tests/test_loaders.py
import numpy as np
import pytest
from xaibench.data.loaders import load_dataset

EXPECTED = {   # (n_rows, n_features, positive_rate) — NORMATIVE, protocol Table T1
    "pima":  (768,  8, 0.349),
    "heart": (303, 13, 0.459),
    "ckd":   (400, 24, 0.625),
}

@pytest.mark.parametrize("ds_id", ["pima", "heart", "ckd"])
def test_shape_and_balance_match_protocol(ds_id):
    n, d, rate = EXPECTED[ds_id]
    ds = load_dataset(ds_id)
    assert ds.X.shape == (n, d)
    assert len(ds.y) == n
    assert ds.y.isin([0, 1]).all()
    assert ds.positive_rate == pytest.approx(rate, abs=0.005)

@pytest.mark.parametrize("ds_id", ["pima", "heart", "ckd"])
def test_column_lists_partition_the_frame(ds_id):
    ds = load_dataset(ds_id)
    assert set(ds.numeric) | set(ds.categorical) == set(ds.X.columns)
    assert not (set(ds.numeric) & set(ds.categorical))
    for c in ds.numeric:
        assert np.issubdtype(ds.X[c].dtype, np.number)

def test_pima_impossible_zeros_became_missing():
    ds = load_dataset("pima")
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        assert (ds.X[col] == 0).sum() == 0, f"{col} still has impossible zeros"
        assert ds.X[col].isna().sum() > 0, f"{col} should now carry NaNs"
    # Pregnancies zeros are legitimate and must survive
    assert (ds.X["Pregnancies"] == 0).sum() > 0

def test_heart_target_is_binarised_from_five_levels():
    ds = load_dataset("heart")
    assert "num" not in ds.X.columns
    assert set(ds.y.unique()) == {0, 1}

def test_ckd_object_columns_are_stripped_of_stray_whitespace():
    ds = load_dataset("ckd")
    for c in ds.categorical:
        vals = ds.X[c].dropna().astype(str)
        assert (vals == vals.str.strip()).all(), f"{c} carries untrimmed values"
        assert not vals.str.contains("\t").any()

@pytest.mark.parametrize("ds_id", ["pima", "heart", "ckd"])
def test_metadata_is_present_for_the_paper(ds_id):
    ds = load_dataset(ds_id)
    assert ds.source_url.startswith("https://")
    assert ds.licence and ds.citation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_loaders.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.data.loaders'`

- [ ] **Step 3: Write `src/xaibench/data/fetch.py`**

All three URLs were verified reachable on 2026-08-29. Checksums are recorded on first fetch and asserted on every later run; a mismatch is a hard failure because a silently changed source invalidates every number in the paper.

```python
"""Download, verify, and vendor the three source datasets."""
from __future__ import annotations
import hashlib, json, urllib.request
from dataclasses import dataclass
from xaibench.config import paths

@dataclass(frozen=True)
class Source:
    dataset_id: str
    url: str
    licence: str
    citation: str
    has_header: bool
    column_names: tuple[str, ...] | None = None

SOURCES = {
    "pima": Source(
        "pima",
        "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        "CC0 1.0 (public domain); originally collected by the US National Institute of "
        "Diabetes and Digestive and Kidney Diseases",
        "J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes, "
        "'Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,' "
        "in Proc. Annu. Symp. Comput. Appl. Med. Care, 1988, pp. 261-265.",
        has_header=True,
    ),
    "heart": Source(
        "heart",
        "https://archive.ics.uci.edu/static/public/45/data.csv",
        "CC BY 4.0 (UCI Machine Learning Repository, dataset 45)",
        "R. Detrano et al., 'International application of a new probability algorithm for "
        "the diagnosis of coronary artery disease,' Am. J. Cardiol., vol. 64, no. 5, "
        "pp. 304-310, 1989. UCI ML Repository, doi:10.24432/C52P4X.",
        has_header=True,
    ),
    "ckd": Source(
        "ckd",
        "https://archive.ics.uci.edu/static/public/336/data.csv",
        "CC BY 4.0 (UCI Machine Learning Repository, dataset 336)",
        "L. J. Rubini, P. Soundarapandian, and P. Eswaran, 'Chronic Kidney Disease,' "
        "UCI Machine Learning Repository, 2015, doi:10.24432/C5G020.",
        has_header=True,
    ),
}


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_all(force: bool = False) -> dict[str, str]:
    p = paths(); p.mkdirs()
    lock_file = p.data_raw / "checksums.json"
    recorded = json.loads(lock_file.read_text()) if lock_file.exists() else {}
    digests: dict[str, str] = {}

    for ds_id, src in SOURCES.items():
        dest = p.data_raw / f"{ds_id}.csv"
        if force or not dest.exists():
            print(f"[fetch] {ds_id} <- {src.url}")
            req = urllib.request.Request(src.url, headers={"User-Agent": "xaibench/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                dest.write_bytes(r.read())
        digest = _sha256(dest)
        if ds_id in recorded and recorded[ds_id] != digest:
            raise RuntimeError(
                f"{ds_id}: SHA-256 mismatch.\n  recorded {recorded[ds_id]}\n  found    {digest}\n"
                "The upstream source changed. Every result derived from it is now suspect. "
                "Investigate before rerunning; do not overwrite checksums.json to silence this."
            )
        digests[ds_id] = digest
        print(f"[fetch] {ds_id}: {dest.stat().st_size:,} bytes  sha256={digest[:16]}...")

    if not recorded:
        lock_file.write_text(json.dumps(digests, indent=2) + "\n")
        print(f"[fetch] recorded checksums -> {lock_file}")
    return digests


if __name__ == "__main__":
    fetch_all()
```

- [ ] **Step 4: Write `src/xaibench/data/loaders.py`**

```python
"""Load the three datasets with their per-dataset semantics fixes applied."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from xaibench.config import paths
from xaibench.data.fetch import SOURCES, fetch_all

PIMA_IMPOSSIBLE_ZEROS = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
HEART_CATEGORICAL = ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")


@dataclass(frozen=True)
class Dataset:
    id: str
    X: pd.DataFrame
    y: pd.Series
    numeric: list[str]
    categorical: list[str]
    source_url: str
    licence: str
    citation: str

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean())

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def missing_rate(self) -> float:
        return float(self.X.isna().to_numpy().mean())


def _raw(dataset_id: str) -> pd.DataFrame:
    p = paths().data_raw / f"{dataset_id}.csv"
    if not p.exists():
        fetch_all()
    return pd.read_csv(p)


def _load_pima() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = _raw("pima")
    y = df.pop("Outcome").astype(int)
    # Biologically impossible zeros are missing data, not measurements. This is a
    # data-semantics correction applied at load time, so it is not fold-dependent.
    for col in PIMA_IMPOSSIBLE_ZEROS:
        df[col] = df[col].replace(0, np.nan)
    return df, y, list(df.columns), []


def _load_heart() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = _raw("heart")
    y = (pd.to_numeric(df.pop("num"), errors="coerce").fillna(0) > 0).astype(int)
    df = df.apply(pd.to_numeric, errors="coerce")
    categorical = [c for c in HEART_CATEGORICAL if c in df.columns]
    numeric = [c for c in df.columns if c not in categorical]
    # Nominal codes must not be treated as magnitudes by the linear model.
    for c in categorical:
        df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
    return df, y, numeric, categorical


def _load_ckd() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = _raw("ckd")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.replace("\t", "", regex=False)
            df[c] = df[c].replace({"?": np.nan, "nan": np.nan, "NaN": np.nan, "": np.nan})
    y = df.pop("class").map({"ckd": 1, "notckd": 0}).astype(int)
    numeric, categorical = [], []
    for c in df.columns:
        coerced = pd.to_numeric(df[c], errors="coerce")
        # A column is numeric only if coercion loses no observed value.
        if coerced.notna().sum() == df[c].notna().sum():
            df[c] = coerced
            numeric.append(c)
        else:
            categorical.append(c)
    return df, y, numeric, categorical


_LOADERS = {"pima": _load_pima, "heart": _load_heart, "ckd": _load_ckd}


def load_dataset(dataset_id: str) -> Dataset:
    if dataset_id not in _LOADERS:
        raise ValueError(f"unknown dataset {dataset_id!r}; expected one of {sorted(_LOADERS)}")
    X, y, numeric, categorical = _LOADERS[dataset_id]()
    src = SOURCES[dataset_id]
    return Dataset(id=dataset_id, X=X.reset_index(drop=True), y=y.reset_index(drop=True),
                   numeric=numeric, categorical=categorical,
                   source_url=src.url, licence=src.licence, citation=src.citation)
```

- [ ] **Step 5: Fetch the data and run the tests**

Run: `.venv/bin/python -m xaibench.data.fetch && .venv/bin/pytest tests/test_loaders.py -v`
Expected: checksums recorded for three files; 15 passed.

If `test_shape_and_balance_match_protocol` fails on `heart` with 303 vs 304 rows, the CSV's trailing newline is being read as a row — confirm with `wc -l`, and if the source genuinely changed, stop and log an amendment in `docs/PROTOCOL.md` §10 rather than adjusting the expected value.

- [ ] **Step 6: Commit**

```bash
git add src/xaibench/data tests/test_loaders.py data/raw
git commit -m "feat: dataset acquisition with checksum verification and semantics fixes

Vendors the three source CSVs so the package runs offline, and asserts
SHA-256 on every load: a silently changed upstream source would invalidate
every number in the paper, so it fails loudly instead. Expected shapes and
class balances are asserted against docs/PROTOCOL.md Table T1."
```

---

### Task 3: Leakage-safe preprocessing and one-hot attribution aggregation

**Files:**
- Create: `src/xaibench/models/pipelines.py`
- Create: `tests/test_pipelines.py`

**Interfaces:**
- Consumes: `Dataset` from Task 2.
- Produces: `build_preprocessor(ds: Dataset) -> ColumnTransformer`; `onehot_group_map(preprocessor, ds) -> list[str]` returning one original-feature name per *output* column, in output-column order; `aggregate_attributions(values: np.ndarray, groups: list[str], feature_order: list[str]) -> np.ndarray` mapping `(n, n_out) -> (n, n_features)`; `transformed_feature_names(preprocessor) -> list[str]`; `categorical_output_indices(groups, ds) -> list[int]`.

- [ ] **Step 1: Write the failing test**

Aggregation is where a comparison study silently breaks: one-hot encoding fragments a categorical feature across several model inputs, so rankings from different models would otherwise be over different vocabularies. These tests pin the invariant that matters — the aggregated value equals the exact sum of its constituent columns.

```python
# tests/test_pipelines.py
import numpy as np
import pandas as pd
import pytest
from xaibench.data.loaders import Dataset, load_dataset
from xaibench.models.pipelines import (
    build_preprocessor, onehot_group_map, aggregate_attributions,
    transformed_feature_names, categorical_output_indices,
)

def _toy() -> Dataset:
    X = pd.DataFrame({
        "age":   [20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        "bp":    [80.0, 85.0, np.nan, 95.0, 100.0, 105.0],
        "grade": ["low", "mid", "high", "low", "mid", "high"],   # 3 levels
        "flag":  ["yes", "no", "yes", "no", "yes", "no"],         # binary
    })
    y = pd.Series([0, 1, 0, 1, 0, 1])
    return Dataset(id="toy", X=X, y=y, numeric=["age", "bp"], categorical=["grade", "flag"],
                   source_url="https://example.invalid", licence="n/a", citation="n/a")

def test_group_map_has_one_entry_per_output_column():
    ds = _toy()
    pre = build_preprocessor(ds).fit(ds.X)
    n_out = pre.transform(ds.X).shape[1]
    groups = onehot_group_map(pre, ds)
    assert len(groups) == n_out
    assert set(groups) == set(ds.X.columns)

def test_three_level_categorical_spans_three_columns_and_binary_spans_one():
    ds = _toy()
    pre = build_preprocessor(ds).fit(ds.X)
    groups = onehot_group_map(pre, ds)
    assert groups.count("grade") == 3
    assert groups.count("flag") == 1     # drop='if_binary'
    assert groups.count("age") == 1 and groups.count("bp") == 1

def test_aggregation_sums_exactly_over_constituent_columns():
    ds = _toy()
    pre = build_preprocessor(ds).fit(ds.X)
    groups = onehot_group_map(pre, ds)
    order = list(ds.X.columns)
    rng = np.random.default_rng(0)
    values = rng.normal(size=(5, len(groups)))
    agg = aggregate_attributions(values, groups, order)
    assert agg.shape == (5, len(order))
    for j, feat in enumerate(order):
        cols = [i for i, g in enumerate(groups) if g == feat]
        np.testing.assert_allclose(agg[:, j], values[:, cols].sum(axis=1))

def test_aggregation_preserves_total_attribution():
    # Additivity of SHAP must survive aggregation: the row sum cannot change.
    ds = _toy()
    pre = build_preprocessor(ds).fit(ds.X)
    groups = onehot_group_map(pre, ds)
    values = np.random.default_rng(1).normal(size=(7, len(groups)))
    agg = aggregate_attributions(values, groups, list(ds.X.columns))
    np.testing.assert_allclose(agg.sum(axis=1), values.sum(axis=1), atol=1e-12)

def test_imputer_statistics_differ_between_folds_so_nothing_leaks():
    # If preprocessing were fitted once on all data, these medians would be equal.
    ds = load_dataset("pima")
    a = build_preprocessor(ds).fit(ds.X.iloc[:300])
    b = build_preprocessor(ds).fit(ds.X.iloc[300:])
    ma = a.named_transformers_["num"].named_steps["impute"].statistics_
    mb = b.named_transformers_["num"].named_steps["impute"].statistics_
    assert not np.allclose(ma, mb)

def test_transform_leaves_no_nan_for_the_explainers():
    for ds_id in ["pima", "heart", "ckd"]:
        ds = load_dataset(ds_id)
        pre = build_preprocessor(ds).fit(ds.X)
        out = pre.transform(ds.X)
        assert np.isfinite(np.asarray(out, dtype=float)).all(), ds_id

def test_categorical_output_indices_point_only_at_categoricals():
    ds = _toy()
    pre = build_preprocessor(ds).fit(ds.X)
    groups = onehot_group_map(pre, ds)
    idx = categorical_output_indices(groups, ds)
    assert all(groups[i] in ds.categorical for i in idx)
    assert len(idx) == groups.count("grade") + groups.count("flag")

def test_unseen_category_at_transform_time_does_not_raise():
    ds = _toy()
    pre = build_preprocessor(ds).fit(ds.X)
    novel = ds.X.copy(); novel.loc[0, "grade"] = "unseen-level"
    pre.transform(novel)   # handle_unknown='ignore'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_pipelines.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.models.pipelines'`

- [ ] **Step 3: Write `src/xaibench/models/pipelines.py`**

The group map is built from the fitted encoder's `categories_` and `drop_idx_`, never by parsing generated column-name strings. Parsing breaks on feature names that are prefixes of each other (CKD has both `pc` and `pcc`) and on category values containing underscores.

```python
"""Leakage-safe preprocessing and the one-hot -> original-feature aggregation map."""
from __future__ import annotations
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(ds) -> ColumnTransformer:
    """One ColumnTransformer, always fitted inside a fold or on a train partition only."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                 sparse_output=False, dtype=np.float64)),
    ])
    transformers = []
    if ds.numeric:
        transformers.append(("num", numeric, list(ds.numeric)))
    if ds.categorical:
        transformers.append(("cat", categorical, list(ds.categorical)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def onehot_group_map(preprocessor: ColumnTransformer, ds) -> list[str]:
    """One original-feature name per output column, in output-column order.

    Built from the fitted encoder's own `categories_`/`drop_idx_` rather than from
    generated column-name strings, which are ambiguous when one feature name is a
    prefix of another (CKD has both `pc` and `pcc`).
    """
    groups: list[str] = list(ds.numeric)
    if ds.categorical:
        ohe = preprocessor.named_transformers_["cat"].named_steps["encode"]
        drop_idx = getattr(ohe, "drop_idx_", None)
        for i, (col, cats) in enumerate(zip(ds.categorical, ohe.categories_)):
            n_out = len(cats)
            if drop_idx is not None and drop_idx[i] is not None:
                n_out -= 1
            groups.extend([col] * n_out)
    return groups


def transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return [str(n) for n in preprocessor.get_feature_names_out()]


def categorical_output_indices(groups: list[str], ds) -> list[int]:
    """Output-column indices LIME must treat as categorical rather than discretise."""
    cats = set(ds.categorical)
    return [i for i, g in enumerate(groups) if g in cats]


def aggregate_attributions(values: np.ndarray, groups: list[str],
                           feature_order: list[str]) -> np.ndarray:
    """Sum attributions over the one-hot columns belonging to each original feature.

    Summation, not averaging: SHAP attributions are additive, and only a sum keeps
    the row total (and therefore the additivity property) unchanged.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    if values.shape[1] != len(groups):
        raise ValueError(f"values has {values.shape[1]} columns but groups has {len(groups)}")
    index = {f: j for j, f in enumerate(feature_order)}
    out = np.zeros((values.shape[0], len(feature_order)), dtype=float)
    for col, g in enumerate(groups):
        out[:, index[g]] += values[:, col]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipelines.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/xaibench/models/pipelines.py tests/test_pipelines.py
git commit -m "feat: leakage-safe preprocessing and one-hot attribution aggregation

The group map is derived from the fitted encoder's categories_, not by parsing
column-name strings, because CKD has feature names that prefix each other (pc,
pcc). Aggregation sums rather than averages so SHAP's additivity survives, which
test_aggregation_preserves_total_attribution pins. A leakage regression test
asserts imputer statistics differ between folds."
```

---

### Task 4: Model zoo and parameter grids

**Files:**
- Create: `src/xaibench/models/zoo.py`
- Create: `tests/test_zoo.py`

**Interfaces:**
- Consumes: `config.MODELS`.
- Produces: `build_model(model_id: str, seed: int = SEED) -> BaseEstimator`; `param_grid(model_id: str) -> dict[str, list]`; `shap_backend_kind(model_id: str) -> str` returning one of `"tree"`, `"tree_prob"`, `"permutation"`; `MODEL_LABELS: dict[str, str]` for figures.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_zoo.py
import numpy as np
import pytest
from sklearn.base import is_classifier
from xaibench.config import MODELS
from xaibench.models.zoo import build_model, param_grid, shap_backend_kind, MODEL_LABELS

@pytest.mark.parametrize("model_id", MODELS)
def test_every_model_is_a_classifier_with_predict_proba(model_id):
    m = build_model(model_id)
    assert is_classifier(m)
    assert hasattr(m, "predict_proba")

@pytest.mark.parametrize("model_id", MODELS)
def test_every_model_fits_and_predicts_probabilities_in_range(model_id):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6)); y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    p = build_model(model_id).fit(X, y).predict_proba(X)
    assert p.shape == (120, 2)
    assert np.all((p >= 0) & (p <= 1))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)

@pytest.mark.parametrize("model_id", MODELS)
def test_grids_are_small_enough_for_the_compute_budget(model_id):
    grid = param_grid(model_id)
    combos = int(np.prod([len(v) for v in grid.values()])) if grid else 1
    assert 1 <= combos <= 12, f"{model_id} grid has {combos} combinations, budget is 12"

@pytest.mark.parametrize("model_id", MODELS)
def test_grid_keys_are_real_estimator_parameters(model_id):
    valid = build_model(model_id).get_params().keys()
    for key in param_grid(model_id):
        assert key in valid, f"{model_id}: {key} is not a parameter of the estimator"

def test_backend_assignment_matches_the_protocol():
    # dtree/rf output class proportions, so TreeExplainer's raw output is already
    # probability space. xgb/lgbm output log-odds margins and need model_output.
    # logreg/stack have no tree backend, so they use the model-agnostic sampler.
    assert shap_backend_kind("dtree") == "tree"
    assert shap_backend_kind("rf") == "tree"
    assert shap_backend_kind("xgb") == "tree_prob"
    assert shap_backend_kind("lgbm") == "tree_prob"
    assert shap_backend_kind("logreg") == "permutation"
    assert shap_backend_kind("stack") == "permutation"

@pytest.mark.parametrize("model_id", MODELS)
def test_models_are_reproducible_from_the_seed(model_id):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 5)); y = (X[:, 0] > 0).astype(int)
    a = build_model(model_id, seed=42).fit(X, y).predict_proba(X)
    b = build_model(model_id, seed=42).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b)

def test_labels_exist_for_every_model():
    assert set(MODEL_LABELS) == set(MODELS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_zoo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.models.zoo'`

- [ ] **Step 3: Write `src/xaibench/models/zoo.py`**

```python
"""Model factory, parameter grids, and SHAP-backend assignment."""
from __future__ import annotations
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from xaibench.config import SEED

MODEL_LABELS = {
    "logreg": "Logistic Regression", "dtree": "Decision Tree", "rf": "Random Forest",
    "xgb": "XGBoost", "lgbm": "LightGBM", "stack": "Stacking",
}

# NORMATIVE (protocol §2.2). dtree/rf produce class proportions, so TreeExplainer's
# raw output is already probability space. xgb/lgbm produce log-odds margins and need
# model_output="probability" with an interventional background. logreg/stack have no
# tree structure, so they use the model-agnostic permutation sampler on predict_proba.
_BACKEND = {"logreg": "permutation", "dtree": "tree", "rf": "tree",
            "xgb": "tree_prob", "lgbm": "tree_prob", "stack": "permutation"}


def shap_backend_kind(model_id: str) -> str:
    return _BACKEND[model_id]


def _base_learners(seed: int):
    return [
        ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                      class_weight="balanced", random_state=seed, n_jobs=1)),
        ("xgb", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                              subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                              eval_metric="logloss", random_state=seed, n_jobs=1)),
        ("lgbm", LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                min_child_samples=5, class_weight="balanced",
                                random_state=seed, n_jobs=1, verbose=-1)),
    ]


def build_model(model_id: str, seed: int = SEED):
    if model_id == "logreg":
        return LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
    if model_id == "dtree":
        return DecisionTreeClassifier(max_depth=5, min_samples_leaf=5,
                                      class_weight="balanced", random_state=seed)
    if model_id == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      class_weight="balanced", random_state=seed, n_jobs=1)
    if model_id == "xgb":
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1,
                             subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                             eval_metric="logloss", random_state=seed, n_jobs=1)
    if model_id == "lgbm":
        # num_leaves and min_child_samples are deliberately small: these datasets have
        # 303-768 rows and LightGBM's defaults overfit and warn at that scale.
        return LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                              min_child_samples=5, class_weight="balanced",
                              random_state=seed, n_jobs=1, verbose=-1)
    if model_id == "stack":
        return StackingClassifier(
            estimators=_base_learners(seed),
            final_estimator=LogisticRegression(max_iter=5000, class_weight="balanced",
                                               random_state=seed),
            cv=3, stack_method="predict_proba", n_jobs=1, passthrough=False,
        )
    raise ValueError(f"unknown model {model_id!r}")


# Grids are capped at 12 combinations per model: with 18 cells x 5 outer x 3 inner
# folds, anything larger stops fitting inside the two-hour budget.
_GRIDS = {
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0]},
    "dtree":  {"max_depth": [3, 5, 8], "min_samples_leaf": [2, 5, 10]},
    "rf":     {"max_depth": [None, 8], "min_samples_leaf": [1, 2, 5]},
    "xgb":    {"max_depth": [2, 3, 5], "learning_rate": [0.05, 0.1]},
    "lgbm":   {"num_leaves": [7, 15, 31], "learning_rate": [0.05, 0.1]},
    "stack":  {"final_estimator__C": [0.1, 1.0, 10.0]},
}


def param_grid(model_id: str) -> dict[str, list]:
    if model_id not in _GRIDS:
        raise ValueError(f"unknown model {model_id!r}")
    return {k: list(v) for k, v in _GRIDS[model_id].items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_zoo.py -v`
Expected: 31 passed

- [ ] **Step 5: Commit**

```bash
git add src/xaibench/models/zoo.py tests/test_zoo.py
git commit -m "feat: model zoo, parameter grids, and SHAP backend assignment

Backend assignment is asserted in tests because it is the subtle part: dtree and
rf already emit probabilities from TreeExplainer's raw output, xgb and lgbm emit
log-odds and need model_output='probability', and logreg and stack have no tree
backend at all. Grids are capped at 12 combinations to fit the compute budget."
```

---

## Phase 1 — Metrics (pure functions, test-first)

Every metric in this phase is a pure function of arrays, so each one is checkable against a case whose answer is known analytically rather than against the code's own output. That is the whole reason this phase comes before the pipeline: a bug in `aopc_deletion` discovered after the two-hour run costs a day.

### Task 5: Agreement metrics

**Files:**
- Create: `src/xaibench/metrics/agreement.py`
- Create: `tests/test_agreement.py`

**Interfaces:**
- Consumes: nothing but numpy.
- Produces: `ranking_from_attributions(attr: np.ndarray) -> np.ndarray` (indices, most important first, by descending `|attr|`, ties by ascending index); `spearman(a: np.ndarray, b: np.ndarray) -> float`; `kendall_tau_b(a, b) -> float`; `jaccard_at_k(rank_a, rank_b, k: int) -> float`; `rbo(rank_a, rank_b, p: float = 0.9) -> float`; `topk_set(rank, k) -> set[int]`. `spearman` and `kendall_tau_b` take **rank vectors** (position of each feature), not orderings.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agreement.py
import numpy as np
import pytest
from xaibench.metrics.agreement import (
    ranking_from_attributions, spearman, kendall_tau_b, jaccard_at_k, rbo, topk_set,
)

def _rank_vector(order):
    """order = indices most-important-first -> rank position per feature."""
    r = np.empty(len(order), dtype=float)
    for pos, idx in enumerate(order):
        r[idx] = pos
    return r

def test_ranking_uses_absolute_value_so_sign_does_not_reorder():
    attr = np.array([0.1, -0.9, 0.5, -0.2])
    np.testing.assert_array_equal(ranking_from_attributions(attr), [1, 2, 3, 0])

def test_ranking_breaks_ties_by_index_for_determinism():
    attr = np.array([0.5, 0.5, 0.5])
    np.testing.assert_array_equal(ranking_from_attributions(attr), [0, 1, 2])

def test_identical_rankings_score_maximum_on_every_metric():
    order = [3, 0, 2, 1, 4]
    r = _rank_vector(order)
    assert spearman(r, r) == pytest.approx(1.0)
    assert kendall_tau_b(r, r) == pytest.approx(1.0)
    assert jaccard_at_k(order, order, 3) == pytest.approx(1.0)
    assert rbo(order, order, p=0.9) == pytest.approx(1.0)

def test_reversed_rankings_are_perfectly_discordant():
    order = [0, 1, 2, 3, 4]
    rev = order[::-1]
    assert spearman(_rank_vector(order), _rank_vector(rev)) == pytest.approx(-1.0)
    assert kendall_tau_b(_rank_vector(order), _rank_vector(rev)) == pytest.approx(-1.0)
    assert jaccard_at_k(order, rev, 2) == pytest.approx(0.0)
    assert rbo(order, rev, p=0.9) < 0.35

def test_jaccard_at_k_on_hand_computed_sets():
    a = [0, 1, 2, 3, 4]
    b = [0, 1, 5, 6, 7]
    # top-3: {0,1,2} vs {0,1,5} -> intersection 2, union 4
    assert jaccard_at_k(a, b, 3) == pytest.approx(2 / 4)
    # top-5: {0,1,2,3,4} vs {0,1,5,6,7} -> intersection 2, union 8
    assert jaccard_at_k(a, b, 5) == pytest.approx(2 / 8)

def test_jaccard_k_larger_than_list_is_clipped_not_an_error():
    assert jaccard_at_k([0, 1], [1, 0], 5) == pytest.approx(1.0)

def test_rbo_matches_a_hand_computed_three_item_case():
    # S=[a,b,c], T=[a,c,b] with p=0.9.
    # A_1 = 1, A_2 = 1/2, A_3 = 1
    # RBO_ext = (1-p)(p^0*1 + p^1*0.5 + p^2*1) + p^3 * A_3
    #         = 0.1*(1 + 0.45 + 0.81) + 0.729 = 0.226 + 0.729 = 0.955
    assert rbo([0, 1, 2], [0, 2, 1], p=0.9) == pytest.approx(0.955, abs=1e-9)

def test_rbo_is_top_weighted_unlike_spearman():
    # Swapping ranks 1 and 2 must cost more than swapping ranks 5 and 6.
    base = [0, 1, 2, 3, 4, 5, 6, 7]
    swap_head = [1, 0, 2, 3, 4, 5, 6, 7]
    swap_tail = [0, 1, 2, 3, 5, 4, 6, 7]
    assert rbo(base, swap_head, p=0.9) < rbo(base, swap_tail, p=0.9)

def test_rbo_is_symmetric_and_bounded():
    rng = np.random.default_rng(3)
    for _ in range(20):
        a = list(rng.permutation(8)); b = list(rng.permutation(8))
        v = rbo(a, b, p=0.9)
        assert 0.0 <= v <= 1.0
        assert v == pytest.approx(rbo(b, a, p=0.9))

def test_spearman_handles_constant_input_without_nan():
    # A degenerate explanation (all attributions equal) must not poison aggregation.
    assert spearman(np.zeros(5), np.arange(5.0)) == pytest.approx(0.0)

def test_topk_set_returns_first_k_indices():
    assert topk_set([4, 2, 7, 1], 2) == {4, 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_agreement.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.metrics.agreement'`

- [ ] **Step 3: Write `src/xaibench/metrics/agreement.py`**

```python
"""Rank-agreement metrics. Rankings are index orderings; rank vectors are positions."""
from __future__ import annotations
import numpy as np
from scipy import stats


def ranking_from_attributions(attr: np.ndarray) -> np.ndarray:
    """Feature indices ordered by descending |attribution|, ties broken by index.

    Ties must break deterministically or two runs of the same deterministic explainer
    would disagree. lexsort with the index as the secondary key does that.
    """
    a = np.abs(np.asarray(attr, dtype=float))
    return np.lexsort((np.arange(a.size), -a))


def rank_vector(order) -> np.ndarray:
    """Inverse of a ranking: position occupied by each feature."""
    order = np.asarray(order, dtype=int)
    out = np.empty(order.size, dtype=float)
    out[order] = np.arange(order.size, dtype=float)
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0          # constant input: undefined correlation, reported as no association
    rho = stats.spearmanr(a, b).statistic
    return 0.0 if not np.isfinite(rho) else float(rho)


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    tau = stats.kendalltau(a, b, variant="b").statistic
    return 0.0 if not np.isfinite(tau) else float(tau)


def topk_set(order, k: int) -> set[int]:
    order = list(np.asarray(order, dtype=int))
    return set(order[: max(1, min(k, len(order)))])


def jaccard_at_k(order_a, order_b, k: int) -> float:
    sa, sb = topk_set(order_a, k), topk_set(order_b, k)
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else 1.0


def rbo(order_a, order_b, p: float = 0.90) -> float:
    """Extrapolated Rank-Biased Overlap (Webber, Moffat & Zobel, 2010).

    Both arguments are full permutations of the same feature set, so the tail
    agreement A_k is exactly 1 and the extrapolation term reduces to p**k. Unlike
    Jaccard, RBO needs no choice of k; unlike Spearman, it weights the head of the
    list, which is what a practitioner reading a top-5 explanation actually uses.
    """
    a = list(np.asarray(order_a, dtype=int))
    b = list(np.asarray(order_b, dtype=int))
    if len(a) != len(b):
        raise ValueError("RBO here assumes equal-length permutations of one feature set")
    k = len(a)
    if k == 0:
        return 1.0
    seen_a: set[int] = set()
    seen_b: set[int] = set()
    overlap = 0
    weighted = 0.0
    for d in range(1, k + 1):
        seen_a.add(a[d - 1])
        seen_b.add(b[d - 1])
        if a[d - 1] in seen_b:
            overlap += 1
        if b[d - 1] in seen_a and a[d - 1] != b[d - 1]:
            overlap += 1
        weighted += (p ** (d - 1)) * (overlap / d)
    return float((1.0 - p) * weighted + (p ** k) * (overlap / k))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_agreement.py -v`
Expected: 11 passed. If `test_rbo_matches_a_hand_computed_three_item_case` fails, the incremental overlap counter is wrong — `overlap` must equal `|top-d(a) ∩ top-d(b)|` at the end of each iteration; assert that directly against a set intersection while debugging.

- [ ] **Step 5: Commit**

```bash
git add src/xaibench/metrics/agreement.py tests/test_agreement.py
git commit -m "feat: agreement metrics with analytically verified test cases

RBO is checked against a hand-computed three-item case and against the property
that distinguishes it from Spearman: a head swap must cost more than a tail swap.
Ties break by index so a deterministic explainer cannot disagree with itself."
```

---

### Task 6: Fidelity metrics

**Files:**
- Create: `src/xaibench/metrics/fidelity.py`
- Create: `tests/test_fidelity.py`

**Interfaces:**
- Consumes: `config.AOPC_MAX_K`.
- Produces: `deletion_curve(predict_proba, x, ranking, baseline, k_max) -> np.ndarray` of length `k_max+1` holding the model's probability for its own originally-predicted class after deleting the top-0…top-k features; `aopc_deletion(...) -> float`; `insertion_curve(...) -> np.ndarray`; `auc_insertion(...) -> float`; `comprehensiveness(..., k=5) -> float`; `sufficiency(..., k=5) -> float`; `median_baseline(X: np.ndarray) -> np.ndarray`; `marginal_baselines(X, n_draws, seed) -> np.ndarray`. `predict_proba` is called on a 2-D array and returns `(n, 2)`.

- [ ] **Step 1: Write the failing test**

The synthetic model `f(x) = σ(3x₀ + 2x₁ + 0·x₂)` has a known true importance order `(x₀, x₁, x₂)`, so the correct ranking must maximise AOPC over all six permutations. That single assertion catches sign errors, off-by-one indexing in the deletion loop, and using the wrong class column — the three ways this code actually breaks.

```python
# tests/test_fidelity.py
import itertools
import numpy as np
import pytest
from xaibench.metrics.fidelity import (
    deletion_curve, aopc_deletion, insertion_curve, auc_insertion,
    comprehensiveness, sufficiency, median_baseline, marginal_baselines,
)

COEF = np.array([3.0, 2.0, 0.0])

def linear_proba(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    p1 = 1.0 / (1.0 + np.exp(-(X @ COEF)))
    return np.column_stack([1.0 - p1, p1])

X_INSTANCE = np.array([1.0, 1.0, 1.0])
BASELINE = np.zeros(3)

def test_correct_ranking_maximises_aopc_over_all_permutations():
    scores = {perm: aopc_deletion(linear_proba, X_INSTANCE, list(perm), BASELINE, k_max=3)
              for perm in itertools.permutations(range(3))}
    assert max(scores, key=scores.get) == (0, 1, 2)
    assert min(scores, key=scores.get) == (2, 1, 0)

def test_deletion_curve_starts_at_the_undeleted_prediction():
    curve = deletion_curve(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k_max=3)
    assert len(curve) == 4
    assert curve[0] == pytest.approx(linear_proba(X_INSTANCE)[0, 1])

def test_deletion_curve_is_monotone_for_an_additive_positive_model():
    curve = deletion_curve(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k_max=3)
    assert np.all(np.diff(curve) <= 1e-12)

def test_deleting_a_zero_coefficient_feature_changes_nothing():
    curve = deletion_curve(linear_proba, X_INSTANCE, [2, 0, 1], BASELINE, k_max=1)
    assert curve[1] == pytest.approx(curve[0])

def test_insertion_curve_ends_at_the_full_prediction():
    curve = insertion_curve(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k_max=3)
    assert curve[0] == pytest.approx(linear_proba(BASELINE)[0, 1])
    assert curve[-1] == pytest.approx(linear_proba(X_INSTANCE)[0, 1])

def test_auc_insertion_prefers_the_correct_ranking():
    good = auc_insertion(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k_max=3)
    bad = auc_insertion(linear_proba, X_INSTANCE, [2, 1, 0], BASELINE, k_max=3)
    assert good > bad

def test_metrics_track_the_models_own_predicted_class_not_class_one():
    # For this instance the model predicts class 0, so a faithful deletion must
    # reduce p(class 0). Anything hard-coded to column 1 gets this backwards.
    x_neg = np.array([-1.0, -1.0, 0.0])
    assert linear_proba(x_neg)[0].argmax() == 0
    curve = deletion_curve(linear_proba, x_neg, [0, 1, 2], BASELINE, k_max=3)
    assert curve[0] == pytest.approx(linear_proba(x_neg)[0, 0])
    assert curve[-1] < curve[0]

def test_comprehensiveness_and_sufficiency_have_the_expected_signs():
    comp = comprehensiveness(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k=2)
    suff = sufficiency(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k=2)
    assert comp > 0.0                    # removing the top-2 must hurt
    assert abs(suff) < abs(comp)         # the top-2 alone nearly suffice here

def test_median_baseline_ignores_nan_and_matches_numpy():
    X = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]])
    np.testing.assert_allclose(median_baseline(X), [3.0, 20.0])

def test_marginal_baselines_are_drawn_from_observed_values_only():
    X = np.array([[1.0], [2.0], [3.0]])
    draws = marginal_baselines(X, n_draws=8, seed=0)
    assert draws.shape == (8, 1)
    assert set(np.unique(draws)).issubset({1.0, 2.0, 3.0})

def test_marginal_baselines_are_reproducible_from_the_seed():
    X = np.arange(20.0).reshape(-1, 1)
    np.testing.assert_allclose(marginal_baselines(X, 5, seed=7), marginal_baselines(X, 5, seed=7))

def test_k_max_is_clipped_to_the_feature_count():
    curve = deletion_curve(linear_proba, X_INSTANCE, [0, 1, 2], BASELINE, k_max=99)
    assert len(curve) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fidelity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.metrics.fidelity'`

- [ ] **Step 3: Write `src/xaibench/metrics/fidelity.py`**

Every curve is computed with a **single batched `predict_proba` call** over all `k_max+1` perturbed rows. The datasets are small but there are ~1.7 million curve evaluations across the study; one call per row would dominate the runtime.

```python
"""Deletion/insertion fidelity metrics, evaluated against the model's own class."""
from __future__ import annotations
import numpy as np


def median_baseline(X: np.ndarray) -> np.ndarray:
    return np.nanmedian(np.asarray(X, dtype=float), axis=0)


def marginal_baselines(X: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    """Independent draws from each feature's observed training marginal.

    Median replacement pushes instances off the data manifold, which can inflate
    AOPC. This is the protocol's sensitivity variant (§5.3): sample real observed
    values instead, so perturbed rows stay marginally plausible.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty((n_draws, X.shape[1]), dtype=float)
    for j in range(X.shape[1]):
        col = X[np.isfinite(X[:, j]), j]
        col = col if col.size else np.zeros(1)
        out[:, j] = rng.choice(col, size=n_draws, replace=True)
    return out


def _target_class(predict_proba, x: np.ndarray) -> int:
    return int(np.asarray(predict_proba(np.atleast_2d(x)))[0].argmax())


def _curve(predict_proba, rows: np.ndarray, cls: int) -> np.ndarray:
    return np.asarray(predict_proba(rows), dtype=float)[:, cls]


def deletion_curve(predict_proba, x, ranking, baseline, k_max: int) -> np.ndarray:
    """p(own class) after deleting the top-0, top-1, ..., top-k ranked features."""
    x = np.asarray(x, dtype=float).ravel()
    baseline = np.asarray(baseline, dtype=float).ravel()
    order = np.asarray(ranking, dtype=int).ravel()
    k_max = int(min(k_max, order.size))
    cls = _target_class(predict_proba, x)
    rows = np.tile(x, (k_max + 1, 1))
    for k in range(1, k_max + 1):
        rows[k:, order[k - 1]] = baseline[order[k - 1]]
    return _curve(predict_proba, rows, cls)


def insertion_curve(predict_proba, x, ranking, baseline, k_max: int) -> np.ndarray:
    """p(own class) as the top-k features are restored onto a fully-baseline row."""
    x = np.asarray(x, dtype=float).ravel()
    baseline = np.asarray(baseline, dtype=float).ravel()
    order = np.asarray(ranking, dtype=int).ravel()
    k_max = int(min(k_max, order.size))
    cls = _target_class(predict_proba, x)
    rows = np.tile(baseline, (k_max + 1, 1))
    for k in range(1, k_max + 1):
        rows[k:, order[k - 1]] = x[order[k - 1]]
    return _curve(predict_proba, rows, cls)


def aopc_deletion(predict_proba, x, ranking, baseline, k_max: int) -> float:
    """Mean probability drop over k = 1..K. Higher means a more faithful ranking."""
    curve = deletion_curve(predict_proba, x, ranking, baseline, k_max)
    return float(np.mean(curve[0] - curve[1:])) if curve.size > 1 else 0.0


def auc_insertion(predict_proba, x, ranking, baseline, k_max: int) -> float:
    """Normalised area under the insertion curve. Higher is more faithful."""
    curve = insertion_curve(predict_proba, x, ranking, baseline, k_max)
    return float(np.trapezoid(curve, dx=1.0) / max(len(curve) - 1, 1))


def comprehensiveness(predict_proba, x, ranking, baseline, k: int = 5) -> float:
    """p(own class) lost by removing the top-k (DeYoung et al., 2020). Higher is better."""
    curve = deletion_curve(predict_proba, x, ranking, baseline, k)
    return float(curve[0] - curve[-1])


def sufficiency(predict_proba, x, ranking, baseline, k: int = 5) -> float:
    """p(own class) lost by keeping ONLY the top-k. Nearer zero is better."""
    curve = insertion_curve(predict_proba, x, ranking, baseline, k)
    full = _curve(predict_proba, np.atleast_2d(np.asarray(x, dtype=float)),
                  _target_class(predict_proba, x))[0]
    return float(full - curve[-1])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_fidelity.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/xaibench/metrics/fidelity.py tests/test_fidelity.py
git commit -m "feat: deletion/insertion fidelity metrics with a known-answer test

On a synthetic model f(x)=sigmoid(3x0+2x1+0x2) the true importance order must
maximise AOPC over all six permutations; that one assertion catches sign errors,
off-by-one deletion indexing, and reading the wrong class column. Curves use a
single batched predict_proba call because the study evaluates ~1.7M of them."
```

---

### Task 7: Stability metrics

**Files:**
- Create: `src/xaibench/metrics/stability.py`
- Create: `tests/test_stability.py`

**Interfaces:**
- Consumes: `agreement.ranking_from_attributions`, `agreement.spearman`, `agreement.rank_vector`.
- Produces: `rerun_cv(repeats: np.ndarray) -> float` where `repeats` is `(R, d)`; `repeat_pair_rho(repeats) -> float` (mean Spearman over all `R(R-1)/2` pairs of derived rank vectors); `local_lipschitz(phi_x, x, phi_neighbours, neighbours) -> float`; `sample_neighbours(x, sigma_vector, n, seed, fixed_mask=None) -> np.ndarray`; `retrain_agreement(global_attr_per_refit: np.ndarray) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stability.py
import numpy as np
import pytest
from xaibench.metrics.stability import (
    rerun_cv, repeat_pair_rho, local_lipschitz, sample_neighbours, retrain_agreement,
)

def test_a_deterministic_explainer_scores_exactly_zero_variance():
    # This is the protocol's a-priori expectation for tree SHAP, asserted here so the
    # claim is a tested property of the metric rather than a hope about the backend.
    repeats = np.tile(np.array([0.4, -0.2, 0.1]), (10, 1))
    assert rerun_cv(repeats) == pytest.approx(0.0)
    assert repeat_pair_rho(repeats) == pytest.approx(1.0)

def test_rerun_cv_is_positive_when_repeats_differ():
    rng = np.random.default_rng(0)
    repeats = np.array([0.4, -0.2, 0.1]) + rng.normal(scale=0.05, size=(10, 3))
    assert rerun_cv(repeats) > 0.0

def test_rerun_cv_grows_with_noise():
    rng = np.random.default_rng(1)
    base = np.array([0.4, -0.2, 0.1])
    low = base + rng.normal(scale=0.01, size=(20, 3))
    high = base + rng.normal(scale=0.20, size=(20, 3))
    assert rerun_cv(low) < rerun_cv(high)

def test_rerun_cv_survives_a_zero_mean_feature_without_dividing_by_zero():
    repeats = np.zeros((5, 3)); repeats[:, 0] = [1.0, -1.0, 1.0, -1.0, 1.0]
    assert np.isfinite(rerun_cv(repeats))

def test_repeat_pair_rho_falls_when_orderings_disagree():
    stable = np.tile([3.0, 2.0, 1.0], (6, 1))
    shuffled = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]] * 3)
    assert repeat_pair_rho(shuffled) < repeat_pair_rho(stable)

def test_lipschitz_is_zero_for_a_locally_constant_explainer():
    x = np.array([0.0, 0.0])
    nbrs = np.array([[0.1, 0.0], [0.0, 0.1]])
    phi_x = np.array([1.0, 2.0])
    phi_n = np.tile(phi_x, (2, 1))
    assert local_lipschitz(phi_x, x, phi_n, nbrs) == pytest.approx(0.0)

def test_lipschitz_equals_the_hand_computed_worst_case_ratio():
    x = np.array([0.0, 0.0])
    nbrs = np.array([[0.1, 0.0], [0.0, 0.2]])
    phi_x = np.array([0.0, 0.0])
    phi_n = np.array([[0.05, 0.0], [0.0, 0.02]])
    # ratios: 0.05/0.1 = 0.5 and 0.02/0.2 = 0.1 -> worst case 0.5
    assert local_lipschitz(phi_x, x, phi_n, nbrs) == pytest.approx(0.5)

def test_lipschitz_ignores_a_coincident_neighbour_instead_of_dividing_by_zero():
    x = np.array([1.0, 1.0])
    nbrs = np.array([[1.0, 1.0], [1.1, 1.0]])
    phi_x = np.array([1.0, 0.0])
    phi_n = np.array([[1.0, 0.0], [1.05, 0.0]])
    assert local_lipschitz(phi_x, x, phi_n, nbrs) == pytest.approx(0.5, abs=1e-9)

def test_neighbours_scale_with_sigma_and_hold_fixed_columns():
    x = np.array([0.0, 0.0, 5.0])
    sigma = np.array([1.0, 1.0, 1.0])
    fixed = np.array([False, False, True])
    n = sample_neighbours(x, sigma * 0.1, n=200, seed=0, fixed_mask=fixed)
    assert n.shape == (200, 3)
    assert np.all(n[:, 2] == 5.0)                  # categorical columns must not move
    assert n[:, 0].std() == pytest.approx(0.1, abs=0.03)

def test_neighbours_are_reproducible_from_the_seed():
    x = np.zeros(4); sigma = np.ones(4)
    np.testing.assert_allclose(sample_neighbours(x, sigma, 10, seed=3),
                               sample_neighbours(x, sigma, 10, seed=3))

def test_retrain_agreement_is_one_for_identical_refits_and_lower_when_they_differ():
    same = np.tile([0.5, 0.3, 0.1], (5, 1))
    assert retrain_agreement(same) == pytest.approx(1.0)
    differ = np.array([[0.5, 0.3, 0.1], [0.1, 0.3, 0.5], [0.5, 0.3, 0.1], [0.1, 0.3, 0.5]])
    assert retrain_agreement(differ) < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_stability.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.metrics.stability'`

- [ ] **Step 3: Write `src/xaibench/metrics/stability.py`**

```python
"""Stability across three separable sources: explainer randomness, input
perturbation, and model retraining. Conflating them is what single-notion
stability studies get wrong, and only the first is degenerate for tree SHAP."""
from __future__ import annotations
from itertools import combinations
import numpy as np
from xaibench.metrics.agreement import ranking_from_attributions, rank_vector, spearman

EPS = 1e-12


def rerun_cv(repeats: np.ndarray) -> float:
    """Mean per-feature coefficient of variation across R repeats of one instance.

    Unit of analysis for H2a. Exactly 0.0 for a deterministic explainer.
    """
    r = np.atleast_2d(np.asarray(repeats, dtype=float))
    if r.shape[0] < 2:
        return 0.0
    sd = r.std(axis=0, ddof=1)
    mu = np.abs(r.mean(axis=0))
    return float(np.mean(sd / (mu + EPS)))


def repeat_pair_rho(repeats: np.ndarray) -> float:
    """Mean Spearman rho over all R(R-1)/2 pairs of rank vectors derived from repeats."""
    r = np.atleast_2d(np.asarray(repeats, dtype=float))
    if r.shape[0] < 2:
        return 1.0
    vectors = [rank_vector(ranking_from_attributions(row)) for row in r]
    pairs = [spearman(a, b) for a, b in combinations(vectors, 2)]
    return float(np.mean(pairs)) if pairs else 1.0


def sample_neighbours(x: np.ndarray, sigma: np.ndarray, n: int, seed: int,
                      fixed_mask: np.ndarray | None = None) -> np.ndarray:
    """Gaussian neighbours of x. Columns flagged in fixed_mask are held constant,
    because jittering a one-hot column produces an input the model never sees."""
    x = np.asarray(x, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=1.0, size=(n, x.size)) * sigma
    if fixed_mask is not None:
        noise[:, np.asarray(fixed_mask, dtype=bool)] = 0.0
    return x + noise


def local_lipschitz(phi_x: np.ndarray, x: np.ndarray,
                    phi_neighbours: np.ndarray, neighbours: np.ndarray) -> float:
    """max over neighbours of ||phi(x)-phi(x')||_2 / ||x-x'||_2.

    The estimator of Alvarez-Melis & Jaakkola (2018). Lower is more robust. A
    neighbour that coincides with x is skipped rather than dividing by zero.
    """
    phi_x = np.asarray(phi_x, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    phi_n = np.atleast_2d(np.asarray(phi_neighbours, dtype=float))
    nbrs = np.atleast_2d(np.asarray(neighbours, dtype=float))
    input_dist = np.linalg.norm(nbrs - x, axis=1)
    attr_dist = np.linalg.norm(phi_n - phi_x, axis=1)
    usable = input_dist > 1e-12
    return float(np.max(attr_dist[usable] / input_dist[usable])) if usable.any() else 0.0


def retrain_agreement(global_attr_per_refit: np.ndarray) -> float:
    """Mean pairwise Spearman rho between global rankings from B model refits.

    Separates model variance from explainer variance: an explanation can be
    perfectly reproducible (rerun_cv == 0) and still worthless if a marginally
    different training sample reorders its top features.
    """
    return repeat_pair_rho(global_attr_per_refit)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_stability.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/xaibench/metrics/stability.py tests/test_stability.py
git commit -m "feat: three-way stability metrics (rerun, perturbation, retraining)

test_a_deterministic_explainer_scores_exactly_zero_variance encodes the
protocol's a-priori expectation for tree SHAP as a tested property of the metric.
Lipschitz skips coincident neighbours; neighbour sampling holds one-hot columns
fixed because jittering them produces inputs the model never sees."
```

---

### Task 8: Statistical testing utilities

**Files:**
- Create: `src/xaibench/stats/tests.py`
- Create: `tests/test_stats.py`

**Interfaces:**
- Consumes: `config.ALPHA`.
- Produces: `TestResult` frozen dataclass with fields `n: int`, `statistic: float`, `p_value: float`, `effect_size: float`, `effect_name: str`, `median_diff: float`, `degenerate: bool`; `wilcoxon_with_effect(a, b=None, alternative="two-sided", mu=0.0) -> TestResult`; `holm(p_values: list[float]) -> np.ndarray`; `bca_ci(data, statistic, n_resamples, seed, confidence=0.95) -> tuple[float, float]`; `cliffs_delta(a, b) -> float`; `mannwhitney_with_effect(a, b, alternative="two-sided") -> TestResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stats.py
import numpy as np
import pytest
from xaibench.stats.tests import (
    wilcoxon_with_effect, holm, bca_ci, cliffs_delta, mannwhitney_with_effect,
)

def test_all_positive_differences_give_effect_size_plus_one():
    r = wilcoxon_with_effect(np.arange(1.0, 11.0))
    assert r.n == 10
    assert r.effect_size == pytest.approx(1.0)
    assert r.effect_name == "rank_biserial"
    assert r.p_value < 0.05

def test_all_negative_differences_give_effect_size_minus_one():
    assert wilcoxon_with_effect(-np.arange(1.0, 11.0)).effect_size == pytest.approx(-1.0)

def test_paired_form_equals_the_difference_form():
    a = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    b = np.array([1.0, 3.0, 2.0, 7.0, 4.0])
    assert (wilcoxon_with_effect(a, b).p_value
            == pytest.approx(wilcoxon_with_effect(a - b).p_value))

def test_mu_shifts_the_null_so_a_threshold_hypothesis_can_be_tested():
    # H1 tests rho > 0.50, i.e. Wilcoxon on (rho_i - 0.50).
    rho = np.array([0.55, 0.60, 0.58, 0.62, 0.57, 0.61])
    assert wilcoxon_with_effect(rho, mu=0.50, alternative="greater").p_value < 0.05
    assert wilcoxon_with_effect(rho, mu=0.90, alternative="greater").p_value > 0.5

def test_all_zero_differences_are_reported_degenerate_not_forced_through_a_test():
    # Expected for tree SHAP determinism. scipy raises here; we must not.
    r = wilcoxon_with_effect(np.zeros(8))
    assert r.degenerate is True
    assert r.p_value == 1.0
    assert r.effect_size == 0.0

def test_holm_matches_a_hand_computed_example():
    # p = [0.01, 0.04, 0.03], n = 3, sorted: 0.01(x3), 0.03(x2), 0.04(x1)
    # -> 0.03, 0.06, 0.06 with monotonicity enforced
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

def test_holm_is_monotone_and_never_exceeds_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=25)
    adj = holm(p)
    assert np.all(adj <= 1.0)
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)

def test_holm_is_never_more_lenient_than_the_raw_p_value():
    p = [0.001, 0.02, 0.5]
    assert np.all(holm(p) >= np.asarray(p) - 1e-12)

def test_holm_of_a_single_test_is_the_identity():
    np.testing.assert_allclose(holm([0.023]), [0.023])

def test_bca_ci_brackets_a_known_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=1.0, size=200)
    lo, hi = bca_ci(data, np.mean, n_resamples=800, seed=1)
    assert lo < 5.0 < hi
    assert hi - lo < 1.0

def test_bca_ci_is_reproducible_from_the_seed():
    data = np.random.default_rng(2).normal(size=80)
    assert bca_ci(data, np.mean, 400, seed=5) == bca_ci(data, np.mean, 400, seed=5)

def test_bca_ci_degenerates_gracefully_on_constant_data():
    lo, hi = bca_ci(np.full(30, 2.5), np.mean, 300, seed=0)
    assert lo == pytest.approx(2.5) and hi == pytest.approx(2.5)

def test_cliffs_delta_is_plus_one_when_a_strictly_dominates_b():
    assert cliffs_delta([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert cliffs_delta([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(-1.0)

def test_cliffs_delta_is_zero_for_identical_samples():
    x = [1.0, 2.0, 3.0, 4.0]
    assert cliffs_delta(x, x) == pytest.approx(0.0)

def test_mannwhitney_reports_cliffs_delta_as_its_effect_size():
    r = mannwhitney_with_effect([4.0, 5.0, 6.0, 7.0], [1.0, 2.0, 3.0, 3.5])
    assert r.effect_name == "cliffs_delta"
    assert r.effect_size == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.stats.tests'`

- [ ] **Step 3: Write `src/xaibench/stats/tests.py`**

`wilcoxon_with_effect` must not raise on all-zero differences: that case is *expected* for tree SHAP's determinism, and scipy raises on it. Returning a `degenerate=True` result keeps the results table complete and honest instead of dropping cells.

```python
"""Non-parametric tests with effect sizes, Holm correction, and BCa intervals."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


@dataclass(frozen=True)
class TestResult:
    n: int
    statistic: float
    p_value: float
    effect_size: float
    effect_name: str
    median_diff: float
    degenerate: bool = False


def wilcoxon_with_effect(a, b=None, alternative: str = "two-sided",
                         mu: float = 0.0) -> TestResult:
    """Wilcoxon signed-rank with the rank-biserial correlation (Kerby, 2014).

    `mu` shifts the null so a threshold hypothesis can be tested directly:
    H1 in the protocol is rho > 0.50, which is this test on (rho_i - 0.50).
    """
    a = np.asarray(a, dtype=float).ravel()
    d = a - (np.asarray(b, dtype=float).ravel() if b is not None else 0.0) - mu
    d = d[np.isfinite(d)]
    nonzero = d[d != 0.0]

    if nonzero.size == 0:
        return TestResult(n=int(d.size), statistic=0.0, p_value=1.0, effect_size=0.0,
                          effect_name="rank_biserial", median_diff=0.0, degenerate=True)

    res = stats.wilcoxon(nonzero, alternative=alternative, zero_method="wilcox",
                         method="auto")
    ranks = stats.rankdata(np.abs(nonzero))
    w_pos = float(ranks[nonzero > 0].sum())
    w_neg = float(ranks[nonzero < 0].sum())
    total = w_pos + w_neg
    effect = (w_pos - w_neg) / total if total > 0 else 0.0
    return TestResult(n=int(d.size), statistic=float(res.statistic),
                      p_value=float(res.pvalue), effect_size=float(effect),
                      effect_name="rank_biserial", median_diff=float(np.median(d)))


def cliffs_delta(a, b) -> float:
    """(#a>b - #a<b) / (n_a * n_b). Nonparametric, in [-1, 1]."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    diff = np.sign(a[:, None] - b[None, :])
    return float(diff.sum() / (a.size * b.size))


def mannwhitney_with_effect(a, b, alternative: str = "two-sided") -> TestResult:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0 or (np.ptp(np.concatenate([a, b])) == 0):
        return TestResult(n=int(a.size + b.size), statistic=0.0, p_value=1.0,
                          effect_size=0.0, effect_name="cliffs_delta",
                          median_diff=0.0, degenerate=True)
    res = stats.mannwhitneyu(a, b, alternative=alternative)
    return TestResult(n=int(a.size + b.size), statistic=float(res.statistic),
                      p_value=float(res.pvalue), effect_size=cliffs_delta(a, b),
                      effect_name="cliffs_delta",
                      median_diff=float(np.median(a) - np.median(b)))


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0:
        return p
    if p.size == 1:
        return p.copy()
    return multipletests(p, method="holm")[1]


def bca_ci(data, statistic, n_resamples: int, seed: int,
           confidence: float = 0.95) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap interval.

    Falls back to the percentile interval when the jackknife acceleration is
    undefined (constant data, or a statistic invariant to leaving one point out).
    """
    x = np.asarray(data, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"))
    theta_hat = float(statistic(x))
    if n < 3 or np.ptp(x) == 0:
        return (theta_hat, theta_hat)

    rng = np.random.default_rng(seed)
    boots = np.array([float(statistic(x[rng.integers(0, n, n)])) for _ in range(n_resamples)])
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return (theta_hat, theta_hat)

    alpha = 1.0 - confidence
    prop = float(np.mean(boots < theta_hat))
    prop = min(max(prop, 1.0 / (2 * boots.size)), 1.0 - 1.0 / (2 * boots.size))
    z0 = stats.norm.ppf(prop)

    jack = np.array([float(statistic(np.delete(x, i))) for i in range(n)])
    dev = jack.mean() - jack
    denom = 6.0 * (np.sum(dev ** 2) ** 1.5)
    acc = float(np.sum(dev ** 3) / denom) if denom > 0 else 0.0

    def adjusted(z_alpha: float) -> float:
        num = z0 + z_alpha
        return float(stats.norm.cdf(z0 + num / (1.0 - acc * num)))

    lo_q = adjusted(stats.norm.ppf(alpha / 2.0))
    hi_q = adjusted(stats.norm.ppf(1.0 - alpha / 2.0))
    if not (np.isfinite(lo_q) and np.isfinite(hi_q)) or lo_q >= hi_q:
        lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    return (float(np.quantile(boots, lo_q)), float(np.quantile(boots, hi_q)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_stats.py -v`
Expected: 15 passed

- [ ] **Step 5: Run the whole suite so far**

Run: `.venv/bin/pytest -q`
Expected: all passing, ~80 tests.

- [ ] **Step 6: Commit**

```bash
git add src/xaibench/stats/tests.py tests/test_stats.py
git commit -m "feat: Wilcoxon with rank-biserial effect, Holm, BCa bootstrap, Cliff's delta

wilcoxon_with_effect returns degenerate=True on all-zero differences instead of
raising, because that case is expected for tree SHAP determinism and dropping
those cells would leave holes in the results table. The mu parameter lets H1 test
rho > 0.50 directly rather than against a straw-man null of zero."
```

---

## Phase 2 — Explainers

All explainers share one output contract, so metrics are written once and adding a seventh explainer touches exactly one file. Preprocessing is fitted on the train partition and then frozen; both SHAP and LIME operate on the **transformed** space and their attributions are aggregated back to original features. That keeps the two explainers on identical footing — the alternative, wrapping the whole pipeline, gives LIME a categorical space it cannot sample sensibly.

### Task 9: Explanation contract and cell builder

**Files:**
- Create: `src/xaibench/explain/base.py`, `src/xaibench/models/cells.py`
- Create: `tests/test_cells.py`

**Interfaces:**
- Consumes: `Dataset` (Task 2), `build_preprocessor`/`onehot_group_map`/`categorical_output_indices` (Task 3), `build_model` (Task 4).
- Produces: `Explanation` frozen dataclass with fields `dataset: str`, `model: str`, `explainer: str`, `instance_id: int`, `attributions: np.ndarray` (shape `(d,)`, original-feature space), `feature_names: tuple[str, ...]`, `runtime_s: float`, `seed: int`, `config_hash: str`; `Explanation.ranking` property returning `np.ndarray`; `explanations_to_frame(list[Explanation]) -> pd.DataFrame` (long format, one row per explanation × feature, columns `dataset, model, explainer, instance_id, feature, attribution, rank, runtime_s, seed, config_hash`); `config_hash(**kwargs) -> str`.
- Produces: `Cell` frozen dataclass with fields `dataset: str`, `model: str`, `preprocessor`, `classifier`, `X_train_t: np.ndarray`, `X_test_t: np.ndarray`, `y_test: np.ndarray`, `groups: list[str]`, `feature_order: list[str]`, `categorical_idx: list[int]`, `instance_ids: np.ndarray`, `train_index: np.ndarray`, `X_train_raw: pd.DataFrame`, `y_train_raw: pd.Series`; `Cell.predict_proba` bound to the classifier on transformed input; `build_cell(dataset_id, model_id, params, profile, seed=SEED) -> Cell`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cells.py
import numpy as np
import pytest
from xaibench.config import QUICK
from xaibench.explain.base import Explanation, explanations_to_frame, config_hash
from xaibench.models.cells import build_cell

def _explanation(**kw):
    base = dict(dataset="pima", model="rf", explainer="shap", instance_id=0,
                attributions=np.array([0.3, -0.7, 0.1]),
                feature_names=("a", "b", "c"), runtime_s=0.01, seed=42,
                config_hash="deadbeef")
    return Explanation(**{**base, **kw})

def test_ranking_orders_by_absolute_attribution():
    np.testing.assert_array_equal(_explanation().ranking, [1, 0, 2])

def test_explanation_rejects_a_length_mismatch_immediately():
    with pytest.raises(ValueError):
        _explanation(attributions=np.array([0.1, 0.2]))

def test_frame_is_long_with_one_row_per_feature():
    df = explanations_to_frame([_explanation(), _explanation(instance_id=1)])
    assert len(df) == 6
    assert set(df.columns) == {"dataset", "model", "explainer", "instance_id",
                               "feature", "attribution", "rank", "runtime_s",
                               "seed", "config_hash"}
    assert df.loc[df.instance_id == 0, "rank"].tolist() == [1, 0, 2]

def test_config_hash_is_stable_and_order_independent():
    assert config_hash(a=1, b="x") == config_hash(b="x", a=1)
    assert config_hash(a=1) != config_hash(a=2)
    assert len(config_hash(a=1)) == 16

@pytest.mark.parametrize("model_id", ["logreg", "rf", "xgb", "lgbm", "stack", "dtree"])
def test_cell_builds_for_every_model_with_a_consistent_contract(model_id):
    cell = build_cell("pima", model_id, params={}, profile=QUICK)
    assert cell.X_train_t.shape[1] == cell.X_test_t.shape[1] == len(cell.groups)
    assert len(cell.feature_order) == 8                       # pima has 8 features
    assert set(cell.groups) == set(cell.feature_order)
    assert cell.instance_ids.size == min(QUICK.n_explain, cell.X_test_t.shape[0])
    p = cell.predict_proba(cell.X_test_t[cell.instance_ids])
    assert p.shape == (cell.instance_ids.size, 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)

def test_test_partition_is_twenty_percent_and_stratified():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    assert cell.X_test_t.shape[0] == 154                      # ceil(768 * 0.2)
    assert abs(cell.y_test.mean() - 0.349) < 0.05

def test_cell_is_reproducible_from_the_seed():
    a = build_cell("heart", "rf", params={}, profile=QUICK)
    b = build_cell("heart", "rf", params={}, profile=QUICK)
    np.testing.assert_allclose(a.X_test_t, b.X_test_t)
    np.testing.assert_array_equal(a.instance_ids, b.instance_ids)
    np.testing.assert_allclose(a.predict_proba(a.X_test_t), b.predict_proba(b.X_test_t))

def test_explained_instances_are_class_stratified():
    cell = build_cell("ckd", "rf", params={}, profile=QUICK)
    chosen = cell.y_test[cell.instance_ids]
    assert 0 in set(chosen) and 1 in set(chosen)

def test_params_are_applied_to_the_classifier():
    cell = build_cell("pima", "rf", params={"max_depth": 3}, profile=QUICK)
    assert cell.classifier.get_params()["max_depth"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cells.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.explain.base'`

- [ ] **Step 3: Write `src/xaibench/explain/base.py`**

```python
"""The single output contract every explainer satisfies."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
import numpy as np
import pandas as pd
from xaibench.metrics.agreement import ranking_from_attributions


def config_hash(**kwargs) -> str:
    """Short stable digest of an explainer configuration, for provenance."""
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Explanation:
    dataset: str
    model: str
    explainer: str
    instance_id: int
    attributions: np.ndarray      # (d,) in ORIGINAL-feature space
    feature_names: tuple[str, ...]
    runtime_s: float
    seed: int
    config_hash: str

    def __post_init__(self) -> None:
        if np.asarray(self.attributions).shape != (len(self.feature_names),):
            raise ValueError(
                f"attributions shape {np.asarray(self.attributions).shape} does not match "
                f"{len(self.feature_names)} feature names; one-hot aggregation was skipped"
            )

    @property
    def ranking(self) -> np.ndarray:
        return ranking_from_attributions(self.attributions)


def explanations_to_frame(explanations: list[Explanation]) -> pd.DataFrame:
    """Long format: one row per (explanation, feature). Parquet-friendly and tidy."""
    rows = []
    for e in explanations:
        rank_of = np.empty(len(e.feature_names), dtype=int)
        rank_of[e.ranking] = np.arange(len(e.feature_names))
        for j, name in enumerate(e.feature_names):
            rows.append({
                "dataset": e.dataset, "model": e.model, "explainer": e.explainer,
                "instance_id": int(e.instance_id), "feature": name,
                "attribution": float(e.attributions[j]), "rank": int(rank_of[j]),
                "runtime_s": float(e.runtime_s), "seed": int(e.seed),
                "config_hash": e.config_hash,
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Write `src/xaibench/models/cells.py`**

```python
"""One fitted (dataset, model) cell: the fixed object every explainer operates on."""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xaibench.config import SEED, TEST_SIZE, Profile
from xaibench.data.loaders import load_dataset
from xaibench.models.pipelines import (
    build_preprocessor, categorical_output_indices, onehot_group_map,
)
from xaibench.models.zoo import build_model


@dataclass(frozen=True)
class Cell:
    dataset: str
    model: str
    preprocessor: object
    classifier: object
    X_train_t: np.ndarray
    X_test_t: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    groups: list[str]
    feature_order: list[str]
    categorical_idx: list[int]
    instance_ids: np.ndarray
    X_train_raw: pd.DataFrame
    y_train_raw: pd.Series

    def predict_proba(self, X_t: np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier.predict_proba(np.atleast_2d(X_t)), dtype=float)

    @property
    def fixed_mask(self) -> np.ndarray:
        """True for output columns that must not be jittered (one-hot columns)."""
        mask = np.zeros(len(self.groups), dtype=bool)
        mask[self.categorical_idx] = True
        return mask

    @property
    def train_sigma(self) -> np.ndarray:
        return self.X_train_t.std(axis=0, ddof=1)


def _stratified_subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Up to n indices, class-proportional, deterministic. Returns sorted indices."""
    n = int(min(n, y.size))
    if n == y.size:
        return np.arange(y.size)
    idx, _ = train_test_split(np.arange(y.size), train_size=n, stratify=y,
                              random_state=seed)
    return np.sort(idx)


def build_cell(dataset_id: str, model_id: str, params: dict, profile: Profile,
               seed: int = SEED) -> Cell:
    """Fit the one model per cell that all explanation analysis runs against.

    Track B of the protocol: a single stratified 80/20 split, hyperparameters taken
    from Track A's inner loop, preprocessing fitted on the train partition only.
    """
    ds = load_dataset(dataset_id)
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=TEST_SIZE, stratify=ds.y, random_state=seed)

    pre = build_preprocessor(ds).fit(X_tr)
    X_tr_t = np.asarray(pre.transform(X_tr), dtype=float)
    X_te_t = np.asarray(pre.transform(X_te), dtype=float)

    clf = build_model(model_id, seed=seed)
    if params:
        clf.set_params(**params)
    clf.fit(X_tr_t, np.asarray(y_tr))

    y_te_arr = np.asarray(y_te)
    return Cell(
        dataset=dataset_id, model=model_id, preprocessor=pre, classifier=clf,
        X_train_t=X_tr_t, X_test_t=X_te_t,
        y_train=np.asarray(y_tr), y_test=y_te_arr,
        groups=onehot_group_map(pre, ds), feature_order=list(ds.X.columns),
        categorical_idx=categorical_output_indices(onehot_group_map(pre, ds), ds),
        instance_ids=_stratified_subsample(y_te_arr, profile.n_explain, seed),
        X_train_raw=X_tr.reset_index(drop=True), y_train_raw=y_tr.reset_index(drop=True),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cells.py -v`
Expected: 14 passed

- [ ] **Step 6: Commit**

```bash
git add src/xaibench/explain/base.py src/xaibench/models/cells.py tests/test_cells.py
git commit -m "feat: Explanation contract and per-cell fitted model builder

Explanation.__post_init__ rejects a length mismatch, which is how a skipped
one-hot aggregation surfaces: it would otherwise silently produce rankings over
transformed columns instead of original features. build_cell implements the
protocol's Track B: one stratified split, preprocessing fitted on train only."
```

---

### Task 10: SHAP backend with an additivity guarantee

**Files:**
- Create: `src/xaibench/explain/shap_backend.py`
- Create: `tests/test_shap_backend.py`

**Interfaces:**
- Consumes: `Cell` (Task 9), `shap_backend_kind` (Task 4), `aggregate_attributions` (Task 3).
- Produces: `explain_shap(cell, instance_ids=None, seed=SEED, background=None) -> list[Explanation]`; `build_background(cell, k: int, seed: int) -> np.ndarray`; `additivity_error(cell, raw_values, base_value, instance_ids) -> float` returning the max absolute discrepancy between `base + Σφ` and `predict_proba(...)[:, 1]`; `SHAP_ADDITIVITY_TOL: float = 5e-3`.

- [ ] **Step 1: Write the failing test**

The additivity check is the load-bearing test in this task. If `base_value + Σφ` reproduces `predict_proba`, the attributions are in probability space by construction — which is exactly the guarantee the protocol requires and which cannot be established by reading the `shap` docs, because the behaviour differs per backend and per version.

```python
# tests/test_shap_backend.py
import numpy as np
import pytest
from xaibench.config import QUICK, MODELS
from xaibench.models.cells import build_cell
from xaibench.explain.shap_backend import (
    explain_shap, build_background, SHAP_ADDITIVITY_TOL,
)

@pytest.mark.parametrize("model_id", MODELS)
def test_attributions_are_in_probability_space_for_every_backend(model_id):
    # The definitive check: base_value + sum(phi) must reproduce predict_proba.
    # Passing this for xgb/lgbm proves model_output='probability' actually took effect.
    cell = build_cell("pima", model_id, params={}, profile=QUICK)
    exps = explain_shap(cell)
    assert len(exps) == cell.instance_ids.size
    for e in exps:
        assert e.attributions.shape == (len(cell.feature_order),)
        assert np.isfinite(e.attributions).all()

@pytest.mark.parametrize("model_id", MODELS)
def test_additivity_holds_within_tolerance(model_id):
    from xaibench.explain.shap_backend import _raw_shap_values
    cell = build_cell("pima", model_id, params={}, profile=QUICK)
    values, base = _raw_shap_values(cell, cell.instance_ids,
                                    build_background(cell, QUICK.shap_background, 42), 42)
    reconstructed = base + values.sum(axis=1)
    actual = cell.predict_proba(cell.X_test_t[cell.instance_ids])[:, 1]
    assert np.max(np.abs(reconstructed - actual)) < SHAP_ADDITIVITY_TOL

def test_tree_backends_are_bit_identical_across_reruns():
    # The protocol declares tree SHAP deterministic a priori; this is that claim
    # under test, and it is what makes H2a degenerate for these four models.
    for model_id in ["dtree", "rf", "xgb", "lgbm"]:
        cell = build_cell("pima", model_id, params={}, profile=QUICK)
        a = explain_shap(cell, seed=1)
        b = explain_shap(cell, seed=999)      # a different seed must change nothing
        for ea, eb in zip(a, b):
            np.testing.assert_array_equal(ea.attributions, eb.attributions)

def test_permutation_backend_is_stochastic_so_h2a_is_not_vacuous_everywhere():
    cell = build_cell("pima", "stack", params={}, profile=QUICK)
    a = explain_shap(cell, seed=1)
    b = explain_shap(cell, seed=2)
    diffs = [np.abs(ea.attributions - eb.attributions).max() for ea, eb in zip(a, b)]
    assert max(diffs) > 0.0

def test_attributions_are_aggregated_to_original_features_on_a_categorical_dataset():
    cell = build_cell("ckd", "rf", params={}, profile=QUICK)
    assert len(cell.groups) > len(cell.feature_order)      # one-hot expanded
    for e in explain_shap(cell):
        assert e.attributions.shape == (len(cell.feature_order),)
        assert e.feature_names == tuple(cell.feature_order)

def test_background_is_summarised_to_the_requested_size():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    bg = build_background(cell, k=15, seed=42)
    assert bg.shape == (15, cell.X_train_t.shape[1])
    np.testing.assert_allclose(bg, build_background(cell, k=15, seed=42))

def test_background_falls_back_when_k_exceeds_the_training_rows():
    cell = build_cell("heart", "rf", params={}, profile=QUICK)
    bg = build_background(cell, k=10_000, seed=42)
    assert bg.shape[0] == cell.X_train_t.shape[0]

def test_runtime_is_recorded_and_positive():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    assert all(e.runtime_s > 0 for e in explain_shap(cell))

def test_explainer_label_and_config_hash_are_set():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    e = explain_shap(cell)[0]
    assert e.explainer == "shap"
    assert len(e.config_hash) == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_shap_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.explain.shap_backend'`

- [ ] **Step 3: Write `src/xaibench/explain/shap_backend.py`**

Binary classification makes the class choice free: `p(class 0) = 1 − p(class 1)`, so `φ₀ = −φ₁` exactly, and a ranking by `|φ|` is identical either way. Class 1 is used throughout and that equivalence is stated in the paper.

```python
"""SHAP attributions in predicted-probability space, per-backend.

Backend assignment (protocol §2.2), and why each is what it is:
  dtree, rf   TreeExplainer, raw output IS the class proportion -> already probability
  xgb, lgbm   TreeExplainer with model_output='probability' + interventional background,
              because their raw output is a log-odds margin
  logreg,     PermutationExplainer over predict_proba: no tree structure to exploit,
  stack       and LinearExplainer would return log-odds, breaking comparability with LIME
"""
from __future__ import annotations
import time
import numpy as np
import shap
from sklearn.cluster import KMeans
from xaibench.config import SEED
from xaibench.explain.base import Explanation, config_hash
from xaibench.models.pipelines import aggregate_attributions
from xaibench.models.zoo import shap_backend_kind

SHAP_ADDITIVITY_TOL = 5e-3


def build_background(cell, k: int, seed: int) -> np.ndarray:
    """k-means centroids of the training rows, used as the interventional reference.

    Centroids rather than a random subsample: with 50 references the variance of a
    random draw is visible in the attributions, and the protocol needs a fixed
    background so tree SHAP stays deterministic.
    """
    X = np.asarray(cell.X_train_t, dtype=float)
    if k >= X.shape[0]:
        return X.copy()
    km = KMeans(n_clusters=int(k), n_init=10, random_state=seed).fit(X)
    return np.asarray(km.cluster_centers_, dtype=float)


def _normalise(values, base) -> tuple[np.ndarray, float]:
    """Reduce any shap return shape to (n, n_out) for class 1 and a scalar base value."""
    v = np.asarray(values, dtype=float)
    if v.ndim == 3:                      # (n, n_out, n_classes)
        v = v[:, :, -1]
    b = np.asarray(base, dtype=float).ravel()
    return v, float(b[-1] if b.size > 1 else b[0])


def _raw_shap_values(cell, instance_ids, background, seed) -> tuple[np.ndarray, float]:
    """Attributions in TRANSFORMED-column space plus the scalar base value."""
    X = np.asarray(cell.X_test_t, dtype=float)[np.asarray(instance_ids, dtype=int)]
    kind = shap_backend_kind(cell.model)

    if kind == "tree":
        ex = shap.TreeExplainer(cell.classifier)
        return _normalise(ex.shap_values(X, check_additivity=False), ex.expected_value)

    if kind == "tree_prob":
        ex = shap.TreeExplainer(cell.classifier, data=background,
                                model_output="probability",
                                feature_perturbation="interventional")
        return _normalise(ex.shap_values(X, check_additivity=False), ex.expected_value)

    masker = shap.maskers.Independent(background, max_samples=background.shape[0])
    ex = shap.PermutationExplainer(cell.classifier.predict_proba, masker, seed=seed)
    out = ex(X, max_evals=2 * X.shape[1] + 1, silent=True)
    return _normalise(out.values, out.base_values)


def additivity_error(cell, values: np.ndarray, base: float, instance_ids) -> float:
    """max |base + sum(phi) - p(class 1)|. The check that proves probability space."""
    actual = cell.predict_proba(
        np.asarray(cell.X_test_t)[np.asarray(instance_ids, dtype=int)])[:, 1]
    return float(np.max(np.abs(base + values.sum(axis=1) - actual)))


def explain_shap(cell, instance_ids=None, seed: int = SEED,
                 background: np.ndarray | None = None) -> list[Explanation]:
    ids = np.asarray(cell.instance_ids if instance_ids is None else instance_ids, dtype=int)
    if background is None:
        background = build_background(cell, k=min(50, cell.X_train_t.shape[0]), seed=SEED)

    cfg = config_hash(explainer="shap", backend=shap_backend_kind(cell.model),
                      background=int(background.shape[0]), space="probability")

    out: list[Explanation] = []
    # Timed per instance so RQ4 measures the per-explanation cost the protocol defines,
    # not an amortised batch cost. Backends that only support batching divide evenly.
    start = time.perf_counter()
    values, _base = _raw_shap_values(cell, ids, background, seed)
    elapsed = time.perf_counter() - start
    per_instance = max(elapsed / max(ids.size, 1), 1e-9)

    agg = aggregate_attributions(values, cell.groups, cell.feature_order)
    for row, inst in enumerate(ids):
        out.append(Explanation(
            dataset=cell.dataset, model=cell.model, explainer="shap",
            instance_id=int(inst), attributions=agg[row],
            feature_names=tuple(cell.feature_order), runtime_s=per_instance,
            seed=int(seed), config_hash=cfg))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_shap_backend.py -v`
Expected: 15 passed.

If `test_additivity_holds_within_tolerance` fails for `xgb` or `lgbm`, protocol risk **R1** has fired: `model_output="probability"` is not taking effect in the installed `shap` version. Do not loosen the tolerance. Record the fallback to log-odds space for that backend in `results/explainer_config.csv`, add an amendment to `docs/PROTOCOL.md` §10, and note in the paper that those cells are compared within log-odds space only.

- [ ] **Step 5: Commit**

```bash
git add src/xaibench/explain/shap_backend.py tests/test_shap_backend.py
git commit -m "feat: SHAP backend with an enforced probability-space guarantee

test_additivity_holds_within_tolerance is the load-bearing test: if base_value +
sum(phi) reproduces predict_proba, the attributions are in probability space by
construction. That cannot be established from the shap docs because the behaviour
differs per backend and per version, and getting it wrong means comparing
log-odds SHAP against probability LIME - a silently invalidating error."
```

---

### Task 11: LIME backend

**Files:**
- Create: `src/xaibench/explain/lime_backend.py`
- Create: `tests/test_lime_backend.py`

**Interfaces:**
- Consumes: `Cell` (Task 9), `aggregate_attributions` (Task 3).
- Produces: `explain_lime(cell, instance_ids=None, num_samples=5000, seed=SEED) -> list[Explanation]`; `build_lime_explainer(cell, seed) -> LimeTabularExplainer`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lime_backend.py
import numpy as np
import pytest
from xaibench.config import QUICK, MODELS
from xaibench.models.cells import build_cell
from xaibench.explain.lime_backend import explain_lime

@pytest.mark.parametrize("model_id", MODELS)
def test_contract_matches_shap_so_metrics_need_no_special_cases(model_id):
    cell = build_cell("pima", model_id, params={}, profile=QUICK)
    exps = explain_lime(cell, num_samples=200, seed=42)
    assert len(exps) == cell.instance_ids.size
    for e in exps:
        assert e.explainer == "lime"
        assert e.attributions.shape == (len(cell.feature_order),)
        assert e.feature_names == tuple(cell.feature_order)
        assert np.isfinite(e.attributions).all()
        assert e.runtime_s > 0

def test_same_seed_reproduces_and_different_seed_does_not():
    # LIME is sampling-based, which is the whole substance of H2a. Both halves of
    # this test matter: reproducible under a fixed seed, variable across seeds.
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    a = explain_lime(cell, num_samples=200, seed=7)
    b = explain_lime(cell, num_samples=200, seed=7)
    c = explain_lime(cell, num_samples=200, seed=8)
    for ea, eb in zip(a, b):
        np.testing.assert_allclose(ea.attributions, eb.attributions)
    assert max(np.abs(ea.attributions - ec.attributions).max()
               for ea, ec in zip(a, c)) > 0.0

def test_variance_shrinks_as_num_samples_grows():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    inst = cell.instance_ids[:2]

    def spread(n):
        runs = np.array([explain_lime(cell, inst, num_samples=n, seed=1000 + r)[0].attributions
                         for r in range(5)])
        return float(runs.std(axis=0).mean())

    assert spread(1000) < spread(100)

def test_attributions_aggregate_to_original_features_on_a_categorical_dataset():
    cell = build_cell("ckd", "rf", params={}, profile=QUICK)
    for e in explain_lime(cell, num_samples=200, seed=42):
        assert e.attributions.shape == (len(cell.feature_order),)

def test_config_hash_distinguishes_num_samples_for_provenance():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    a = explain_lime(cell, cell.instance_ids[:1], num_samples=100, seed=1)[0]
    b = explain_lime(cell, cell.instance_ids[:1], num_samples=200, seed=1)[0]
    assert a.config_hash != b.config_hash

def test_explaining_a_subset_returns_only_that_subset():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    subset = cell.instance_ids[:3]
    exps = explain_lime(cell, subset, num_samples=100, seed=1)
    assert [e.instance_id for e in exps] == [int(i) for i in subset]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_lime_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.explain.lime_backend'`

- [ ] **Step 3: Write `src/xaibench/explain/lime_backend.py`**

```python
"""LIME attributions, returned through the same contract as SHAP."""
from __future__ import annotations
import time
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from xaibench.config import SEED
from xaibench.explain.base import Explanation, config_hash
from xaibench.models.pipelines import aggregate_attributions

DISCRETIZE_CONTINUOUS = True     # NORMATIVE primary (protocol §2.3); False is the sensitivity variant


def build_lime_explainer(cell, seed: int, discretize: bool = DISCRETIZE_CONTINUOUS):
    """One explainer per cell, built on the transformed training rows.

    categorical_features is passed so LIME does not discretise one-hot columns into
    bins, which would produce neighbourhood samples the model never sees.
    """
    return LimeTabularExplainer(
        training_data=np.asarray(cell.X_train_t, dtype=float),
        feature_names=[f"c{i}" for i in range(cell.X_train_t.shape[1])],
        class_names=["negative", "positive"],
        categorical_features=list(cell.categorical_idx),
        discretize_continuous=discretize,
        mode="classification",
        random_state=int(seed),
    )


def explain_lime(cell, instance_ids=None, num_samples: int = 5000,
                 seed: int = SEED, discretize: bool = DISCRETIZE_CONTINUOUS
                 ) -> list[Explanation]:
    ids = np.asarray(cell.instance_ids if instance_ids is None else instance_ids, dtype=int)
    n_out = cell.X_train_t.shape[1]
    explainer = build_lime_explainer(cell, seed, discretize)
    cfg = config_hash(explainer="lime", num_samples=int(num_samples),
                      discretize=bool(discretize), space="probability")

    out: list[Explanation] = []
    for inst in ids:
        x = np.asarray(cell.X_test_t, dtype=float)[int(inst)]
        start = time.perf_counter()
        exp = explainer.explain_instance(
            data_row=x, predict_fn=cell.classifier.predict_proba,
            num_features=n_out, num_samples=int(num_samples), labels=(1,))
        elapsed = time.perf_counter() - start

        raw = np.zeros(n_out, dtype=float)
        for col, weight in exp.as_map()[1]:
            raw[int(col)] = float(weight)
        agg = aggregate_attributions(raw[None, :], cell.groups, cell.feature_order)[0]

        out.append(Explanation(
            dataset=cell.dataset, model=cell.model, explainer="lime",
            instance_id=int(inst), attributions=agg,
            feature_names=tuple(cell.feature_order), runtime_s=float(elapsed),
            seed=int(seed), config_hash=cfg))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_lime_backend.py -v`
Expected: 11 passed. `test_variance_shrinks_as_num_samples_grows` is stochastic by nature; if it fails, raise the repeat count from 5 to 10 rather than weakening the assertion — the property is real and is the basis of figure F7.

- [ ] **Step 5: Commit**

```bash
git add src/xaibench/explain/lime_backend.py tests/test_lime_backend.py
git commit -m "feat: LIME backend on the shared Explanation contract

One-hot columns are declared categorical so LIME does not bin them into
neighbourhood samples the model never sees. Tests pin both halves of LIME's
stochasticity: reproducible under a fixed seed, variable across seeds, with
variance shrinking as num_samples grows - the property figure F7 reports."
```

---

### Task 12: Consensus rules and the random control

**Files:**
- Create: `src/xaibench/explain/consensus.py`
- Create: `tests/test_consensus.py`

**Interfaces:**
- Consumes: `Explanation` (Task 9), `config.RRF_K`, `config.CALIB_FRACTION`, `config.RANDOM_CONTROL_DRAWS`.
- Produces: `normalised_ranks(attr: np.ndarray) -> np.ndarray` in `[0, 1]` with 0 = most important; `fuse_rankavg(a, b) -> np.ndarray`; `fuse_rrf(a, b, k=RRF_K) -> np.ndarray`; `fuse_weighted(a, b, w: float) -> np.ndarray`; `calibration_split(instance_ids, seed, fraction=CALIB_FRACTION) -> tuple[np.ndarray, np.ndarray]`; `fidelity_weight(aopc_shap: float, aopc_lime: float) -> float`; `build_consensus(shap_exps, lime_exps, weight, rule) -> list[Explanation]`; `random_control(cell, instance_ids, seed, n_draws=RANDOM_CONTROL_DRAWS) -> list[Explanation]`. All `fuse_*` return **pseudo-attributions**: `1 − fused_normalised_rank`, so downstream ranking by `|value|` reproduces the fused order and the shared contract needs no special case.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_consensus.py
import numpy as np
import pytest
from xaibench.config import QUICK, CALIB_FRACTION
from xaibench.models.cells import build_cell
from xaibench.explain.base import Explanation
from xaibench.metrics.agreement import ranking_from_attributions
from xaibench.explain.consensus import (
    normalised_ranks, fuse_rankavg, fuse_rrf, fuse_weighted, calibration_split,
    fidelity_weight, build_consensus, random_control,
)

A = np.array([0.9, -0.5, 0.1, 0.05])     # order: 0, 1, 2, 3
B = np.array([0.1, -0.8, 0.7, 0.02])     # order: 1, 2, 0, 3

def test_normalised_ranks_put_the_most_important_feature_at_zero():
    r = normalised_ranks(A)
    assert r[0] == pytest.approx(0.0)
    assert r[3] == pytest.approx(1.0)
    assert np.all((r >= 0) & (r <= 1))

def test_fusing_a_ranking_with_itself_is_the_identity():
    for fuse in (fuse_rankavg, fuse_rrf):
        np.testing.assert_array_equal(ranking_from_attributions(fuse(A, A)),
                                      ranking_from_attributions(A))

def test_extreme_weights_recover_each_base_explainer_exactly():
    np.testing.assert_array_equal(ranking_from_attributions(fuse_weighted(A, B, w=1.0)),
                                  ranking_from_attributions(A))
    np.testing.assert_array_equal(ranking_from_attributions(fuse_weighted(A, B, w=0.0)),
                                  ranking_from_attributions(B))

def test_equal_weight_matches_plain_rank_averaging():
    np.testing.assert_allclose(fuse_weighted(A, B, w=0.5), fuse_rankavg(A, B))

def test_fusion_lands_between_the_two_inputs_on_a_disputed_feature():
    # Feature 2 is 3rd for A and 2nd for B, so fused must not be worse than both.
    fused = ranking_from_attributions(fuse_rankavg(A, B)).tolist()
    assert fused.index(2) <= max(ranking_from_attributions(A).tolist().index(2),
                                 ranking_from_attributions(B).tolist().index(2))

def test_rrf_is_more_tolerant_of_one_bad_ranking_than_mean_rank():
    # A feature ranked 1st by one method and last by the other: RRF's 1/(k+r) keeps
    # it higher than the arithmetic mean of ranks does. This is why both are tested.
    a = np.array([1.0, 0.1, 0.05, 0.01])
    b = np.array([0.01, 0.9, 0.5, 0.2])
    pos_rrf = ranking_from_attributions(fuse_rrf(a, b)).tolist().index(0)
    pos_avg = ranking_from_attributions(fuse_rankavg(a, b)).tolist().index(0)
    assert pos_rrf <= pos_avg

def test_calibration_and_evaluation_sets_are_disjoint_and_exhaustive():
    ids = np.arange(100)
    calib, evalu = calibration_split(ids, seed=42)
    assert set(calib) | set(evalu) == set(ids)
    assert not (set(calib) & set(evalu))
    assert len(calib) == pytest.approx(100 * CALIB_FRACTION, abs=1)

def test_calibration_split_is_reproducible():
    a = calibration_split(np.arange(50), seed=1)
    b = calibration_split(np.arange(50), seed=1)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])

def test_fidelity_weight_is_a_proportion_and_handles_degenerate_input():
    assert fidelity_weight(0.3, 0.1) == pytest.approx(0.75)
    assert fidelity_weight(0.0, 0.0) == pytest.approx(0.5)     # no signal -> unweighted
    assert 0.0 <= fidelity_weight(-0.2, 0.4) <= 1.0            # clipped, not negative

def test_build_consensus_preserves_instance_alignment():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    names = tuple(cell.feature_order)
    d = len(names)
    rng = np.random.default_rng(0)
    shap_exps = [Explanation("pima", "rf", "shap", int(i), rng.normal(size=d), names,
                             0.01, 42, "h1") for i in cell.instance_ids]
    lime_exps = [Explanation("pima", "rf", "lime", int(i), rng.normal(size=d), names,
                             0.02, 42, "h2") for i in cell.instance_ids]
    fused = build_consensus(shap_exps, lime_exps, weight=0.6, rule="fidwt")
    assert [e.instance_id for e in fused] == [e.instance_id for e in shap_exps]
    assert all(e.explainer == "fidwt" for e in fused)
    # Cost is the sum of both base explainers: fusion is not free.
    assert fused[0].runtime_s == pytest.approx(0.03)

def test_build_consensus_rejects_misaligned_inputs():
    names = ("a", "b")
    s = [Explanation("d", "m", "shap", 0, np.array([1.0, 0.0]), names, 0.0, 1, "h")]
    l = [Explanation("d", "m", "lime", 9, np.array([0.0, 1.0]), names, 0.0, 1, "h")]
    with pytest.raises(ValueError):
        build_consensus(s, l, weight=0.5, rule="rankavg")

def test_random_control_is_reproducible_and_uninformative():
    cell = build_cell("pima", "rf", params={}, profile=QUICK)
    a = random_control(cell, cell.instance_ids, seed=42)
    b = random_control(cell, cell.instance_ids, seed=42)
    for ea, eb in zip(a, b):
        np.testing.assert_allclose(ea.attributions, eb.attributions)
    assert all(e.explainer == "random" for e in a)
    # Across instances the rankings must differ, or it is not a random control.
    orders = {tuple(e.ranking) for e in a}
    assert len(orders) > 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_consensus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.explain.consensus'`

- [ ] **Step 3: Write `src/xaibench/explain/consensus.py`**

```python
"""Three rank-fusion rules plus the random control, on the shared contract.

Explanation aggregation is not new; the contribution is a controlled, leakage-safe
evaluation of these rules against a strict bar (beat max(SHAP, LIME), not their mean).
"""
from __future__ import annotations
import numpy as np
from xaibench.config import CALIB_FRACTION, RANDOM_CONTROL_DRAWS, RRF_K, SEED
from xaibench.explain.base import Explanation, config_hash
from xaibench.metrics.agreement import ranking_from_attributions


def normalised_ranks(attr: np.ndarray) -> np.ndarray:
    """Position of each feature in [0, 1]; 0 = most important."""
    order = ranking_from_attributions(attr)
    d = order.size
    out = np.empty(d, dtype=float)
    out[order] = np.arange(d, dtype=float)
    return out / max(d - 1, 1)


def _as_pseudo_attributions(fused_rank: np.ndarray) -> np.ndarray:
    """1 - normalised rank, so ranking by |value| reproduces the fused order."""
    return 1.0 - np.asarray(fused_rank, dtype=float)


def fuse_rankavg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _as_pseudo_attributions(0.5 * (normalised_ranks(a) + normalised_ranks(b)))


def fuse_weighted(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    w = float(np.clip(w, 0.0, 1.0))
    return _as_pseudo_attributions(w * normalised_ranks(a) + (1.0 - w) * normalised_ranks(b))


def fuse_rrf(a: np.ndarray, b: np.ndarray, k: int = RRF_K) -> np.ndarray:
    """Reciprocal rank fusion. Bounded contribution per list, so one badly wrong
    ranking degrades the fusion less than arithmetic rank averaging does."""
    d = np.asarray(a).size
    ra = normalised_ranks(a) * max(d - 1, 1)
    rb = normalised_ranks(b) * max(d - 1, 1)
    score = 1.0 / (k + 1.0 + ra) + 1.0 / (k + 1.0 + rb)
    order = np.lexsort((np.arange(d), -score))
    fused = np.empty(d, dtype=float)
    fused[order] = np.arange(d, dtype=float) / max(d - 1, 1)
    return _as_pseudo_attributions(fused)


def calibration_split(instance_ids, seed: int,
                      fraction: float = CALIB_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint calibration / evaluation instance sets.

    The fidelity weight is estimated on the calibration set only. Without this
    split, the weight would be tuned on the same instances H6 is tested on, and
    the consensus rule would win by construction.
    """
    ids = np.asarray(instance_ids, dtype=int)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    n_calib = max(1, int(round(fraction * ids.size))) if ids.size > 1 else 0
    return np.sort(shuffled[:n_calib]), np.sort(shuffled[n_calib:])


def fidelity_weight(aopc_shap: float, aopc_lime: float) -> float:
    """Share of weight given to SHAP, from calibration-set fidelity."""
    a, b = max(float(aopc_shap), 0.0), max(float(aopc_lime), 0.0)
    return 0.5 if (a + b) <= 0.0 else float(a / (a + b))


_RULES = {"rankavg": lambda a, b, w: fuse_rankavg(a, b),
          "rrf": lambda a, b, w: fuse_rrf(a, b),
          "fidwt": lambda a, b, w: fuse_weighted(a, b, w)}


def build_consensus(shap_exps: list[Explanation], lime_exps: list[Explanation],
                    weight: float, rule: str) -> list[Explanation]:
    if rule not in _RULES:
        raise ValueError(f"unknown fusion rule {rule!r}; expected one of {sorted(_RULES)}")
    if len(shap_exps) != len(lime_exps):
        raise ValueError("consensus needs one SHAP and one LIME explanation per instance")

    cfg = config_hash(explainer=rule, weight=round(float(weight), 6))
    out: list[Explanation] = []
    for s, l in zip(shap_exps, lime_exps):
        if s.instance_id != l.instance_id:
            raise ValueError(
                f"instance misalignment: shap={s.instance_id} lime={l.instance_id}")
        out.append(Explanation(
            dataset=s.dataset, model=s.model, explainer=rule, instance_id=s.instance_id,
            attributions=_RULES[rule](s.attributions, l.attributions, weight),
            feature_names=s.feature_names,
            # Fusion consumes both base explanations, so it inherits both costs.
            runtime_s=float(s.runtime_s + l.runtime_s), seed=s.seed, config_hash=cfg))
    return out


def random_control(cell, instance_ids, seed: int = SEED,
                   n_draws: int = RANDOM_CONTROL_DRAWS) -> list[Explanation]:
    """Uniformly random rankings. Without this, an AOPC value has no scale:
    deleting any five of eight features moves a prediction."""
    ids = np.asarray(instance_ids, dtype=int)
    d = len(cell.feature_order)
    rng = np.random.default_rng(seed)
    cfg = config_hash(explainer="random", n_draws=int(n_draws))
    return [Explanation(
        dataset=cell.dataset, model=cell.model, explainer="random", instance_id=int(i),
        attributions=rng.permutation(d).astype(float),
        feature_names=tuple(cell.feature_order), runtime_s=1e-6,
        seed=int(seed), config_hash=cfg) for i in ids]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_consensus.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add src/xaibench/explain/consensus.py tests/test_consensus.py
git commit -m "feat: three rank-fusion rules, calibration split, and random control

The calibration split is the leakage control that makes H6 meaningful: without
it the fidelity weight would be fitted on the same instances the hypothesis is
tested on and the fused rule would win by construction. Fused explanations carry
the summed runtime of both base explainers, because fusion is not free."
```

---

### Task 13: Plausibility metrics and the frozen reference feature sets

**Files:**
- Create: `docs/reference_features.yaml`, `src/xaibench/metrics/plausibility.py`
- Create: `tests/test_plausibility.py`

**Interfaces:**
- Consumes: `config.paths()`, `Dataset` feature names (Task 2).
- Produces: `load_reference_sets() -> dict[str, ReferenceSet]` where `ReferenceSet` has `features: list[str]`, `citations: list[str]`, `note: str`; `jaccard_vs_reference(ranking, feature_order, reference, k=5) -> float`; `precision_at_k(ranking, feature_order, reference, k=5) -> float`.

- [ ] **Step 1: Write `docs/reference_features.yaml` and commit it BEFORE the metric**

This file must be committed before any explanation is computed — that ordering is what makes RQ5 a pre-registered check rather than a post-hoc rationalisation. Every entry carries its citation.

```yaml
# Pre-registered plausibility reference sets. FROZEN 2026-08-29.
#
# These are NOT ground truth. They are literature-derived reference sets used only
# as a corroborating plausibility check (protocol §5.5). They are incomplete, they
# are not causal, and they reflect the measurement and publication conventions of
# their source literature. A low overlap score is as consistent with an incomplete
# reference set as with a poor explanation. RQ5 is the weakest metric in this study
# and is reported as such.
pima:
  features: [Glucose, BMI, Age, DiabetesPedigreeFunction, Insulin]
  citations:
    - "American Diabetes Association, 'Classification and diagnosis of diabetes:
       Standards of Medical Care in Diabetes-2021,' Diabetes Care, vol. 44, 2021."
    - "J. W. Smith et al., 'Using the ADAP learning algorithm to forecast the onset
       of diabetes mellitus,' Proc. Annu. Symp. Comput. Appl. Med. Care, 1988."
  note: >
    Plasma glucose concentration is the diagnostic basis for type 2 diabetes, and
    BMI and age are the most consistently reported risk factors in this cohort.
heart:
  features: [cp, thalach, oldpeak, ca, thal]
  citations:
    - "R. Detrano et al., 'International application of a new probability algorithm
       for the diagnosis of coronary artery disease,' Am. J. Cardiol., vol. 64, 1989."
    - "S. Nashif et al., 'Heart disease detection by using machine learning
       algorithms and a real-time cardiovascular health monitoring system,'
       World J. Eng. Technol., vol. 6, no. 4, 2018."
  note: >
    Chest-pain type, exercise-induced ST depression, maximum achieved heart rate,
    and the vessel/thalassemia indicators dominate the Cleveland literature.
ckd:
  features: [hemo, sc, al, sg, pcv]
  citations:
    - "L. J. Rubini, P. Soundarapandian, and P. Eswaran, 'Chronic Kidney Disease,'
       UCI Machine Learning Repository, 2015, doi:10.24432/C5G020."
    - "KDIGO, 'Clinical Practice Guideline for the Evaluation and Management of
       Chronic Kidney Disease,' Kidney Int. Suppl., vol. 3, no. 1, 2013."
  note: >
    Serum creatinine and haemoglobin are central to CKD staging; albumin, specific
    gravity, and packed cell volume are the next most frequently reported.
```

```bash
git add docs/reference_features.yaml
git commit -m "docs: freeze RQ5 plausibility reference sets with citations

Committed before any explanation is computed, which is what makes RQ5 a
pre-registered check rather than a post-hoc rationalisation. The file states in
its own header that these sets are not ground truth."
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_plausibility.py
import pytest
from xaibench.config import DATASETS
from xaibench.data.loaders import load_dataset
from xaibench.metrics.plausibility import (
    load_reference_sets, jaccard_vs_reference, precision_at_k,
)

def test_every_dataset_has_a_reference_set_with_citations():
    refs = load_reference_sets()
    assert set(refs) == set(DATASETS)
    for ds_id, ref in refs.items():
        assert len(ref.features) == 5, f"{ds_id}: protocol fixes |R_D| = 5"
        assert len(ref.citations) >= 2, f"{ds_id}: needs at least two sources"
        assert ref.note.strip()

def test_reference_features_all_exist_in_their_dataset():
    # A typo here would silently score zero overlap forever.
    refs = load_reference_sets()
    for ds_id, ref in refs.items():
        cols = set(load_dataset(ds_id).X.columns)
        missing = set(ref.features) - cols
        assert not missing, f"{ds_id}: {missing} not in the dataset columns"

def test_perfect_and_zero_overlap_score_as_expected():
    order = ["a", "b", "c", "d", "e", "f"]
    ranking = list(range(6))
    assert jaccard_vs_reference(ranking, order, ["a", "b", "c", "d", "e"], k=5) == pytest.approx(1.0)
    assert precision_at_k(ranking, order, ["a", "b", "c", "d", "e"], k=5) == pytest.approx(1.0)
    assert jaccard_vs_reference(ranking, order, ["f"], k=5) == pytest.approx(0.0)

def test_partial_overlap_matches_hand_computation():
    order = ["a", "b", "c", "d", "e"]
    ranking = [0, 1, 2, 3, 4]          # top-3 = {a, b, c}
    ref = ["c", "d", "e"]
    # intersection {c} = 1, union {a,b,c,d,e} = 5
    assert jaccard_vs_reference(ranking, order, ref, k=3) == pytest.approx(1 / 5)
    assert precision_at_k(ranking, order, ref, k=3) == pytest.approx(1 / 3)

def test_a_reference_feature_absent_from_the_ranking_is_ignored_not_an_error():
    assert jaccard_vs_reference([0, 1], ["a", "b"], ["a", "zzz"], k=2) == pytest.approx(1 / 2)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_plausibility.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.metrics.plausibility'`

- [ ] **Step 4: Write `src/xaibench/metrics/plausibility.py`**

```python
"""Overlap between top-k explanation features and pre-registered reference sets.

The weakest metric in the study, by design and by declaration: reference sets are
literature-derived, incomplete, and not causal. Reported as corroborating evidence
only, never as a correctness criterion (protocol §5.5).
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import yaml
from xaibench.config import paths


@dataclass(frozen=True)
class ReferenceSet:
    dataset: str
    features: list[str]
    citations: list[str]
    note: str


@lru_cache(maxsize=1)
def load_reference_sets() -> dict[str, ReferenceSet]:
    raw = yaml.safe_load((paths().docs / "reference_features.yaml").read_text())
    return {k: ReferenceSet(dataset=k, features=list(v["features"]),
                            citations=list(v["citations"]), note=str(v.get("note", "")))
            for k, v in raw.items()}


def _topk_names(ranking, feature_order, k: int) -> set[str]:
    order = np.asarray(ranking, dtype=int).ravel()
    k = max(1, min(int(k), order.size))
    return {feature_order[i] for i in order[:k]}


def jaccard_vs_reference(ranking, feature_order, reference, k: int = 5) -> float:
    top = _topk_names(ranking, feature_order, k)
    # Reference entries absent from this dataset cannot be found or missed.
    ref = set(reference) & set(feature_order)
    union = top | ref
    return float(len(top & ref) / len(union)) if union else 0.0


def precision_at_k(ranking, feature_order, reference, k: int = 5) -> float:
    top = _topk_names(ranking, feature_order, k)
    ref = set(reference) & set(feature_order)
    return float(len(top & ref) / len(top)) if top else 0.0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_plausibility.py -v`
Expected: 5 passed. `test_reference_features_all_exist_in_their_dataset` is the one that matters: a typo in the YAML would otherwise score zero overlap for the rest of the study.

- [ ] **Step 6: Commit**

```bash
git add src/xaibench/metrics/plausibility.py tests/test_plausibility.py
git commit -m "feat: plausibility overlap against the frozen reference sets

test_reference_features_all_exist_in_their_dataset guards the failure mode that
would otherwise go unnoticed: a misspelled reference feature scores zero overlap
forever and looks like a finding."
```

---

## Phase 3 — Pipeline stages

Each stage is a module with a `main()` and a `__main__` guard, reads only what earlier stages wrote, and skips completed work unless `--force`. That resumability is not a nicety on a two-day schedule: a figure tweak must never require re-running two hours of LIME.

### Task 14: Stage 1 — predictive performance via nested cross-validation

**Files:**
- Create: `src/xaibench/pipeline/stage1_performance.py`
- Create: `tests/test_stage1.py`
- Writes: `results/performance.csv`, `results/best_params.json`

**Interfaces:**
- Consumes: `load_dataset`, `build_preprocessor`, `build_model`, `param_grid`, `config.CELLS`.
- Produces: `evaluate_cell(dataset_id, model_id, profile, seed) -> tuple[pd.DataFrame, dict]`; `main(force: bool = False) -> None`. `results/performance.csv` columns: `dataset, model, fold, accuracy, precision, recall, f1, roc_auc, pr_auc, mcc, brier`. `results/best_params.json` maps `"{dataset}|{model}"` to the modal winning parameter dict across outer folds.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage1.py
import json
import numpy as np
import pandas as pd
import pytest
from xaibench.config import QUICK, CELLS, paths
from xaibench.pipeline.stage1_performance import evaluate_cell, main

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "brier"]

def test_evaluate_cell_returns_one_row_per_outer_fold():
    df, params = evaluate_cell("heart", "logreg", QUICK, seed=42)
    assert len(df) == 5                     # 5 outer folds
    assert set(METRICS).issubset(df.columns)
    assert isinstance(params, dict)

def test_metrics_are_in_their_valid_ranges():
    df, _ = evaluate_cell("heart", "logreg", QUICK, seed=42)
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert df[m].between(0.0, 1.0).all(), m
    assert df["mcc"].between(-1.0, 1.0).all()
    assert df["brier"].between(0.0, 1.0).all()

def test_the_difficulty_gradient_the_study_is_built_on_actually_holds():
    # The three datasets are chosen to span easy -> hard. If this ordering does not
    # hold, RQ1c (agreement vs difficulty) has no gradient to measure and the
    # framing in the paper must change. Better to learn it here than in Stage 4.
    auc = {ds: evaluate_cell(ds, "rf", QUICK, seed=42)[0]["roc_auc"].mean()
           for ds in ["pima", "heart", "ckd"]}
    assert auc["pima"] < auc["ckd"]
    assert auc["heart"] < auc["ckd"]
    assert auc["ckd"] > 0.95        # CKD is near-separable, protocol risk R2

def test_best_params_keys_are_real_grid_parameters():
    from xaibench.models.zoo import param_grid
    _, params = evaluate_cell("heart", "xgb", QUICK, seed=42)
    assert set(params).issubset(set(param_grid("xgb")))

def test_main_writes_every_cell_and_is_idempotent(tmp_path, monkeypatch):
    main(force=True)
    p = paths()
    df = pd.read_csv(p.results / "performance.csv")
    assert set(zip(df.dataset, df.model)) == set(CELLS)
    assert len(df) == len(CELLS) * 5
    best = json.loads((p.results / "best_params.json").read_text())
    assert set(best) == {f"{d}|{m}" for d, m in CELLS}
    before = (p.results / "performance.csv").read_text()
    main(force=False)                      # must skip, not recompute
    assert (p.results / "performance.csv").read_text() == before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_stage1.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.pipeline.stage1_performance'`

- [ ] **Step 3: Write `src/xaibench/pipeline/stage1_performance.py`**

```python
"""Track A: nested cross-validation for unbiased performance estimation.

Outer 5-fold estimates performance; inner 3-fold selects hyperparameters. Every
transformer lives inside the Pipeline, so imputation and scaling are fitted on the
inner training rows only. This is the leak the two-track design exists to prevent.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xaibench.config import CELLS, SEED, Profile, get_profile, paths
from xaibench.data.loaders import load_dataset
from xaibench.models.pipelines import build_preprocessor
from xaibench.models.zoo import build_model, param_grid

OUTER_FOLDS, INNER_FOLDS = 5, 3       # NORMATIVE (protocol §3)


def _score(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
    }


def evaluate_cell(dataset_id: str, model_id: str, profile: Profile,
                  seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    ds = load_dataset(dataset_id)
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
    grid = {f"clf__{k}": v for k, v in param_grid(model_id).items()}

    rows, winners = [], []
    for fold, (tr, te) in enumerate(outer.split(ds.X, ds.y)):
        pipe = Pipeline([("pre", build_preprocessor(ds)),
                         ("clf", build_model(model_id, seed=seed))])
        search = GridSearchCV(
            pipe, grid,
            cv=StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed),
            scoring="roc_auc", n_jobs=-1, refit=True, error_score="raise")
        search.fit(ds.X.iloc[tr], ds.y.iloc[tr])

        y_true = ds.y.iloc[te]
        y_prob = search.predict_proba(ds.X.iloc[te])[:, 1]
        rows.append({"dataset": dataset_id, "model": model_id, "fold": fold,
                     **_score(y_true, search.predict(ds.X.iloc[te]), y_prob)})
        winners.append(tuple(sorted(
            (k.removeprefix("clf__"), v) for k, v in search.best_params_.items())))

    # Modal winner across outer folds: the single configuration Track B refits with.
    best = dict(Counter(winners).most_common(1)[0][0]) if winners else {}
    return pd.DataFrame(rows), best


def main(force: bool = False) -> None:
    profile, p = get_profile(), paths()
    p.mkdirs()
    out_csv, out_json = p.results / "performance.csv", p.results / "best_params.json"
    if out_csv.exists() and out_json.exists() and not force:
        print(f"[stage1] {out_csv.name} exists; skipping (use --force to recompute)")
        return

    frames, best_params = [], {}
    for dataset_id, model_id in CELLS:
        print(f"[stage1] {dataset_id:6s} {model_id:7s} nested CV "
              f"({OUTER_FOLDS}x{INNER_FOLDS})", flush=True)
        df, best = evaluate_cell(dataset_id, model_id, profile)
        frames.append(df)
        best_params[f"{dataset_id}|{model_id}"] = best

    pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(best_params, indent=2, sort_keys=True) + "\n")
    print(f"[stage1] wrote {out_csv} and {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_stage1.py -v`
Expected: 5 passed, ~4 minutes.

If `test_the_difficulty_gradient_the_study_is_built_on_actually_holds` fails, stop and reconsider the framing before continuing — RQ1c and figure F6 both depend on that gradient existing.

- [ ] **Step 5: Commit**

```bash
git add src/xaibench/pipeline/stage1_performance.py tests/test_stage1.py
git commit -m "feat: Stage 1 nested-CV performance evaluation

Every transformer lives inside the Pipeline so imputation and scaling are fitted
on inner-training rows only. test_the_difficulty_gradient asserts the easy/hard
ordering across datasets that RQ1c and figure F6 both depend on; if it ever
fails, the paper's framing is wrong, not the test."
```

---

### Task 15: Stage 2 — compute every explanation

**Files:**
- Create: `src/xaibench/pipeline/stage2_explain.py`
- Create: `tests/test_stage2.py`
- Writes: `results/attributions/{dataset}_{model}.parquet`, `results/explainer_config.csv`, `results/instance_budget.csv`, `results/stability_rerun.parquet`, `results/stability_robust.parquet`, `results/stability_retrain.parquet`, `results/consensus_splits.csv`

**Interfaces:**
- Consumes: everything in Phase 2, plus `results/best_params.json` from Task 14.
- Produces: `explain_cell(dataset_id, model_id, params, profile) -> CellArtifacts` where `CellArtifacts` is a dataclass of `attributions: pd.DataFrame`, `rerun: pd.DataFrame`, `robust: pd.DataFrame`, `retrain: pd.DataFrame`, `config_row: dict`, `budget_row: dict`, `calib_ids: np.ndarray`, `eval_ids: np.ndarray`; `main(force: bool = False) -> None`.

The `rerun` frame has columns `dataset, model, explainer, instance_id, num_samples, rerun_cv, repeat_pair_rho`. The `robust` frame has `dataset, model, explainer, instance_id, lipschitz`. The `retrain` frame has `dataset, model, explainer, refit, feature, global_attribution`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage2.py
import numpy as np
import pandas as pd
import pytest
from xaibench.config import QUICK, ALL_EXPLAINERS, ENSEMBLE_MODELS, paths
from xaibench.pipeline.stage2_explain import explain_cell, main

@pytest.fixture(scope="module")
def artifacts():
    return explain_cell("pima", "rf", params={}, profile=QUICK)

def test_every_explainer_produced_the_same_instance_coverage(artifacts):
    df = artifacts.attributions
    assert set(df.explainer) == set(ALL_EXPLAINERS)
    counts = df.groupby("explainer").instance_id.nunique()
    assert counts.nunique() == 1, f"uneven coverage: {counts.to_dict()}"

def test_attributions_are_in_original_feature_space(artifacts):
    df = artifacts.attributions
    assert df.feature.nunique() == 8                      # pima has 8 features
    per = df.groupby(["explainer", "instance_id"]).size()
    assert (per == 8).all()

def test_ranks_are_a_permutation_per_explanation(artifacts):
    for _, g in artifacts.attributions.groupby(["explainer", "instance_id"]):
        assert sorted(g["rank"]) == list(range(8))

def test_config_row_records_the_probability_space_check(artifacts):
    row = artifacts.config_row
    assert row["shap_backend"] in {"tree", "tree_prob", "permutation"}
    assert row["additivity_error"] < 5e-3
    assert row["space"] == "probability"

def test_calibration_and_evaluation_ids_are_disjoint(artifacts):
    assert not (set(artifacts.calib_ids) & set(artifacts.eval_ids))
    assert set(artifacts.calib_ids) | set(artifacts.eval_ids) == set(
        artifacts.attributions.instance_id.unique())

def test_rerun_records_shap_as_exactly_zero_and_lime_as_positive(artifacts):
    r = artifacts.rerun
    shap_cv = r.loc[r.explainer == "shap", "rerun_cv"]
    lime_cv = r.loc[r.explainer == "lime", "rerun_cv"]
    assert (shap_cv == 0.0).all()          # tree backend, deterministic a priori
    assert (lime_cv > 0.0).any()

def test_robustness_frame_covers_both_base_explainers(artifacts):
    assert set(artifacts.robust.explainer) == {"shap", "lime"}
    assert (artifacts.robust.lipschitz >= 0).all()

def test_retrain_runs_only_for_ensembles_per_protocol_r3():
    ens = explain_cell("pima", "rf", params={}, profile=QUICK)
    non = explain_cell("pima", "logreg", params={}, profile=QUICK)
    assert "rf" in ENSEMBLE_MODELS and not ens.retrain.empty
    assert "logreg" not in ENSEMBLE_MODELS and non.retrain.empty

def test_main_writes_one_parquet_per_cell_and_resumes(capsys):
    main(force=True)
    p = paths()
    files = sorted(p.attributions.glob("*.parquet"))
    assert len(files) == 18
    stamp = files[0].stat().st_mtime
    main(force=False)
    assert files[0].stat().st_mtime == stamp, "resume did not skip a completed cell"
    assert (p.results / "explainer_config.csv").exists()
    assert (p.results / "instance_budget.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_stage2.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.pipeline.stage2_explain'`

- [ ] **Step 3: Write `src/xaibench/pipeline/stage2_explain.py`**

```python
"""Stage 2: every explanation the study needs, checkpointed per cell.

Per cell this produces base attributions (SHAP, LIME at each num_samples), three
fusion variants, the random control, and the three stability components. One
parquet per cell so a crash at cell 15 does not discard cells 1-14.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from sklearn.utils import resample
from xaibench.config import (
    CELLS, ENSEMBLE_MODELS, SEED, Profile, get_profile, paths,
)
from xaibench.explain.base import explanations_to_frame
from xaibench.explain.consensus import (
    build_consensus, calibration_split, fidelity_weight, random_control,
)
from xaibench.explain.lime_backend import explain_lime
from xaibench.explain.shap_backend import (
    additivity_error, build_background, explain_shap, _raw_shap_values,
)
from xaibench.metrics.fidelity import aopc_deletion, median_baseline
from xaibench.metrics.stability import (
    local_lipschitz, rerun_cv, repeat_pair_rho, sample_neighbours,
)
from xaibench.models.cells import build_cell
from xaibench.models.zoo import build_model, shap_backend_kind
from xaibench.config import AOPC_MAX_K, LIPSCHITZ_SIGMA


@dataclass
class CellArtifacts:
    attributions: pd.DataFrame
    rerun: pd.DataFrame
    robust: pd.DataFrame
    retrain: pd.DataFrame
    config_row: dict
    budget_row: dict
    calib_ids: np.ndarray
    eval_ids: np.ndarray


def _mean_aopc(cell, exps, baseline) -> float:
    return float(np.mean([
        aopc_deletion(cell.predict_proba, cell.X_test_t[e.instance_id],
                      _transformed_ranking(cell, e), baseline, AOPC_MAX_K)
        for e in exps])) if exps else 0.0


def _transformed_ranking(cell, explanation) -> np.ndarray:
    """Lift an original-feature ranking to transformed-column order.

    Fidelity perturbs the model's actual inputs, which are transformed columns, but
    rankings live in original-feature space. Deleting feature f means deleting every
    one-hot column belonging to f, so the columns of a group stay adjacent here.
    """
    groups = np.asarray(cell.groups)
    order = []
    for feat_idx in explanation.ranking:
        feat = cell.feature_order[int(feat_idx)]
        order.extend(np.flatnonzero(groups == feat).tolist())
    return np.asarray(order, dtype=int)


def _rerun_rows(cell, profile, ids) -> list[dict]:
    """RQ2a: repeat each explainer R times with different seeds on a small subset."""
    rows = []
    subset = ids[: profile.rerun_instances]
    bg = build_background(cell, profile.shap_background, SEED)

    shap_runs = np.array([[e.attributions for e in
                           explain_shap(cell, subset, seed=1000 + r, background=bg)]
                          for r in range(profile.rerun_repeats)])
    for i, inst in enumerate(subset):
        rows.append({"dataset": cell.dataset, "model": cell.model, "explainer": "shap",
                     "instance_id": int(inst), "num_samples": 0,
                     "rerun_cv": rerun_cv(shap_runs[:, i, :]),
                     "repeat_pair_rho": repeat_pair_rho(shap_runs[:, i, :])})

    for n_samp in profile.lime_num_samples:
        runs = np.array([[e.attributions for e in
                          explain_lime(cell, subset, num_samples=n_samp, seed=1000 + r)]
                         for r in range(profile.rerun_repeats)])
        for i, inst in enumerate(subset):
            rows.append({"dataset": cell.dataset, "model": cell.model, "explainer": "lime",
                         "instance_id": int(inst), "num_samples": int(n_samp),
                         "rerun_cv": rerun_cv(runs[:, i, :]),
                         "repeat_pair_rho": repeat_pair_rho(runs[:, i, :])})
    return rows


def _robust_rows(cell, profile, ids) -> list[dict]:
    """RQ2b: local Lipschitz estimate over Gaussian neighbours in standardised space."""
    rows = []
    subset = ids[: profile.robust_instances]
    bg = build_background(cell, profile.shap_background, SEED)
    sigma = cell.train_sigma * LIPSCHITZ_SIGMA

    for inst in subset:
        x = cell.X_test_t[int(inst)]
        nbrs = sample_neighbours(x, sigma, profile.robust_neighbours,
                                 seed=SEED + int(inst), fixed_mask=cell.fixed_mask)
        # Neighbours are not test rows, so they are appended and explained by index.
        extended = np.vstack([cell.X_test_t, nbrs])
        probe = _CellView(cell, extended)
        nbr_ids = np.arange(cell.X_test_t.shape[0], extended.shape[0])

        for name, fn in (("shap", lambda i: explain_shap(probe, i, seed=SEED, background=bg)),
                         ("lime", lambda i: explain_lime(
                             probe, i, num_samples=profile.lime_num_samples[0], seed=SEED))):
            phi_x = fn(np.array([int(inst)]))[0].attributions
            phi_n = np.array([e.attributions for e in fn(nbr_ids)])
            rows.append({"dataset": cell.dataset, "model": cell.model, "explainer": name,
                         "instance_id": int(inst),
                         "lipschitz": local_lipschitz(phi_x, x, phi_n, nbrs)})
    return rows


class _CellView:
    """A Cell with a substituted test matrix, so neighbours can be explained by index."""
    def __init__(self, cell, X_test_t):
        self._cell, self.X_test_t = cell, np.asarray(X_test_t, dtype=float)

    def __getattr__(self, name):
        return getattr(self._cell, name)


def _retrain_rows(cell, profile, ids) -> list[dict]:
    """RQ2c: refit on bootstrap resamples and compare GLOBAL rankings.

    Restricted to ensemble models (protocol R3): the LIME cost of running this for
    all six models does not fit the compute budget, and the restriction is disclosed.
    """
    if cell.model not in ENSEMBLE_MODELS:
        return []
    rows = []
    subset = ids[: profile.bootstrap_instances]
    for b in range(profile.bootstrap_refits):
        idx = resample(np.arange(cell.X_train_t.shape[0]), replace=True,
                       n_samples=cell.X_train_t.shape[0], stratify=cell.y_train,
                       random_state=2000 + b)
        clf = build_model(cell.model, seed=SEED).fit(cell.X_train_t[idx], cell.y_train[idx])
        refit = _CellView(cell, cell.X_test_t)
        refit._cell = type("C", (), {**cell.__dict__, "classifier": clf})()
        for name, exps in (("shap", explain_shap(refit, subset, seed=SEED)),
                           ("lime", explain_lime(refit, subset,
                                                 num_samples=profile.lime_num_samples[0],
                                                 seed=SEED))):
            glob = np.mean([np.abs(e.attributions) for e in exps], axis=0)
            for j, feat in enumerate(cell.feature_order):
                rows.append({"dataset": cell.dataset, "model": cell.model,
                             "explainer": name, "refit": b, "feature": feat,
                             "global_attribution": float(glob[j])})
    return rows


def explain_cell(dataset_id: str, model_id: str, params: dict,
                 profile: Profile) -> CellArtifacts:
    cell = build_cell(dataset_id, model_id, params, profile)
    ids = cell.instance_ids
    bg = build_background(cell, profile.shap_background, SEED)
    baseline = median_baseline(cell.X_train_t)

    raw, base = _raw_shap_values(cell, ids, bg, SEED)
    add_err = additivity_error(cell, raw, base, ids)

    shap_exps = explain_shap(cell, ids, seed=SEED, background=bg)
    frames = [explanations_to_frame(shap_exps)]
    lime_by_n = {}
    for n_samp in profile.lime_num_samples:
        lime_by_n[n_samp] = explain_lime(cell, ids, num_samples=n_samp, seed=SEED)
        df = explanations_to_frame(lime_by_n[n_samp])
        df["num_samples"] = int(n_samp)
        frames.append(df)

    primary_lime = lime_by_n[profile.lime_num_samples[0]]
    calib_ids, eval_ids = calibration_split(ids, seed=SEED)
    pos = {int(i): k for k, i in enumerate(ids)}
    calib_pos = [pos[int(i)] for i in calib_ids]
    weight = fidelity_weight(
        _mean_aopc(cell, [shap_exps[k] for k in calib_pos], baseline),
        _mean_aopc(cell, [primary_lime[k] for k in calib_pos], baseline))

    for rule in ("rankavg", "rrf", "fidwt"):
        frames.append(explanations_to_frame(
            build_consensus(shap_exps, primary_lime, weight, rule)))
    frames.append(explanations_to_frame(random_control(cell, ids, seed=SEED)))

    attributions = pd.concat(frames, ignore_index=True)
    attributions["num_samples"] = attributions.get(
        "num_samples", pd.Series(0, index=attributions.index)).fillna(0).astype(int)

    return CellArtifacts(
        attributions=attributions,
        rerun=pd.DataFrame(_rerun_rows(cell, profile, ids)),
        robust=pd.DataFrame(_robust_rows(cell, profile, ids)),
        retrain=pd.DataFrame(_retrain_rows(cell, profile, ids)),
        config_row={"dataset": dataset_id, "model": model_id,
                    "shap_backend": shap_backend_kind(model_id),
                    "space": "probability", "additivity_error": add_err,
                    "background": int(bg.shape[0]), "fidelity_weight_shap": weight,
                    "lime_num_samples_primary": int(profile.lime_num_samples[0]),
                    "profile": profile.name},
        budget_row={"dataset": dataset_id, "model": model_id,
                    "n_test": int(cell.X_test_t.shape[0]), "n_explained": int(ids.size),
                    "n_calibration": int(calib_ids.size), "n_evaluation": int(eval_ids.size)},
        calib_ids=calib_ids, eval_ids=eval_ids)


def main(force: bool = False) -> None:
    profile, p = get_profile(), paths()
    p.mkdirs()
    best = json.loads((p.results / "best_params.json").read_text())
    configs, budgets, splits, reruns, robusts, retrains = [], [], [], [], [], []

    for dataset_id, model_id in CELLS:
        target = p.attributions / f"{dataset_id}_{model_id}.parquet"
        side = p.attributions / f"{dataset_id}_{model_id}_side.json"
        if target.exists() and side.exists() and not force:
            print(f"[stage2] {dataset_id:6s} {model_id:7s} cached")
            payload = json.loads(side.read_text())
        else:
            print(f"[stage2] {dataset_id:6s} {model_id:7s} explaining "
                  f"(profile={profile.name})", flush=True)
            art = explain_cell(dataset_id, model_id, best.get(f"{dataset_id}|{model_id}", {}),
                               profile)
            art.attributions.to_parquet(target, index=False)
            for name, frame in (("rerun", art.rerun), ("robust", art.robust),
                                ("retrain", art.retrain)):
                if not frame.empty:
                    frame.to_parquet(
                        p.attributions / f"{dataset_id}_{model_id}_{name}.parquet",
                        index=False)
            payload = {"config": art.config_row, "budget": art.budget_row,
                       "calib_ids": art.calib_ids.tolist(),
                       "eval_ids": art.eval_ids.tolist()}
            side.write_text(json.dumps(payload, indent=2) + "\n")

        configs.append(payload["config"])
        budgets.append(payload["budget"])
        splits.append({"dataset": dataset_id, "model": model_id,
                       "calibration_ids": " ".join(map(str, payload["calib_ids"])),
                       "evaluation_ids": " ".join(map(str, payload["eval_ids"]))})

    for name, frame in (("explainer_config", pd.DataFrame(configs)),
                        ("instance_budget", pd.DataFrame(budgets)),
                        ("consensus_splits", pd.DataFrame(splits))):
        frame.to_csv(p.results / f"{name}.csv", index=False)

    for name in ("rerun", "robust", "retrain"):
        parts = [pd.read_parquet(f) for f in sorted(p.attributions.glob(f"*_{name}.parquet"))]
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(
                p.results / f"stability_{name}.parquet", index=False)
    print("[stage2] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_stage2.py -v`
Expected: 9 passed under QUICK, ~6 minutes.

- [ ] **Step 5: Commit**

```bash
git add src/xaibench/pipeline/stage2_explain.py tests/test_stage2.py
git commit -m "feat: Stage 2 explanation computation with per-cell checkpointing

_transformed_ranking is the subtle piece: fidelity perturbs the model's actual
inputs (transformed columns) while rankings live in original-feature space, so
deleting a categorical feature must delete all of its one-hot columns together.
Additivity error is recorded per cell, so protocol risk R1 is auditable rather
than assumed away."
```

---

### Task 16: Stage 3 — metric computation

**Files:**
- Create: `src/xaibench/pipeline/stage3_metrics.py`
- Create: `tests/test_stage3.py`
- Writes: `results/metrics_agreement.csv`, `results/metrics_fidelity.csv`, `results/metrics_plausibility.csv`, `results/metrics_runtime.csv`, `results/metrics_stability.csv`

**Interfaces:**
- Consumes: Stage 2 outputs and all Phase 1 metrics.
- Produces: `agreement_metrics(attributions) -> pd.DataFrame` with columns `dataset, model, instance_id, spearman, kendall, jaccard3, jaccard5, rbo`; `fidelity_metrics(...) -> pd.DataFrame` with `dataset, model, explainer, instance_id, baseline, aopc, auc_ins, comprehensiveness, sufficiency, is_evaluation`; `plausibility_metrics(...)`; `runtime_metrics(...)`; `stability_summary(...)`; `main(force=False)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage3.py
import numpy as np
import pandas as pd
import pytest
from xaibench.config import ALL_EXPLAINERS, CELLS, paths
from xaibench.pipeline.stage3_metrics import main

@pytest.fixture(scope="module")
def outputs():
    main(force=True)
    p = paths()
    return {n: pd.read_csv(p.results / f"metrics_{n}.csv")
            for n in ["agreement", "fidelity", "plausibility", "runtime", "stability"]}

def test_agreement_covers_every_cell_and_stays_in_range(outputs):
    df = outputs["agreement"]
    assert set(zip(df.dataset, df.model)) == set(CELLS)
    assert df.spearman.between(-1, 1).all()
    assert df.kendall.between(-1, 1).all()
    for col in ["jaccard3", "jaccard5", "rbo"]:
        assert df[col].between(0, 1).all(), col

def test_fidelity_covers_every_explainer_including_the_control(outputs):
    df = outputs["fidelity"]
    assert set(df.explainer) == set(ALL_EXPLAINERS)
    assert set(df.baseline) == {"median", "marginal"}
    assert df.is_evaluation.dtype == bool

def test_real_explainers_beat_the_random_control_on_average(outputs):
    # Not a hypothesis test, a sanity check. If SHAP does not beat random here,
    # something is wrong with the ranking or the perturbation loop, not with SHAP.
    df = outputs["fidelity"].query("baseline == 'median'")
    means = df.groupby("explainer").aopc.mean()
    assert means["shap"] > means["random"]
    assert means["lime"] > means["random"]

def test_runtime_is_positive_and_lime_costs_more_than_tree_shap(outputs):
    df = outputs["runtime"]
    assert (df.median_seconds > 0).all()
    tree = df.query("model in ['rf','xgb','lgbm','dtree'] and explainer == 'shap'")
    lime = df.query("model in ['rf','xgb','lgbm','dtree'] and explainer == 'lime'")
    assert lime.median_seconds.median() > tree.median_seconds.median()

def test_plausibility_scores_are_bounded_and_cover_both_base_explainers(outputs):
    df = outputs["plausibility"]
    assert df.jaccard5.between(0, 1).all()
    assert df.precision5.between(0, 1).all()
    assert {"shap", "lime"}.issubset(set(df.explainer))

def test_stability_summary_reports_all_three_components(outputs):
    df = outputs["stability"]
    assert {"rerun_cv", "repeat_pair_rho", "lipschitz"}.issubset(df.columns)
    shap_rows = df.query("explainer == 'shap' and model in ['rf','xgb','lgbm','dtree']")
    assert (shap_rows.rerun_cv.fillna(0) == 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_stage3.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.pipeline.stage3_metrics'`

- [ ] **Step 3: Write `src/xaibench/pipeline/stage3_metrics.py`**

Implement each function against the contracts above, reusing Phase 1 metrics without reimplementation:

- `agreement_metrics` pivots the attribution frame to one row per `(dataset, model, instance_id)`, extracts the SHAP and primary-LIME rank vectors, and calls `spearman`, `kendall_tau_b`, `jaccard_at_k(k=3)`, `jaccard_at_k(k=5)`, and `rbo` from `metrics.agreement`. A global variant is appended with `instance_id = -1`, computed from mean `|attribution|` per feature.
- `fidelity_metrics` rebuilds each cell with `build_cell` (cheap and deterministic), computes `median_baseline` and `marginal_baselines(n_draws=BASELINE_MARGINAL_DRAWS)`, and for every explainer × instance × baseline variant calls `aopc_deletion`, `auc_insertion`, `comprehensiveness(k=5)`, `sufficiency(k=5)` using `_transformed_ranking` from Stage 2. `is_evaluation` is `instance_id in eval_ids` read from `results/consensus_splits.csv`, so H6 can be tested on evaluation instances only.
- `plausibility_metrics` calls `jaccard_vs_reference` and `precision_at_k` per explainer × instance against `load_reference_sets()`.
- `runtime_metrics` aggregates `runtime_s` to median and IQR per `(dataset, model, explainer, num_samples)`, and adds `fidelity_per_second = aopc / median_seconds`.
- `stability_summary` outer-joins the three Stage 2 stability frames on `(dataset, model, explainer, instance_id)` and adds `retrain_agreement` per cell via `metrics.stability.retrain_agreement` over the pivoted `global_attribution` values.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_stage3.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/xaibench/pipeline/stage3_metrics.py tests/test_stage3.py
git commit -m "feat: Stage 3 metric computation across all five families

test_real_explainers_beat_the_random_control is a sanity check rather than a
hypothesis test: if SHAP does not beat a random ranking, the bug is in the
ranking lift or the perturbation loop, not in SHAP."
```

---

### Task 17: Stage 4 — hypothesis testing

**Files:**
- Create: `src/xaibench/pipeline/stage4_stats.py`
- Create: `tests/test_stage4.py`
- Writes: `results/stats_hypotheses.csv`, `results/stats_estimates.csv`

**Interfaces:**
- Consumes: Stage 3 outputs, `stats.tests`.
- Produces: `run_all_hypotheses(profile) -> pd.DataFrame` with columns `hypothesis, dataset, model, unit, n, test, alternative, statistic, p_value, p_adj, effect_size, effect_name, median_diff, ci_low, ci_high, degenerate, decision`; `estimate_table(profile) -> pd.DataFrame` with `metric, dataset, model, explainer, point, ci_low, ci_high, n`; `main(force=False)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage4.py
import numpy as np
import pandas as pd
import pytest
from xaibench.config import ALPHA, paths
from xaibench.pipeline.stage4_stats import main

EXPECTED_HYPOTHESES = {"H1", "H1b", "H1c", "H2a", "H2b", "H2c", "H3a", "H3b", "H4", "H5", "H6"}

@pytest.fixture(scope="module")
def stats_frame():
    main(force=True)
    return pd.read_csv(paths().results / "stats_hypotheses.csv")

def test_all_eleven_pre_registered_hypotheses_are_tested(stats_frame):
    assert set(stats_frame.hypothesis) == EXPECTED_HYPOTHESES

def test_family_sizes_match_the_protocol(stats_frame):
    counts = stats_frame.hypothesis.value_counts().to_dict()
    assert counts["H1"] == 18 and counts["H2a"] == 18 and counts["H2b"] == 18
    assert counts["H3a"] == 18 and counts["H3b"] == 36 and counts["H4"] == 18
    assert counts["H6"] == 18 and counts["H2c"] == 9
    assert counts["H1b"] == 1 and counts["H1c"] == 1 and counts["H5"] == 1

def test_holm_adjustment_is_never_more_lenient_than_the_raw_p(stats_frame):
    assert (stats_frame.p_adj >= stats_frame.p_value - 1e-12).all()

def test_holm_is_applied_within_family_not_across_all_tests(stats_frame):
    # A family of 18 must not be corrected as if it were a family of 173.
    fam = stats_frame.query("hypothesis == 'H1' and not degenerate")
    if len(fam) > 1:
        ratio = (fam.p_adj / fam.p_value.clip(lower=1e-300)).max()
        assert ratio <= len(fam) + 1e-9

def test_decision_follows_the_adjusted_p_value(stats_frame):
    rejected = stats_frame.query("decision == 'reject H0'")
    assert (rejected.p_adj < ALPHA).all()
    assert (stats_frame.query("decision == 'fail to reject H0'").p_adj >= ALPHA).all()

def test_every_non_degenerate_test_reports_an_effect_size(stats_frame):
    live = stats_frame.query("not degenerate")
    assert live.effect_size.notna().all()
    assert live.effect_name.notna().all()

def test_deterministic_shap_cells_are_marked_degenerate_not_dropped(stats_frame):
    # H2a for tree models: all differences are exactly zero. The row must exist,
    # flagged, so table T10 has no holes.
    rows = stats_frame.query("hypothesis == 'H2a' and model in ['rf','xgb','lgbm','dtree']")
    assert len(rows) == 12
    assert rows.degenerate.any()

def test_estimates_carry_confidence_intervals_that_bracket_the_point(stats_frame):
    est = pd.read_csv(paths().results / "stats_estimates.csv")
    ok = est.dropna(subset=["ci_low", "ci_high", "point"])
    assert (ok.ci_low <= ok.point + 1e-9).all()
    assert (ok.point <= ok.ci_high + 1e-9).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_stage4.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.pipeline.stage4_stats'`

- [ ] **Step 3: Write `src/xaibench/pipeline/stage4_stats.py`**

Implement one function per hypothesis, each returning a list of row dicts, then Holm-correct **within** each hypothesis family before assembling. The exact test, alternative, and unit of analysis for each hypothesis come from `docs/PROTOCOL.md` §6; copy them rather than re-deriving:

| Hypothesis | Data | Call |
|-----------|------|------|
| H1 | `metrics_agreement.spearman` per cell (instances only) | `wilcoxon_with_effect(rho, mu=0.50, alternative="greater")` |
| H1b | per-cell median ρ, grouped `{logreg, dtree}` vs `{rf, xgb, lgbm, stack}` | `mannwhitney_with_effect(simple, ensemble)` |
| H1c | per-cell median ρ vs `1 − mean roc_auc` from `performance.csv` | `spearmanr` + 10 000-draw permutation test |
| H2a | `stability.rerun_cv` per cell, per explainer | `wilcoxon_with_effect(cv, alternative="greater")` |
| H2b | `stability.lipschitz`, SHAP vs LIME paired on instance | `wilcoxon_with_effect(shap, lime)` |
| H2c | `retrain_agreement` per cell, ensembles only | `wilcoxon_with_effect(shap, lime)` |
| H3a | `metrics_fidelity.aopc` (median baseline), SHAP vs LIME paired | `wilcoxon_with_effect(shap, lime)` |
| H3b | each of SHAP, LIME vs `random`, paired | `wilcoxon_with_effect(x, rand, alternative="greater")` |
| H4 | `runtime_s` per instance, SHAP vs LIME paired | `wilcoxon_with_effect(shap, lime, alternative="less")` |
| H5 | `metrics_plausibility.jaccard5` per cell, SHAP vs LIME paired | `wilcoxon_with_effect(shap, lime)` |
| H6 | `aopc` on **evaluation instances only**, `max(rankavg, rrf, fidwt)` vs `max(shap, lime)` per instance | `wilcoxon_with_effect(best_fused, best_single, alternative="greater")` |

`estimate_table` emits a BCa interval via `bca_ci(..., n_resamples=profile.ci_resamples, seed=SEED)` for every point estimate that appears in a table or figure: ρ, τ, Jaccard@3/@5, RBO, AOPC, AUC-ins, comprehensiveness, sufficiency, Lipschitz, rerun CV, runtime, and plausibility.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_stage4.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/xaibench/pipeline/stage4_stats.py tests/test_stage4.py
git commit -m "feat: Stage 4 hypothesis testing with within-family Holm correction

test_holm_is_applied_within_family guards the error that would quietly destroy
the study's power: correcting 18 cells as if they were one family of 173. H6 is
tested on evaluation instances only, so the fidelity weight never sees them."
```

---

### Task 18: Figures

**Files:**
- Create: `src/xaibench/viz/theme.py`, `src/xaibench/viz/figures.py`
- Create: `tests/test_figures.py`
- Writes: `figures/F1..F14.{pdf,png}`

**Interfaces:**
- Consumes: everything in `results/`.
- Produces: `apply_theme() -> None`; `PALETTE: dict[str, str]`; `save(fig, name: str, profile) -> None` (writes both formats, stamps the QUICK watermark); `FIGURES: dict[str, Callable[[Profile], Figure]]` keyed `"F1"`…`"F14"`; `build_all(profile) -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figures.py
import matplotlib
matplotlib.use("Agg")
import pytest
from xaibench.config import QUICK, paths
from xaibench.viz.figures import FIGURES, build_all
from xaibench.viz.theme import PALETTE, apply_theme

def test_manifest_covers_every_planned_figure():
    assert set(FIGURES) == {f"F{i}" for i in range(1, 15)}

def test_palette_is_colourblind_safe_and_distinct_in_greyscale():
    import numpy as np
    lum = {}
    for name, hexcode in PALETTE.items():
        r, g, b = (int(hexcode[i:i + 2], 16) / 255 for i in (1, 3, 5))
        lum[name] = 0.2126 * r + 0.7152 * g + 0.0722 * b
    vals = sorted(lum.values())
    assert min(np.diff(vals)) > 0.04, f"luminance too close for print: {lum}"

def test_every_figure_writes_both_formats(tmp_path):
    apply_theme()
    names = build_all(QUICK)
    p = paths()
    for name in names:
        assert (p.figures / f"{name}.pdf").exists(), name
        assert (p.figures / f"{name}.png").exists(), name
        assert (p.figures / f"{name}.pdf").stat().st_size > 1000, name

def test_quick_profile_watermarks_so_a_draft_cannot_be_mistaken_for_results():
    apply_theme()
    fig = FIGURES["F3"](QUICK)
    texts = [t.get_text() for t in fig.texts]
    assert any("QUICK" in t for t in texts)

def test_axes_carry_labels_because_captions_must_stand_alone():
    apply_theme()
    for name in ["F3", "F4", "F10", "F11"]:
        fig = FIGURES[name](QUICK)
        for ax in fig.get_axes():
            if ax.get_visible() and ax.has_data():
                assert ax.get_xlabel() or ax.get_ylabel(), f"{name}: unlabelled axis"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_figures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.viz.figures'`

- [ ] **Step 3: Write `src/xaibench/viz/theme.py`**

```python
"""One visual system for every figure: colourblind-safe, legible in greyscale."""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito, chosen for deuteranopia safety AND separated luminance, so the figures
# survive a black-and-white print of the paper.
PALETTE = {
    "shap": "#0072B2", "lime": "#D55E00", "rankavg": "#009E73",
    "rrf": "#CC79A7", "fidwt": "#56B4E9", "random": "#7F7F7F",
}
DATASET_ORDER = ("pima", "heart", "ckd")
DATASET_LABELS = {"pima": "PIMA Diabetes", "heart": "Heart Disease", "ckd": "Chronic Kidney"}
EXPLAINER_LABELS = {"shap": "SHAP", "lime": "LIME", "rankavg": "Fusion: mean-rank",
                    "rrf": "Fusion: RRF", "fidwt": "Fusion: fidelity-weighted",
                    "random": "Random control"}


def apply_theme() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "axes.grid": True,
        "axes.grid.axis": "y", "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.4,
        "figure.constrained_layout.use": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def watermark(fig, profile) -> None:
    if not profile.is_reportable:
        fig.text(0.5, 0.5, f"PROFILE={profile.name}\nNOT FOR REPORTING",
                 ha="center", va="center", fontsize=26, color="red",
                 alpha=0.18, rotation=30, zorder=1000)
```

- [ ] **Step 4: Write `src/xaibench/viz/figures.py`**

One function per figure, each taking `profile` and returning a `Figure`, registered in `FIGURES`. Follow `docs/PAPER_PLAN.md` §4 for what each shows. Requirements every function must satisfy, because these are what make the figures usable in a paper rather than notebook output:

- Axis labels with units and the sample size (`"Spearman ρ (n = 154 instances)"`).
- Colours only from `PALETTE`, keyed by explainer, never matplotlib defaults.
- Explicit `set_ylim` on bounded metrics so panels are visually comparable.
- Error bars are bootstrap CIs read from `results/stats_estimates.csv`, never `±1 SD`.
- `save(fig, name, profile)` writes `.pdf` and `.png` and applies `watermark`.
- F1 is a schematic drawn with `matplotlib.patches`, not an imported image, so it regenerates with the rest.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_figures.py -v`
Expected: 5 passed; 28 files in `figures/`.

- [ ] **Step 6: Commit**

```bash
git add src/xaibench/viz tests/test_figures.py
git commit -m "feat: figure system with a print-safe, colourblind-safe palette

The palette test asserts separated luminance, not just distinct hues: a reader
printing the paper in black and white must still be able to tell SHAP from LIME.
QUICK-profile figures carry a watermark so a draft cannot be mistaken for results."
```

---

### Task 19: LaTeX tables, Stage 5 orchestration, and the reproducibility manifest

**Files:**
- Create: `src/xaibench/stats/report.py`, `src/xaibench/pipeline/stage5_outputs.py`
- Create: `tests/test_report.py`, `tests/test_determinism.py`
- Writes: `tables/T1..T12.tex`, `results/MANIFEST.json`

**Interfaces:**
- Consumes: everything in `results/` and `figures/`.
- Produces: `to_booktabs(df, caption, label, column_format=None, notes=None) -> str`; `fmt(value, digits=3) -> str`; `fmt_ci(point, lo, hi, digits=3) -> str`; `fmt_p(p) -> str` (renders `< 0.001` below that threshold); `TABLES: dict[str, Callable[[Profile], str]]` keyed `"T1"`…`"T12"`; `write_manifest(profile) -> dict`; `main(force=False)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
import json
import pandas as pd
import pytest
from xaibench.config import QUICK, paths
from xaibench.stats.report import TABLES, fmt, fmt_ci, fmt_p, to_booktabs
from xaibench.pipeline.stage5_outputs import main

def test_manifest_covers_every_planned_table():
    assert set(TABLES) == {f"T{i}" for i in range(1, 13)}

def test_booktabs_output_is_compilable_latex():
    df = pd.DataFrame({"model": ["RF"], "value": [0.5]})
    tex = to_booktabs(df, caption="Cap", label="tbl:x")
    for token in [r"\begin{table}", r"\toprule", r"\midrule", r"\bottomrule",
                  r"\end{table}", r"\label{tbl:x}"]:
        assert token in tex
    assert tex.count(r"\begin{tabular}") == tex.count(r"\end{tabular}") == 1

def test_latex_special_characters_are_escaped():
    df = pd.DataFrame({"name": ["a_b & c% 50$"]})
    tex = to_booktabs(df, caption="C", label="l")
    assert r"a\_b" in tex and r"\&" in tex and r"\%" in tex and r"\$" in tex

def test_p_values_are_rendered_with_a_floor_not_as_zero():
    assert fmt_p(0.0000001) == r"$<$0.001"
    assert fmt_p(0.0234) == "0.023"
    assert fmt_p(float("nan")) == "--"

def test_missing_values_render_as_a_dash_not_the_word_nan():
    assert fmt(float("nan")) == "--"
    assert fmt_ci(float("nan"), 1.0, 2.0) == "--"
    assert fmt(0.12345, digits=3) == "0.123"

def test_ci_format_is_compact_and_readable():
    assert fmt_ci(0.5, 0.4, 0.6) == "0.500 [0.400, 0.600]"

def test_stage5_writes_every_table_figure_and_the_manifest():
    main(force=True)
    p = paths()
    for i in range(1, 13):
        f = p.tables / f"T{i}.tex"
        assert f.exists() and f.stat().st_size > 60, f"T{i}"
    assert len(list(p.figures.glob("*.pdf"))) == 14
    manifest = json.loads((p.results / "MANIFEST.json").read_text())
    for key in ["profile", "git_commit", "seeds", "library_versions",
                "artifacts", "duration_s", "hardware"]:
        assert key in manifest, key
    assert all(len(v["sha256"]) == 64 for v in manifest["artifacts"].values())

def test_no_table_contains_a_literal_nan_string():
    # A "nan" in a submitted table is the most embarrassing possible defect.
    p = paths()
    for f in p.tables.glob("*.tex"):
        text = f.read_text()
        assert "nan" not in text.lower().replace("banana", ""), f.name
```

```python
# tests/test_determinism.py
import hashlib
import pandas as pd
import pytest
from xaibench.config import paths

VOLATILE = {"runtime_s", "median_seconds", "iqr_seconds", "fidelity_per_second",
            "duration_s", "q1_seconds", "q3_seconds"}

def _digest(path):
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    df = df.drop(columns=[c for c in df.columns if c in VOLATILE], errors="ignore")
    df = df.sort_index(axis=1).sort_values(list(df.columns)).reset_index(drop=True)
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()

@pytest.mark.slow
def test_two_runs_from_the_same_seed_agree_on_every_non_timing_column():
    """Run manually: make smoke && python -m tests.snapshot save && make smoke
    && pytest tests/test_determinism.py. This is the test that keeps the
    reproducibility claim honest rather than aspirational."""
    p = paths()
    snapshot = p.results / "_determinism_snapshot.json"
    if not snapshot.exists():
        pytest.skip("no snapshot; see the docstring for the procedure")
    import json
    expected = json.loads(snapshot.read_text())
    for name, want in expected.items():
        assert _digest(p.results / name) == want, f"{name} changed between runs"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaibench.stats.report'`

- [ ] **Step 3: Write `src/xaibench/stats/report.py`**

`fmt`, `fmt_ci`, and `fmt_p` are the only paths by which a number reaches a document, so all missing-value and rounding policy lives in exactly one place. `to_booktabs` escapes `_ & % $ # { } ~ ^` in every cell — CKD feature names contain underscores and would otherwise fail the LaTeX build.

Each `TABLES[Ti]` reads from `results/` and returns a complete `table` environment, following the manifest in `docs/PAPER_PLAN.md` §5. T10, the hypothesis decision table, is built directly from `results/stats_hypotheses.csv` with columns: hypothesis, unit, n, test, statistic, p, p_adj, effect size, 95 % CI, decision.

- [ ] **Step 4: Write `src/xaibench/pipeline/stage5_outputs.py`**

Calls `build_all(profile)` for figures, writes every `TABLES` entry to `tables/Ti.tex`, then `write_manifest(profile)` recording: profile name, `git rev-parse HEAD`, the full seed set, `platform` and CPU/RAM, resolved versions of every library in `requirements.lock.txt`, wall-clock duration, and the SHA-256 of every file in `results/`, `figures/`, and `tables/`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_report.py -v && .venv/bin/pytest -q`
Expected: 8 passed, then the whole suite green.

- [ ] **Step 6: Run the full QUICK pipeline end to end**

Run: `make smoke`
Expected: stages 1–5 complete in roughly five minutes; `results/`, `figures/`, and `tables/` populated; `results/MANIFEST.json` present.

- [ ] **Step 7: Commit**

```bash
git add src/xaibench/stats/report.py src/xaibench/pipeline/stage5_outputs.py tests/test_report.py tests/test_determinism.py
git commit -m "feat: LaTeX table generation, Stage 5 orchestration, and manifest

fmt/fmt_ci/fmt_p are the only path by which a number reaches a document, so
missing-value policy lives in one place and test_no_table_contains_a_literal_nan
can guard the whole output surface. The manifest records a SHA-256 per artefact
plus commit, seeds, hardware, and resolved versions."
```

---

## Phase 4 — Documents

The three LaTeX skeletons are built and compiled **before** the expensive run, against QUICK numbers. That ordering is deliberate: on a two-day schedule, discovering a LaTeX failure after the two-hour experiment is the mistake that costs a day.

### Task 20: Conference paper skeleton, compiling against QUICK numbers

**Files:**
- Create: `paper/main.tex`, `paper/refs.bib`, `paper/preamble.tex`, `paper/sections/{01_introduction,02_related_work,03_method,04_results,05_discussion,06_limitations,07_conclusion,08_backmatter}.tex`
- Create: `tests/test_paper_build.py`

**Interfaces:**
- Consumes: `figures/*.pdf`, `tables/*.tex`.
- Produces: `paper/main.pdf`. Every result enters through `\input{../tables/Ti}` or `\includegraphics{../figures/Fi.pdf}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paper_build.py
import re
import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["paper", "report", "slides"]

@pytest.mark.parametrize("doc", ["paper"])
def test_document_compiles_without_errors(doc):
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                        "-halt-on-error", "main.tex"],
                       cwd=ROOT / doc, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-4000:]
    assert (ROOT / doc / "main.pdf").exists()

@pytest.mark.parametrize("doc", ["paper"])
def test_no_unresolved_references_or_citations(doc):
    log = (ROOT / doc / "main.log").read_text(errors="ignore")
    assert "There were undefined references" not in log
    assert "Citation" not in log or "undefined" not in log

def test_every_number_enters_through_a_generated_file():
    """The load-bearing constraint: no result is typed by hand.

    Scans section bodies for bare decimals. A number is allowed only inside an
    \\input'd table, a figure, or an explicitly whitelisted context (page counts,
    fold counts, seeds, thresholds already fixed in the protocol).
    """
    allowed = re.compile(
        r"(seed|fold|alpha|α|0\.05|p\s*=|k\s*=|top-\d|RQ\d|H\d|IEEE|"
        r"\\cite|\\ref|\\label|\\input|\\includegraphics|%)")
    offenders = []
    for f in sorted((ROOT / "paper" / "sections").glob("*.tex")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("%") or allowed.search(stripped):
                continue
            # A decimal with 2+ fractional digits in prose is almost certainly a
            # transcribed result.
            if re.search(r"\d+\.\d{2,}", stripped):
                offenders.append(f"{f.name}:{i}: {stripped[:90]}")
    assert not offenders, "hand-typed results found:\n" + "\n".join(offenders)

def test_paper_inputs_the_generated_tables_rather_than_inlining_them():
    body = "\n".join(p.read_text() for p in (ROOT / "paper" / "sections").glob("*.tex"))
    for table in ["T1", "T4", "T6", "T10"]:
        assert f"tables/{table}" in body, f"{table} is not \\input anywhere"

def test_bibliography_has_enough_entries_for_a_conference_paper():
    bib = (ROOT / "paper" / "refs.bib").read_text()
    assert bib.count("@") >= 30, f"only {bib.count('@')} entries; target is 30-40"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_paper_build.py -v`
Expected: FAIL — `paper/main.tex` does not exist.

- [ ] **Step 3: Write `paper/preamble.tex`**

```latex
% Shared preamble. IEEEtran.cls and IEEEtran.bst ship with TeX Live 2023.
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{multirow}
\usepackage{siunitx}
\usepackage{subcaption}
\usepackage[algo2e,ruled,vlined]{algorithm2e}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\usepackage{cite}
\graphicspath{{../figures/}}
\sisetup{detect-weight=true, detect-family=true}
\newcommand{\shap}{\textsc{shap}}
\newcommand{\lime}{\textsc{lime}}
\newcommand{\aopc}{\textsc{aopc}}
\newcommand{\rbo}{\textsc{rbo}}
```

- [ ] **Step 4: Write `paper/main.tex`**

```latex
\documentclass[conference]{IEEEtran}
\input{preamble}

\begin{document}
\title{SHAP versus LIME on Clinical Tabular Data: A Reproducible Benchmark of
Explanation Agreement, Stability, Fidelity, and Rank-Level Consensus}

\author{%
\IEEEauthorblockN{Samira Oasela Antika, Yeasmin Akter Shuchona,
MD.\ Marajul Haque, and Gian Juti Tripura}
\IEEEauthorblockA{Department of Computer Science and Engineering\\
University of Chittagong, Chittagong-4331, Bangladesh\\
\{22701043, 22701052, 23701070, 22701073\}@std.cu.ac.bd}%
}

\maketitle

\begin{abstract}
% Replaced with real numbers in Task 23. Placeholder text only so the build is
% verifiable before the FULL run.
Abstract pending final results.
\end{abstract}

\begin{IEEEkeywords}
explainable artificial intelligence, SHAP, LIME, ensemble learning,
model interpretability, tabular data, reproducibility
\end{IEEEkeywords}

\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_method}
\input{sections/04_results}
\input{sections/05_discussion}
\input{sections/06_limitations}
\input{sections/07_conclusion}
\input{sections/08_backmatter}

\bibliographystyle{IEEEtran}
\bibliography{refs}
\end{document}
```

- [ ] **Step 5: Write each section file as a skeleton with real structure**

Each file carries its `\section{...}`, its subsection headings from `docs/PAPER_PLAN.md` §1, one sentence of intent per subsection, and the `\input`/`\includegraphics` calls for its figures and tables. `04_results.tex` gets one `\subsection` per RQ with the figure and table already wired in, so Task 23 only replaces prose. Example:

```latex
\section{Results}\label{sec:results}

\subsection{Predictive Performance}\label{sec:results-performance}
% Intent: establish that the models are competent before interpreting them, and
% establish the easy/medium/hard difficulty gradient that Section~\ref{sec:rq1} uses.
\input{../tables/T3}
\begin{figure}[t]\centering
  \includegraphics[width=\columnwidth]{F3.pdf}
  \caption{Predictive performance across the three datasets. Error bars are
  nested cross-validation standard deviations over five outer folds.}
  \label{fig:performance}
\end{figure}

\subsection{RQ1: Agreement Between SHAP and LIME}\label{sec:rq1}
% Intent: report concordance, then whether it tracks model family and task difficulty.
\input{../tables/T4}
\begin{figure}[t]\centering
  \includegraphics[width=\columnwidth]{F4.pdf}
  \caption{Global rank agreement between \shap{} and \lime{} for every
  dataset--model cell.}\label{fig:agreement-heatmap}
\end{figure}
```

- [ ] **Step 6: Write `paper/refs.bib` with at least 30 entries**

Required anchors, each with full bibliographic detail:

| Thread | Entries |
|--------|---------|
| Method foundations | Lundberg & Lee (NeurIPS 2017); Lundberg et al. (Nature MI 2020, TreeSHAP); Ribeiro et al. (KDD 2016, LIME); Ribeiro et al. (AAAI 2018, Anchors); Shapley (1953) |
| Evaluation metrics | Samek et al. (TNNLS 2017, AOPC); DeYoung et al. (ACL 2020, ERASER); Alvarez-Melis & Jaakkola (2018, robustness); Hooker et al. (NeurIPS 2019, ROAR); Webber et al. (TOIS 2010, RBO) |
| Known failure modes | Slack et al. (AIES 2020, fooling post-hoc methods); Rudin (Nature MI 2019); Kumar et al. (ICML 2020, Shapley problems); Sundararajan & Najmi (ICML 2020) |
| Prior comparisons | at least six SHAP-vs-LIME comparison papers, for table T11 |
| Aggregation / fusion | Cormack et al. (SIGIR 2009, RRF); Bhatt et al. (2020, aggregating explanations); Rieger & Hansen (2020, ensembles of explanations) |
| Models | Breiman (2001); Chen & Guestrin (KDD 2016, XGBoost); Ke et al. (NeurIPS 2017, LightGBM); Wolpert (1992, stacking) |
| Statistics | Demšar (JMLR 2006); Holm (1979); Kerby (2014, rank-biserial); Efron (1987, BCa); Cliff (1993) |
| Data | the three dataset citations from `data/fetch.py` |
| Context | EU AI Act; Goodman & Flaxman (2017) |

- [ ] **Step 7: Build and run the tests**

Run: `make paper && .venv/bin/pytest tests/test_paper_build.py -v`
Expected: `paper/main.pdf` exists; 5 passed.

- [ ] **Step 8: Commit**

```bash
git add paper tests/test_paper_build.py
git commit -m "feat: IEEE conference paper skeleton compiling against QUICK numbers

Built before the expensive run on purpose: a LaTeX failure discovered after the
two-hour experiment costs a day. test_every_number_enters_through_a_generated_file
enforces the project's load-bearing constraint by scanning prose for bare
decimals, so a transcribed result fails the suite."
```

---

### Task 21: Thesis report and Beamer deck skeletons

**Files:**
- Create: `report/main.tex`, `report/sections/*.tex`, `report/appendices/*.tex`
- Create: `slides/main.tex`, `slides/sections/*.tex`
- Modify: `tests/test_paper_build.py` — extend the parametrised build tests to `report` and `slides`

**Interfaces:**
- Consumes: `figures/`, `tables/`, and the shared `paper/sections/` fragments where the text is identical.
- Produces: `report/main.pdf`, `slides/main.pdf`.

- [ ] **Step 1: Extend the failing test**

```python
# in tests/test_paper_build.py — change the two parametrise decorators to:
@pytest.mark.parametrize("doc", DOCS)      # ["paper", "report", "slides"]

# and add:
def test_report_is_substantially_longer_than_the_paper():
    from pypdf import PdfReader
    paper = len(PdfReader(str(ROOT / "paper" / "main.pdf")).pages)
    report = len(PdfReader(str(ROOT / "report" / "main.pdf")).pages)
    assert 8 <= paper <= 12, f"paper is {paper} pages, target 8-10"
    assert report >= 25, f"report is {report} pages, target 30-40"

def test_slide_count_fits_a_fifteen_minute_talk():
    from pypdf import PdfReader
    n = len(PdfReader(str(ROOT / "slides" / "main.pdf")).pages)
    assert 18 <= n <= 34, f"{n} slides; target ~20 plus backup"

def test_slides_carry_claim_titles_not_topic_labels():
    body = "\n".join(p.read_text() for p in (ROOT / "slides" / "sections").glob("*.tex"))
    topic_only = ["{Results}", "{Agreement Results}", "{Methodology}", "{Conclusion}"]
    for bad in topic_only:
        assert f"\\frametitle{bad}" not in body, f"frametitle {bad} states a topic, not a claim"
```

Add `pypdf` to the environment: `uv pip install --python .venv/bin/python "pypdf>=4"` then refresh `requirements.lock.txt`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_paper_build.py -v`
Expected: FAIL — `report/main.tex` and `slides/main.tex` do not exist.

- [ ] **Step 3: Write `report/main.tex`**

```latex
\documentclass[11pt,a4paper,oneside]{report}
\usepackage[margin=1in]{geometry}
\usepackage{setspace}\onehalfspacing
\input{../paper/preamble}
\usepackage[numbers,sort&compress]{natbib}
\graphicspath{{../figures/}}

\title{SHAP versus LIME on Clinical Tabular Data:\\
A Reproducible Benchmark of Explanation Agreement,\\
Stability, Fidelity, and Rank-Level Consensus}
\author{Samira Oasela Antika \and Yeasmin Akter Shuchona \and
        MD.\ Marajul Haque \and Gian Juti Tripura}
\date{Department of Computer Science and Engineering\\
University of Chittagong\\[1em]
Supervisor: Dr.\ M.\ Jamshed Alam Patwary\\[1em]\today}

\begin{document}
\maketitle
\tableofcontents
\listoffigures
\listoftables
\input{sections/00_abstract}
\input{sections/01_introduction}
\input{sections/02_literature_review}
\input{sections/03_methodology}
\input{sections/04_results}
\input{sections/05_discussion}
\input{sections/06_limitations}
\input{sections/07_conclusion}
\appendix
\input{appendices/A_full_results}
\input{appendices/B_reference_features}
\input{appendices/C_environment}
\input{appendices/D_repository}
\bibliographystyle{IEEEtran}
\bibliography{../paper/refs}
\end{document}
```

Chapter skeletons mirror `docs/PAPER_PLAN.md` §2. Appendix A `\input`s the full per-cell tables; Appendix B `\input`s T8; Appendix C `\input`s T12.

- [ ] **Step 4: Write `slides/main.tex`**

```latex
\documentclass[aspectratio=169,11pt]{beamer}
\usetheme{metropolis}
\usepackage{booktabs}
\usepackage{graphicx}
\graphicspath{{../figures/}}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{caption}{\insertcaption}

\title{SHAP versus LIME on Clinical Tabular Data}
\subtitle{A reproducible benchmark of agreement, stability, fidelity, and consensus}
\author{S.\ O.\ Antika \and Y.\ A.\ Shuchona \and M.\ M.\ Haque \and G.\ J.\ Tripura}
\institute{Department of CSE, University of Chittagong\\
Supervisor: Dr.\ M.\ Jamshed Alam Patwary}
\date{\today}

\begin{document}
\maketitle
\input{sections/01_problem}
\input{sections/02_question}
\input{sections/03_design}
\input{sections/04_pitfalls}
\input{sections/05_results}
\input{sections/06_limitations}
\input{sections/07_close}
\appendix
\input{sections/99_backup}
\end{document}
```

If `metropolis` is unavailable, fall back to `\usetheme{default}\usecolortheme{dove}` rather than installing a theme — check with `kpsewhich beamerthememetropolis.sty` first.

**Slide titles are claims, not topics.** `\frametitle{SHAP and LIME agree less as the task gets harder}`, never `\frametitle{Agreement Results}`. `test_slides_carry_claim_titles_not_topic_labels` enforces this.

- [ ] **Step 5: Build all three and run the tests**

Run: `make docs && .venv/bin/pytest tests/test_paper_build.py -v`
Expected: three PDFs; 8 passed.

- [ ] **Step 6: Commit**

```bash
git add report slides tests/test_paper_build.py requirements.lock.txt
git commit -m "feat: thesis report and Beamer deck skeletons, all three docs compiling

Slide titles must state a claim rather than name a topic, enforced by a test:
'SHAP and LIME agree less as the task gets harder' tells the audience the finding,
'Agreement Results' makes them wait for it."
```

---

### Task 22: The FULL run

**Files:**
- Modify: `docs/PROTOCOL.md` §10 if any pre-registered fallback fired
- Writes: all of `results/`, `figures/`, `tables/` at reporting quality

- [ ] **Step 1: Confirm the whole suite is green before spending two hours**

Run: `.venv/bin/pytest -q`
Expected: every test passing. Do not start the FULL run otherwise.

- [ ] **Step 2: Clear QUICK artefacts so nothing stale survives**

```bash
rm -rf results figures tables
```

- [ ] **Step 3: Launch the FULL run with a log**

```bash
XAIB_PROFILE=FULL nohup make full > logs/full_run.log 2>&1 &
```

Expected: ~2 hours. Monitor with `tail -f logs/full_run.log`. Stage 2 prints one line per cell, so progress is visible and a stalled cell is obvious.

- [ ] **Step 4: Verify the run's integrity before believing any of it**

```bash
.venv/bin/python - <<'PY'
import json, pandas as pd
from xaibench.config import paths, CELLS
p = paths()
cfg = pd.read_csv(p.results / "explainer_config.csv")
print("cells:", len(cfg), "expected:", len(CELLS))
print("max additivity error:", cfg.additivity_error.max(), "(must be < 5e-3)")
print("profiles:", cfg.profile.unique())
budget = pd.read_csv(p.results / "instance_budget.csv")
print(budget[["dataset", "model", "n_test", "n_explained"]].to_string(index=False))
m = json.loads((p.results / "MANIFEST.json").read_text())
print("commit:", m["git_commit"], "duration_s:", round(m["duration_s"]))
print("artifacts:", len(m["artifacts"]))
PY
```

Checks that must pass: 18 cells; `additivity_error < 5e-3` everywhere; `profile == FULL` everywhere; explained counts matching the protocol (pima 150, heart 61, ckd 80); no figure carrying the QUICK watermark.

- [ ] **Step 5: If a pre-registered fallback fired, log the amendment**

If any cell fell back to log-odds space (protocol R1), append a row to `docs/PROTOCOL.md` §10 stating the date, the affected cells, and the reason, and commit that separately before writing prose. The amendment log existing and being used is the honest outcome; a silent edit is not.

- [ ] **Step 6: Rebuild documents against real numbers and commit results**

```bash
make docs
git add -f results figures tables logs/full_run.log
git commit -m "results: FULL profile run, $(date +%Y-%m-%d)

18 cells, additivity verified in probability space for every backend. Artefact
checksums, seeds, hardware, and resolved library versions in results/MANIFEST.json."
```

`-f` is needed because `.gitignore` excludes generated directories; the reported run is committed deliberately so the PDFs and the numbers behind them travel together.

---

### Task 23: Write the conference paper

**Files:**
- Modify: every file in `paper/sections/`, plus the abstract in `paper/main.tex`

- [ ] **Step 1: Read the actual results before writing a word about them**

```bash
.venv/bin/python - <<'PY'
import pandas as pd
from xaibench.config import paths
p = paths()
h = pd.read_csv(p.results / "stats_hypotheses.csv")
print(h.groupby(["hypothesis", "decision"]).size().to_string())
print()
for name in ["agreement", "fidelity", "runtime", "plausibility"]:
    df = pd.read_csv(p.results / f"metrics_{name}.csv")
    num = df.select_dtypes("number").columns
    print(f"--- {name} ---")
    print(df.groupby("explainer")[list(num)].median().round(3).to_string()
          if "explainer" in df else df[list(num)].median().round(3).to_string())
PY
```

Write the findings down before drafting. If H6 was not rejected, protocol R4 applies and the discussion argues the null result from RQ1's agreement level — the paper's structure does not change.

- [ ] **Step 2: Write §I Introduction**

Four moves: high-stakes deployment needs explanations; SHAP and LIME are the defaults and rest on different theory; existing comparisons hold too much fixed; three numbered contributions phrased so a reader can verify each from a named table. Close with a paper roadmap.

- [ ] **Step 3: Write §II Related Work**

Four threads per `docs/PAPER_PLAN.md`, closing with T11's positioning table. State explicitly that explanation aggregation is prior art and cite it — that sentence is what makes contribution 3 defensible instead of attackable.

- [ ] **Step 4: Write §III Methodology, including §III-F**

§III-A datasets and semantics fixes; §III-B the two-track protocol and the leak it prevents; §III-C leakage-safe preprocessing; §III-D explainer configuration; §III-E metric definitions with equations; §III-G the statistical plan.

**§III-F "Methodological considerations often overlooked"** is the half-page that distinguishes this paper. Four subsections, each naming a pitfall and how it was handled:

1. *Attribution-space comparability.* LIME explains probability; tree SHAP defaults to log-odds margin. Report the additivity check and its measured maximum error.
2. *One-hot aggregation.* Fragmented categoricals otherwise put different models on different feature vocabularies.
3. *The random-ranking control.* AOPC has no absolute scale.
4. *Determinism declared a priori.* Tree SHAP's zero re-run variance is a property of the backend, stated before the experiment, not a discovery.

Add Algorithm 1 for the three fusion rules using `algorithm2e`.

- [ ] **Step 5: Write §IV Results, one subsection per RQ**

Each opens with the finding in one sentence, then the figure, then the test. Every claim reports effect size and CI before its *p*-value. Close with T10, the hypothesis decision table.

- [ ] **Step 6: Write §V Discussion, §VI Limitations, §VII Conclusion, and §VIII back matter**

Discussion: what the measured agreement level means for a practitioner choosing one method; the reproducible-versus-robust distinction from RQ2; when LIME's extra cost is justified; what the fusion outcome implies. Limitations: the seven items in `docs/PAPER_PLAN.md` §1 row VI, stated in the paper's own voice before a reader can raise them. Back matter: ethics statement (public de-identified benchmark data, no medical claims), reproducibility statement with the repository and commit hash from `MANIFEST.json`, and the author contributions paragraph from `docs/CONTRIBUTIONS.md`.

- [ ] **Step 7: Write the abstract last, with real numbers**

Structured in six sentences: context, gap, what was done, the three headline numbers, the fusion outcome, and the reproducibility claim.

- [ ] **Step 8: Build and verify**

Run: `make paper && .venv/bin/pytest tests/test_paper_build.py -v`
Expected: 8–10 pages, no undefined references, no hand-typed decimals.

- [ ] **Step 9: Commit**

```bash
git add paper && git commit -m "docs: complete conference paper prose against FULL results"
```

---

### Task 24: Write the thesis report

**Files:**
- Modify: every file in `report/sections/` and `report/appendices/`

- [ ] **Step 1: Expand Chapters 1 and 2**

Chapter 1 adds a problem statement, explicit objectives mapped to the six RQs, scope and delimitations, and a thesis-structure section. Chapter 2 becomes a full literature review of roughly six pages, organised by the four threads, with a synthesis subsection stating what the surveyed work leaves open.

- [ ] **Step 2: Expand Chapter 3**

The complete preprocessing specification, per-dataset missingness tables, the full hyperparameter grids (T2), and a notation table. Nothing is deferred to "see the paper".

- [ ] **Step 3: Expand Chapter 4 to report every cell**

All 18 cells per metric family rather than summaries, with the per-cell tables in Appendix A and the summaries in the chapter body.

- [ ] **Step 4: Write Chapter 6 including a reflection section**

Beyond the paper's limitations, add "What we would do differently": the retraining-stability scope restriction, the plausibility reference sets, the off-manifold baseline problem, and what a human evaluation would have added. This section is expected in an undergraduate thesis and is a natural place to show judgement rather than defensiveness.

- [ ] **Step 5: Fill the four appendices**

A: full per-cell results. B: reference feature sets with citations (T8). C: environment and library versions (T12) plus the hardware line from `MANIFEST.json`. D: repository map and step-by-step rerun instructions.

- [ ] **Step 6: Build and verify**

Run: `make report && .venv/bin/pytest tests/test_paper_build.py -v`
Expected: 30–40 pages, all cross-references resolved.

- [ ] **Step 7: Commit**

```bash
git add report && git commit -m "docs: complete thesis report with appendices and reflection"
```

---

### Task 25: Write the slide deck

**Files:**
- Modify: every file in `slides/sections/`

- [ ] **Step 1: Write the 20 content slides per `docs/PAPER_PLAN.md` §3**

One message per slide, stated as a claim in the `\frametitle`. Figures regenerated at slide aspect ratio from the same code, never scaled from the paper.

- [ ] **Step 2: Make slide 9 the credibility slide**

The four pitfalls from §III-F. This is the slide that shows the work understood its method rather than just ran a library, and it is the one most likely to earn marks in a technical-communication course.

- [ ] **Step 3: Write the backup slides**

Per-dataset breakdowns, both sensitivity checks, full metric definitions, the runtime table, and one slide per anticipated hard question so an answer can be shown rather than described.

- [ ] **Step 4: Add presenter notes**

`\note{...}` on each frame with the intended spoken content, compiled to a separate handout via `\setbeameroption{show notes}` in a second target. These become the raw material for `docs/PRESENTATION_GUIDE.md`.

- [ ] **Step 5: Build and verify**

Run: `make slides && .venv/bin/pytest tests/test_paper_build.py -v`
Expected: 18–34 pages; no topic-only frame titles.

- [ ] **Step 6: Commit**

```bash
git add slides && git commit -m "docs: complete Beamer deck with presenter notes and backup slides"
```

---

### Task 26: README, presentation guide, and clean-clone verification

**Files:**
- Create: `README.md`, `docs/PRESENTATION_GUIDE.md`, `CITATION.cff`, `LICENSE`
- Modify: `docs/CONTRIBUTIONS.md` if the team adjusts the split

- [ ] **Step 1: Write `README.md`**

Sections: what this is and what it claims; the four deliverables with paths; a five-line quick start (`make setup && make data && make test && make smoke`, then `make all`); the repository map; where each figure and table comes from; the runtime expectation for each profile; the pre-registration story with the protocol's freeze commit hash; dataset provenance and licences; how to cite; and the four authors plus advisor.

- [ ] **Step 2: Write `docs/PRESENTATION_GUIDE.md`**

- A per-slide talking track, generated from the Beamer `\note{}` content, with timing for a 12–15 minute talk.
- **Twenty-five anticipated questions with answers**, grouped so each author owns the block matching their row in `docs/CONTRIBUTIONS.md`. Cover at minimum:
  - "What is novel here?" — the honest answer: a controlled benchmark, a stability decomposition, and an evaluation of fusion rules; aggregation itself is prior art and is cited.
  - "Why not just use SHAP, since it is theoretically grounded?" — cost, the model-agnostic case, and the measured agreement level.
  - "Your SHAP stability result is trivially zero." — declared a priori in the protocol; the informative measurements are 2b and 2c.
  - "Isn't median imputation for deletion off-manifold?" — yes; the marginal-draw sensitivity variant, and what it showed.
  - "Three datasets is not many." — agreed; the difficulty gradient is the design compensation, and it is named as a limitation.
  - "How do you know your reference feature sets are right?" — we do not; RQ5 is a plausibility check, pre-registered and cited, and labelled the weakest metric.
  - "Did you correct for multiple comparisons?" — Holm within family; both raw and adjusted *p* reported.
  - "What if the fusion result is null?" — pre-registered as valid; the explanation follows from RQ1.
  - "Can I reproduce this?" — `make all`, pinned lockfile, checksums, per-artefact SHA-256.
- A pre-talk checklist: rebuild the PDFs, confirm the figures carry no QUICK watermark, verify the commit hash in the paper matches `HEAD`, and rehearse the four-pitfall slide.

- [ ] **Step 3: Write `CITATION.cff` and `LICENSE`**

MIT for the code. State in `README.md` that the datasets keep their own licences (CC0 for PIMA, CC BY 4.0 for the two UCI sets) and are redistributed under those terms.

- [ ] **Step 4: Verify reproducibility from a clean clone**

```bash
cd /tmp && rm -rf repro-check && git clone /home/marajul/AI_WorkShop/Undergrade_Thesis repro-check
cd repro-check && make setup && make data && make test && make smoke && make docs
```

Expected: every test passes and all three PDFs build in a directory that has never held a QUICK artefact. This is the check that turns the reproducibility claim from an assertion into a demonstrated fact — and the one that catches an accidental dependency on an untracked file.

- [ ] **Step 5: Final consistency sweep**

```bash
grep -rn "IEEE Access\|novel method\|we prove\|proves that" paper/ report/ slides/ --include=*.tex
grep -rln "QUICK" figures/ 2>/dev/null
.venv/bin/pytest -q
```

Expected: no overclaiming language outside the §7 change log in `thesis_overview.md`; no QUICK watermark in any reported figure; the suite green.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/PRESENTATION_GUIDE.md CITATION.cff LICENSE
git commit -m "docs: README, presentation guide with 25 anticipated questions, licensing

Reproducibility verified from a clean clone in /tmp: setup, data, tests, smoke
pipeline, and all three document builds, in a directory that never held an
artefact from development."
```

---

## Self-Review

**Spec coverage.** Every section of the design spec and the protocol maps to a task: datasets and semantics fixes → T2; leakage-safe preprocessing and one-hot aggregation → T3; the six models and backend assignment → T4, T10; all 16 interpretability measures → T5–T7, T13; the 11 hypotheses with Holm correction and BCa intervals → T8, T17; the two-track protocol → T14, T9; probability-space comparability → T10; the random control → T12; fusion with a leakage-safe calibration split → T12; the three stability components including the R3 scope restriction → T7, T15; QUICK/FULL profiles and the compute budget → T1, T22; 14 figures and 12 tables → T18, T19; the four deliverables → T20, T21, T23, T24, T25, T26; the manifest and determinism check → T19; pre-registered fallbacks R1–R5 → T10 step 4, T14 step 4, T22 step 5.

**Naming consistency.** `Explanation`, `Cell`, `CellArtifacts`, `Profile`, `Dataset`, `ReferenceSet`, `TestResult` are each defined in exactly one task and referenced by the same name thereafter. `explain_shap`/`explain_lime`/`build_consensus`/`random_control` all return `list[Explanation]`. `aggregate_attributions(values, groups, feature_order)` keeps that argument order everywhere. `_transformed_ranking` is defined in T15 and reused by T16, which is noted there rather than reimplemented.

**Two known plan-level risks, with their resolutions already written in.** First, `_CellView` in Task 15 substitutes a test matrix so neighbours and bootstrap refits can be explained by index; if that proxy proves brittle against `shap`'s introspection, replace it with a real `dataclasses.replace(cell, ...)` call — `Cell` is frozen, so that is a one-line change. Second, `test_the_difficulty_gradient` in Task 14 can legitimately fail if CKD is less separable than expected; that is a framing problem, not a test problem, and Task 14 step 4 says to stop and reconsider rather than adjust the assertion.
