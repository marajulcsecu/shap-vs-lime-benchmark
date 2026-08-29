# Design: XAI Explanation Benchmark (SHAP vs. LIME on Clinical Tabular Data)

**Date:** 2026-08-29 · **Course:** ENG 672 Technical Communication and Writing
**Companion documents:** `docs/PROTOCOL.md` (frozen analysis protocol), `docs/PAPER_PLAN.md`
(document outlines and figure/table manifest)

---

## 1. What we are building and why this shape

Four deliverables, one engine.

| Deliverable | Artefact | Size |
|-------------|----------|------|
| Conference paper | `paper/main.pdf` — IEEEtran, two-column | 8–10 pages |
| Thesis report | `report/main.pdf` — single column, appendices | 30–40 pages |
| Slide deck | `slides/main.pdf` — Beamer | ~20 slides + backup |
| Reproducibility package | the repository, `make all` | one command |

The engine is a five-stage pipeline that writes tidy CSV/Parquet to `results/`. The three documents
consume that directory and nothing else. This is the load-bearing architectural decision, so it is
worth stating plainly:

> **Every number in every document is `\input` from a generated file.** No result is transcribed by
> hand. If an experiment is rerun and a value changes, the paper changes on the next `make paper`.

That constraint costs a few hours of plumbing and buys three things a course in technical
communication rewards: the documents cannot silently drift from the experiments, the reproducibility
claim is demonstrable rather than asserted, and late changes to the experiments do not require
hunting for stale numbers across three documents.

### 1.1 What changed from `thesis_overview.md`, and why

The original overview is a sound topic with three claims that would not survive scrutiny. Each is
revised rather than dropped.

**Change 1 — the "novel hybrid consensus method" is reframed, not removed.** Rank-averaging two
explainers is a small idea and explanation aggregation already exists in the literature. Presenting it
as *the* novel contribution invites exactly the question the work cannot answer ("how does this differ
from prior fusion work?"). Instead: three fusion rules are implemented (mean-rank, reciprocal-rank
fusion, fidelity-weighted), evaluated against a strict pre-registered bar, and the contribution is
named as *controlled evaluation* rather than invention. This is both safer and stronger — an
evaluation claim cannot be knocked down by a citation, and a null result stays publishable.

**Change 2 — "publishable IEEE Access paper" becomes "prepared in IEEE conference format."** IEEE
Access review turns on genuine novelty. Writing that the deliverable targets it sets a bar the work
is not trying to clear, and an advisor reading that sentence is entitled to hold the work to it. The
paper uses IEEEtran and reads like a conference submission; that is what "conference paper" required.

**Change 3 — RQ2 (stability) is rebuilt, because as written it was close to vacuous.** SHAP's
TreeExplainer is deterministic: re-running it returns bit-identical attributions, so "coefficient of
variation across re-runs" is exactly 0 for four of six models, by construction, before any experiment
runs. Reporting that as a finding is a mistake. Stability is therefore split into three separable
sources — explainer randomness, input perturbation, and model retraining — of which only the first is
degenerate for tree SHAP, and it is declared degenerate *in advance* in the protocol. The retraining
component (RQ2c) is the genuinely interesting one: it asks whether an explanation that is perfectly
reproducible is nonetheless fragile because the *model* under it moves when the training sample does.

Two further additions the overview did not mention but the study needs:

**Addition 1 — a random-ranking control for fidelity.** Deletion-based AOPC has no absolute scale;
removing any 5 of 8 features moves a prediction. Without a random baseline, "SHAP achieved AOPC 0.31"
is uninterpretable. Every fidelity number is now reported against 20 averaged random rankings.

**Addition 2 — attribution-space comparability.** LIME explains predicted probability; SHAP's tree
backend defaults to log-odds margin. Comparing them directly compares different quantities on
different scales — a frequent and invalidating error in student XAI comparisons. SHAP is therefore run
with `model_output="probability"`, with a pre-registered fallback if that proves too slow.

RQ5 (plausibility against literature-derived feature sets) is kept but demoted: reference sets are
frozen and cited before results exist, the metric is labelled the weakest in the study, and it is
reported as corroborating evidence only.

### 1.2 Revised title

> **SHAP versus LIME on Clinical Tabular Data: A Reproducible Benchmark of Explanation Agreement,
> Stability, Fidelity, and Rank-Level Consensus**

Descriptive, claims nothing it cannot show, and names all six research questions.

---

## 2. Architecture

```
raw sources ──fetch+checksum──> data/raw/*.csv
                                     │
                              loaders.py  (semantics fixes, dtypes, feature groups)
                                     │
                     ┌───────────────┴────────────────┐
              Stage 1: performance            Stage 2: explain
              nested CV, 5×3                  fixed 80/20 split, one model per cell
              → results/performance.csv       → results/attributions.parquet
                                                 results/runtimes.csv
                                     │
                              Stage 3: metrics
                              agreement · stability · fidelity · plausibility · consensus
                              → results/metrics_*.csv
                                     │
                              Stage 4: statistics
                              Wilcoxon · Holm · effect sizes · BCa bootstrap CIs
                              → results/stats_*.csv
                                     │
                              Stage 5: outputs
                              → figures/*.pdf|png   tables/*.tex   results/MANIFEST.json
                                     │
                     ┌───────────────┼────────────────┐
               paper/main.tex   report/main.tex   slides/main.tex
```

Each stage is independently runnable, reads only files written by earlier stages, and is idempotent.
A stage that finds its outputs present and its inputs unchanged skips work unless `--force` is given.
This matters on a 2–3 day schedule: a figure tweak must not require re-running two hours of LIME.

### 2.1 Module layout

```
src/xaibench/
  config.py            Profile dataclass; QUICK and FULL profiles; all paths, seeds, budgets
  data/fetch.py        download → SHA-256 verify → vendor into data/raw/
  data/loaders.py      load_dataset(id) -> Dataset(X, y, meta) ; per-dataset semantics fixes
  models/pipelines.py  leakage-safe ColumnTransformer + column→feature map
  models/zoo.py        model factory, parameter grids, SHAP-backend selection per model
  explain/base.py      Explanation dataclass (attributions, ranks, runtime, config hash)
  explain/shap_backend.py   backend dispatch, background summarisation, one-hot aggregation
  explain/lime_backend.py   LimeTabularExplainer wrapper, same output contract
  explain/consensus.py      rankavg · rrf · fidelity-weighted (with calibration split)
  metrics/agreement.py      spearman · kendall_tau_b · jaccard_at_k · rbo
  metrics/stability.py      rerun_cv · local_lipschitz · bootstrap_retrain_agreement
  metrics/fidelity.py       aopc_deletion · auc_insertion · comprehensiveness · sufficiency
  metrics/plausibility.py   reference-set overlap from docs/reference_features.yaml
  stats/tests.py            wilcoxon_with_effect · holm · bca_bootstrap_ci · cliffs_delta
  stats/report.py           tidy DataFrame → booktabs LaTeX fragment
  viz/theme.py              one matplotlib style: colourblind-safe, greyscale-legible, 8pt-readable
  viz/figures.py            one function per figure, each saving .pdf and .png
  pipeline/stage{1..5}_*.py  CLI entry points
```

The interface that holds it together is a single dataclass:

```python
@dataclass(frozen=True)
class Explanation:
    dataset: str; model: str; explainer: str
    instance_id: int
    attributions: np.ndarray   # shape (d,), original-feature space, one-hot already summed
    feature_names: tuple[str, ...]
    runtime_s: float
    seed: int
    config_hash: str           # hash of explainer configuration, for provenance
```

Every explainer — including each consensus rule and the random control — returns this. Every metric
consumes it. Adding a seventh explainer means writing one function; nothing else changes. This is why
`consensus.py` is a peer of the two backends rather than a special case bolted onto the metrics.

### 2.2 Configuration profiles

Two profiles, one code path:

| Knob | QUICK (pipeline validation) | FULL (reported results) |
|------|----------------------------|-------------------------|
| instances explained per cell | 12 | ≤ 150 |
| LIME `num_samples` | 500 | 5000 (+1000 sensitivity) |
| re-run repeats R | 3 | 10 |
| robustness neighbours m | 2 | 5 |
| bootstrap refits B | 3 | 10 |
| bootstrap CI resamples | 200 | 2000 |
| target wall clock | ≈ 5 min | 1.5–2.5 h |

`make smoke` runs QUICK. Every figure and table is produced in QUICK mode too, with a visible
`PROFILE=QUICK — NOT FOR REPORTING` watermark, so the whole chain including LaTeX is validated long
before the expensive run starts.

---

## 3. Compute budget

Measured against 12 cores / 7.4 GB RAM. The datasets total ~1,470 rows, so memory is never the
constraint; the binding cost is LIME, which needs thousands of forward passes per explanation.

| Workload | Explainer calls | Estimate |
|----------|-----------------|----------|
| Stage 1 nested CV (18 cells × 5 × 3 × ≤12 grid) | — | 5–10 min |
| Stage 2 base explanations, both LIME sample sizes | ~3,500 | ~15 min |
| RQ2a re-run stability (30 inst × 18 cells × R=10) | ~5,400 | ~30 min |
| RQ2b local robustness (25 inst × m=5 × 18 cells) | ~2,250 | ~15 min |
| RQ2c retraining stability (25 inst × B=10 × 9 cells) | ~2,250 | ~20 min |
| SHAP `PermutationExplainer` on `stack` incl. repeats | ~1,200 | ~20 min |
| Stage 3 fidelity (batched `predict_proba`, no explainer calls) | — | ~3 min |
| Stages 4–5 statistics, figures, tables | — | ~5 min |
| **Total** | | **≈ 2 h** |

Three deliberate scope reductions keep it there, all disclosed in the protocol rather than hidden:
RQ2c covers the three ensemble models only; robustness uses 25 instances rather than the full test
partition; and the LIME discretisation sensitivity check runs on one dataset.

---

## 4. Testing strategy

Test-driven where correctness is checkable against something other than the code's own output. Metric
implementations qualify; pipeline orchestration does not.

`tests/test_agreement.py` — identical rankings → ρ = 1, τ = 1, Jaccard = 1, RBO = 1; reversed → ρ = −1;
RBO against the worked examples in Webber et al. (2010); Jaccard@k on hand-computed sets.

`tests/test_fidelity.py` — on a synthetic linear model `f(x) = 3x₀ + 2x₁ + 0·x₂` with known coefficients,
the true importance order is (x₀, x₁, x₂) and AOPC must be maximised by that ordering and minimised by
its reverse. Insertion and deletion curves must be monotone for this model. This catches sign errors
and off-by-one indexing in the perturbation loop, which is where fidelity code actually breaks.

`tests/test_consensus.py` — fusing a ranking with itself is the identity; fidelity weights sum to 1;
w → 1 recovers SHAP exactly; the calibration and evaluation instance sets are provably disjoint.

`tests/test_pipelines.py` — one-hot aggregation: a synthetic frame with a 3-level categorical must
produce attributions in original-feature space whose length equals the original feature count, and the
aggregated value must equal the sum of its one-hot columns. Also asserts no transformer is fitted
outside a fold (checked by fitting on a fold and asserting the imputer's learned statistics differ
between folds).

`tests/test_determinism.py` — the full QUICK profile run twice from the same seed produces
byte-identical `results/*.csv` (excluding recorded wall-clock columns). This is the test that keeps
the reproducibility claim honest.

---

## 5. Failure modes and how the design absorbs them

| Risk | Mitigation |
|------|-----------|
| Probability-space TreeSHAP too slow on some cell | Pre-registered fallback to log-odds for that cell, recorded per cell in `results/explainer_config.csv`, disclosed in the paper; spaces never pooled in one comparison |
| CKD is near-separable (AUC ≈ 0.99) → degenerate explanations | Treated as a finding about easy tasks, not a defect; the difficulty gradient across the three datasets is what H1c tests |
| No fusion rule beats max(SHAP, LIME) | Pre-registered as a valid outcome; paper structure and contribution list do not depend on rejecting H6 |
| LightGBM warns/overfits on <1000 rows | `num_leaves=15`, small `min_child_samples`, `verbose=-1`, tuned inside the inner CV loop |
| Disk at 95 % (26 GB free) | Parquet not CSV for attributions; no pickled model zoo beyond the 18 fitted cells; venv ≈ 1.5 GB; artefacts total < 200 MB |
| A two-hour run dies at 90 % | Stages are separately resumable and idempotent; Stage 2 checkpoints per cell |
| Overclaiming in the write-up | Explicit "Limitations and Threats to Validity" section; ethics statement; no medical claims; the word "prove" is not used |

---

## 6. Order of work

1. Scaffold repo, `uv` venv, pin dependencies, `Makefile`, matplotlib theme.
2. `data/fetch.py` + checksums + `loaders.py`; assert realised shapes match the protocol's Table T1.
3. Metric unit tests first, then metric implementations, until tests pass.
4. Stage 1 (performance) → sanity-check ROC-AUC against the expected difficulty gradient.
5. Stage 2 (explain) under QUICK; verify the `Explanation` contract and one-hot aggregation.
6. Stages 3–5 under QUICK; produce every figure and table with the QUICK watermark.
7. Build `paper`, `report`, `slides` skeletons against QUICK numbers so LaTeX is proven end to end.
8. Run FULL. Regenerate. Diff figures for surprises.
9. Write prose into the three documents against real numbers.
10. `docs/PRESENTATION_GUIDE.md`, `docs/CONTRIBUTIONS.md`, final reproducibility verification from a
    clean clone.

Steps 1–7 deliberately precede the expensive run. On a 2–3 day schedule, discovering a LaTeX or
plumbing problem after the two-hour experiment is the failure that costs a day.

---

## 7. Explicitly out of scope

Deep learning or TabNet-style models; image or text modalities; counterfactual explainers (DiCE),
Anchors, or attention-based methods; human-subject evaluation of explanation usefulness; causal
discovery; hyperparameter search beyond the small pre-specified grids; any clinical validation.

Each is named in the paper's future-work section, which is cheaper and more honest than half-doing one.
