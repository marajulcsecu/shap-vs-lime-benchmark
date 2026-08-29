# Document Plan: Paper, Report, Slides

All three documents draw every number from `results/`, every figure from `figures/`, and every table
from `tables/*.tex` via `\input`. Nothing is transcribed by hand.

---

## 1. Conference paper — `paper/main.pdf`

IEEEtran, two-column, 8–10 pages including references. Target section budget:

| § | Section | Pages | Content |
|---|---------|-------|---------|
| I | Introduction | 1.0 | Black-box models in high-stakes settings; SHAP and LIME as the two dominant post-hoc methods; the gap: comparisons are usually single-model, single-dataset, and metric-inconsistent. Contributions listed as three numbered claims (benchmark, stability decomposition, fusion evaluation) — each phrased so it is verifiable from the results. |
| II | Related Work | 1.0 | Four threads: (a) SHAP and LIME foundations; (b) prior SHAP-vs-LIME comparisons and what each held fixed; (c) explanation evaluation metrics — fidelity, stability, robustness; (d) explanation aggregation, cited explicitly so our fusion work is positioned as evaluation, not invention. Closes with a table positioning this study against 5–7 prior comparisons on: #models, #datasets, #metrics, statistical testing, code released. |
| III | Methodology | 2.5 | Datasets and semantics fixes; the two-track protocol (nested CV for performance, fixed split for explanations) and why they are separated; leakage-safe preprocessing; explainer configuration; **§III-F "Methodological considerations often overlooked"** — attribution-space comparability, one-hot aggregation, the random-ranking control, and SHAP's a-priori determinism. Metric definitions with equations. Statistical plan: non-parametric tests, Holm correction, effect sizes, BCa CIs. Algorithm 1: the three fusion rules. |
| IV | Results | 2.5 | One subsection per RQ, each opening with the finding in one sentence, then the figure, then the test. |
| V | Discussion | 1.0 | What the agreement level means for practice; the stability decomposition's implication (reproducible ≠ robust); when the extra cost of LIME is or is not justified; what a fusion result (positive or null) implies. |
| VI | Limitations and Threats to Validity | 0.5 | Three small datasets; tabular only; plausibility reference sets are not ground truth; median-imputation baselines are off-manifold; single hardware for timing; RQ2c restricted to ensembles; no human evaluation. |
| VII | Conclusion | 0.3 | Three sentences, no new claims. |
| — | Ethics, reproducibility, author contributions, references | 0.7 | Public de-identified benchmark data; no medical claims; repository and commit hash; CRediT-style contributions for four authors. |

**§III-F is the section that distinguishes this paper.** Naming the four pitfalls and showing they were
handled demonstrates command of the method — which is precisely what a technical-communication course
grades — and costs half a page.

---

## 2. Thesis report — `report/main.pdf`

Single column, 30–40 pages. Same spine, expanded, plus:

- Chapter 2 becomes a full literature review (~6 pages) rather than a compressed one.
- Chapter 3 adds the complete preprocessing specification, per-dataset missingness tables, and the
  full hyperparameter grids.
- Chapter 4 reports every cell rather than summaries: all 18 (dataset × model) cells per metric.
- Chapter 6 adds a reflective "what we would do differently" section — expected in an undergraduate
  thesis and a natural place to show judgement.
- Appendix A: full per-cell result tables. Appendix B: reference feature sets with citations.
  Appendix C: environment and library versions. Appendix D: repository map and rerun instructions.

The report shares `sections/` fragments with the paper where the text is identical, so corrections
propagate rather than diverge.

---

## 3. Slides — `slides/main.pdf`

Beamer, ~20 slides for a 12–15 minute talk, plus backup slides for questions.

1. Title, authors, affiliation, advisor
2. The problem in one picture — an accurate model whose reasoning is opaque
3. Why two explainers disagree at all (SHAP = game theory, LIME = local surrogate)
4. The question, in one sentence
5. Contributions (3 bullets)
6. Study design diagram (F1)
7. Datasets and the difficulty gradient (T1 condensed)
8. The two-track protocol, and the leak it prevents
9. Four pitfalls we had to handle (the §III-F content — the slide that earns credibility)
10. Model performance sanity check (F3)
11. RQ1 agreement — heatmap (F4)
12. RQ1 agreement vs. difficulty (F6)
13. RQ2 stability, three kinds — the "reproducible ≠ robust" slide (F8, F9)
14. RQ3 fidelity with the random control (F10)
15. RQ4 cost, and the fidelity-per-second frontier (F11)
16. RQ5 plausibility, with its caveat stated on the slide
17. RQ6 fusion result (F13)
18. Hypothesis decision table (T10)
19. Limitations — said plainly, before anyone asks
20. Conclusion and reproducibility (repo, one command)

Backup: per-dataset breakdowns, the sensitivity checks, full metric definitions, runtime table.

**Slide discipline:** one message per slide, stated in the title as a claim rather than a topic
("SHAP and LIME agree less as the task gets harder", not "Agreement Results"). Every figure is
regenerated at slide aspect ratio from the same code, not scaled from the paper.

---

## 4. Figure manifest

Each figure is one function in `viz/figures.py`, saved as `.pdf` (documents) and `.png` (preview).
Shared theme: colourblind-safe palette, legible in greyscale, minimum 8 pt effective font size,
no chartjunk, units and *n* on every axis.

| ID | Figure | Type | Used in |
|----|--------|------|---------|
| F1 | Study design and data flow | schematic | paper, report, slides |
| F2 | Dataset characteristics: class balance + missingness | 2-panel bar + heatmap | report, slides backup |
| F3 | Model performance across datasets (ROC-AUC, MCC, nested-CV error bars) | grouped bar | paper, report, slides |
| F4 | RQ1 global agreement, Spearman ρ per dataset × model | annotated heatmap | paper, report, slides |
| F5 | RQ1 local agreement distributions per model, faceted by dataset | violin | paper, report |
| F6 | RQ1 agreement vs. task difficulty (ρ against 1 − AUC) | scatter + fit + CI | paper, report, slides |
| F7 | RQ2a LIME variance vs. `num_samples` | line, log-x | report, slides backup |
| F8 | RQ2b local Lipschitz estimates, SHAP vs. LIME | paired dumbbell | paper, report, slides |
| F9 | RQ2c retraining rank stability, three ensembles | grouped bar + CI | paper, report, slides |
| F10 | RQ3 deletion curves: SHAP, LIME, best fusion, random control | 3×2 faceted lines | paper, report, slides |
| F11 | RQ4 runtime distributions (log) + fidelity-per-second frontier | 2-panel box + scatter | paper, report, slides |
| F12 | RQ5 plausibility overlap vs. reference sets | grouped bar | report, slides |
| F13 | RQ6 fusion minus best-single, per cell, with bootstrap CIs | forest plot | paper, report, slides |
| F14 | Summary: mean rank of each explainer across all metrics | critical-difference style | report |

## 5. Table manifest

Generated as `booktabs` fragments into `tables/`.

| ID | Table | Used in |
|----|-------|---------|
| T1 | Dataset summary: n, features, positive rate, missingness, source, licence | paper, report |
| T2 | Hyperparameter search spaces per model | report |
| T3 | Predictive performance, 8 metrics × 6 models × 3 datasets (nested CV mean ± std) | paper (condensed), report (full) |
| T4 | RQ1 agreement summary: ρ, τ, Jaccard@3/@5, RBO with 95 % CIs | paper, report |
| T5 | RQ2 stability summary across all three components | paper, report |
| T6 | RQ3 fidelity: AOPC, AUC-ins, comprehensiveness, sufficiency, + random control | paper, report |
| T7 | RQ4 runtime: median and IQR seconds per explanation | paper, report |
| T8 | RQ5 reference feature sets with citations, plus overlap scores | report |
| T9 | RQ6 fusion results: Δ vs. best single, p_adj, effect size, 95 % CI | paper, report |
| T10 | **Hypothesis decision table** — every H: test, statistic, n, p, p_adj, effect size, CI, decision | paper, report, slides |
| T11 | Positioning against prior SHAP-vs-LIME comparisons | paper, report |
| T12 | Environment and library versions | report appendix |

T10 is the paper's centrepiece: eleven hypotheses, each with its pre-registered test and its
Holm-adjusted decision, in one table a reader can audit against `docs/PROTOCOL.md`.

---

## 6. Writing standards applied throughout

Because the course is technical communication, the prose is a graded artefact, not packaging:

- Every claim is either cited or traceable to a numbered result. No unsupported assertion.
- Results are stated as effect size + interval first, *p*-value second. "Significant" never appears
  without its effect size in the same sentence or row.
- The word "prove" is not used; "suggest", "is consistent with", and "we did not detect" are.
- Figure captions are self-contained: a reader who reads only captions gets the argument.
- Every acronym expanded on first use; a notation table in the report.
- Passive voice avoided where an actor exists; methods sections say what we did.
- Limitations are stated before a reader can raise them, in the paper's own voice.
- References in IEEE numeric style via `IEEEtran.bst`, ~30–40 entries for the paper, ~55–70 for the report.
