# Pre-Registered Analysis Protocol

**Study:** SHAP vs. LIME on clinical tabular data — agreement, stability, fidelity, cost,
plausibility, and rank-level consensus.

**Version:** 1.0 · **Frozen:** 2026-08-29 · **Status:** pre-registered

> **Why this document exists.** It fixes every research question, hypothesis, metric formula,
> statistical test, significance level, and correction procedure *before any result is computed*.
> It is committed to git in its own commit, before any commit that contains results, so the
> repository history is itself evidence that the analysis was not adjusted after seeing outcomes.
> Any change after freezing is recorded in §10 (Amendment log) with a reason — never by silent edit.

---

## 1. Scope, framing, and what we do *not* claim

This is a **computer-science methodology study**. The three datasets are *benchmarks* chosen because
they are small, public, tabular, and heterogeneous in difficulty. We make **no medical or clinical
claims**. No model produced here is fit for diagnostic use, and no feature ranking should be read as
a causal or clinical statement.

**Contribution claims, stated precisely:**

1. **A reproducible benchmark** comparing two post-hoc explainers across 6 classifiers × 3 datasets
   under a single leakage-safe protocol, with every reported number regenerated from code.
2. **A decomposition of explanation stability** into three separable sources — explainer randomness,
   input perturbation, and model retraining variance — measured on the same models.
3. **A controlled evaluation of three explanation-fusion rules**, including a fidelity-weighted
   variant, against a pre-registered bar (must beat the better of the two base explainers).

**What we explicitly do not claim.** We do not claim priority for explanation aggregation; combining
explainer outputs is an existing idea and we cite prior work. Our contribution on that point is
*evaluation under controlled conditions*, not invention. We do not claim novelty of the metrics; all
are drawn from published literature and cited. A null result on Contribution 3 is a valid and
reportable outcome, and is pre-registered as such (§6, H6).

---

## 2. Objects of study

### 2.1 Datasets

| ID | Dataset | Source (retrieved 2026-08-29) | n | Features | Positive rate | Role |
|----|---------|-------------------------------|---|----------|---------------|------|
| `pima` | PIMA Indian Diabetes | plotly/datasets `diabetes.csv` (orig. UCI/NIDDK) | 768 | 8 numeric | 34.9 % | hard task |
| `heart` | Heart Disease (Cleveland) | UCI id 45, `static/public/45/data.csv` | 303 | 13 mixed | ~45.9 % | medium task |
| `ckd` | Chronic Kidney Disease | UCI id 336, `static/public/336/data.csv` | 400 | 24 mixed | 62.5 % | easy task |

The three datasets deliberately span a **difficulty gradient** (expected ROC-AUC roughly 0.83 / 0.90 /
0.99). This is not incidental: it lets us ask whether explainer agreement degrades as the decision
boundary gets harder — a question a single-dataset study cannot pose.

Dataset-specific handling, fixed in advance:

- **`pima`** — zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` are
  biologically impossible and are recoded to missing before any modelling. `Pregnancies` and
  `Outcome` zeros are legitimate and left alone. This is a data-semantics correction, applied at load
  time, not a learned transform, so it is not fold-dependent.
- **`heart`** — target `num` ∈ {0,1,2,3,4} is binarised to `num > 0` (presence of disease), the
  standard treatment. `ca` and `thal` carry a small number of missing values.
- **`ckd`** — `class` ∈ {ckd, notckd} → {1, 0}. Source strings carry stray whitespace and tab
  characters; all object columns are stripped and normalised at load. Missingness is heavy and
  reported per column in Table T1.

Every dataset is downloaded by script, checksummed (SHA-256), and vendored under `data/raw/` so the
pipeline runs offline afterwards. Checksums are recorded in `data/raw/checksums.json` and verified on
every run; a mismatch is a hard failure, not a warning.

### 2.2 Models

Six classifiers, spanning linear → single tree → bagged → boosted → stacked:

| ID | Model | Library | SHAP explainer |
|----|-------|---------|----------------|
| `logreg` | Logistic Regression (L2) | scikit-learn | `LinearExplainer` |
| `dtree` | Decision Tree | scikit-learn | `TreeExplainer` |
| `rf` | Random Forest | scikit-learn | `TreeExplainer` |
| `xgb` | XGBoost | xgboost | `TreeExplainer` |
| `lgbm` | LightGBM | lightgbm | `TreeExplainer` |
| `stack` | Stacking (LR meta-learner over rf + xgb + lgbm) | scikit-learn | `PermutationExplainer` |

`logreg` and `dtree` are included as interpretable reference points, not as competitive baselines.
`stack` is included specifically because it forces a **model-agnostic** SHAP backend, which — unlike
TreeExplainer — is stochastic, making it the one model family where SHAP's own re-run stability is a
non-trivial measurement.

### 2.3 Explainers

| ID | Method | Configuration (primary) | Sensitivity variant |
|----|--------|-------------------------|---------------------|
| `shap` | SHAP | probability-space output, interventional perturbation, background = 50 k-means centroids of train | log-odds space (fallback, see §8 R1) |
| `lime` | LIME | `LimeTabularExplainer`, `discretize_continuous=True`, `num_samples=5000`, all features | `num_samples=1000`; `discretize_continuous=False` |
| `rankavg` | Consensus: mean of normalised ranks (Borda) | unweighted | — |
| `fidwt` | Consensus: fidelity-weighted rank fusion | weights from calibration split | — |
| `rrf` | Consensus: reciprocal rank fusion, `1/(60 + rank)` | standard IR constant | — |
| `random` | Control: uniformly random feature ranking | 20 draws, averaged | — |

**Comparability requirement (non-negotiable).** SHAP and LIME must explain the *same quantity*.
LIME explains predicted class probability. TreeExplainer defaults to margin/log-odds space, so a
naive comparison contrasts log-odds attributions against probability attributions — a common and
invalidating error. We therefore run SHAP with `model_output="probability"` throughout. If this
proves computationally infeasible for a given model, the documented fallback in §8 R1 applies and the
switch is recorded in the results manifest, not hidden.

**One-hot aggregation requirement.** Categorical features are one-hot encoded, so a single original
feature may span several model inputs. Attributions are summed over the columns belonging to each
original feature before any ranking or comparison, so all rankings are over the same
original-feature vocabulary across models and explainers. The column→feature map is emitted with the
processed data and unit-tested.

---

## 3. Experimental protocol

Two deliberately separated tracks. Conflating them is the standard way explanation studies leak.

**Track A — predictive performance.** Nested stratified cross-validation: outer 5-fold for unbiased
performance estimation, inner 3-fold for hyperparameter selection. All preprocessing (imputation,
scaling, encoding) lives inside a `Pipeline` fitted within each fold. Reports mean ± std of
Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, Brier score.

**Track B — explanation analysis.** A single stratified 80/20 train/test split (`seed=42`).
Hyperparameters are taken from Track A's inner-loop winners; the model is refit once on the 80 %
train partition. All explainers operate on this one fixed model per (dataset, model) cell. Rationale:
explanation analysis needs *one* model per cell, not five fold-models, and the test partition must
never have participated in tuning.

**No resampling / no SMOTE.** Class imbalance is mild (34.9–62.5 % positive). Synthetic oversampling
would alter the data manifold that both LIME's neighbourhood sampler and SHAP's background
distribution draw from, confounding the very comparison under study. We use
`class_weight="balanced"` where the estimator supports it and report MCC and PR-AUC alongside
accuracy. This decision is recorded here so it reads as a design choice, not an oversight.

**Instance budget.** Explanations are computed for a stratified sample of up to `N_explain = 150`
test instances per cell, or the whole test partition when smaller. Expected: `pima` 150 of 154,
`heart` all 61, `ckd` all 80. Exact realised counts are emitted to `results/instance_budget.csv`.

**Randomness.** Global seed 42. Repeat *r* uses seed `1000 + r`. Bootstrap resample *b* uses seed
`2000 + b`. Every seed used is written to the results manifest.

---

## 4. Research questions

| # | Question | Primary metric |
|---|----------|----------------|
| RQ1 | To what extent do SHAP and LIME produce concordant feature rankings, and does concordance vary with model family and task difficulty? | Spearman ρ (local, per instance) |
| RQ2 | How stable are the explanations, separating explainer randomness (2a), input perturbation (2b), and model retraining (2c)? | CV; local Lipschitz estimate; cross-resample rank agreement |
| RQ3 | Do the explanations faithfully reflect the model's decision function, relative to a random-ranking control? | AOPC (deletion) |
| RQ4 | What does each explanation cost in wall-clock time, and what fidelity does that buy? | median seconds/explanation |
| RQ5 | Do top-ranked features overlap a pre-registered, literature-derived reference set? | Jaccard@5 |
| RQ6 | Does rank-level fusion of SHAP and LIME beat the better of the two alone? | ΔAOPC vs. max(SHAP, LIME) |

---

## 5. Metric definitions

**Count, for the record.** 8 predictive-performance metrics (Accuracy, Precision, Recall, F1,
ROC-AUC, PR-AUC, MCC, Brier) and 16 interpretability measures in five families: agreement 4
(Spearman ρ, Kendall τ_b, Jaccard@k, RBO); stability 4 (repeat-pair ρ, repeat CV, local Lipschitz
L̂, retraining agreement); fidelity 4 (AOPC, AUC-ins, comprehensiveness, sufficiency); cost 2
(seconds/explanation, fidelity-per-second); plausibility 2 (Jaccard@5, Precision@5).

Let *f* be the fitted classifier, *x* an instance with *d* original features, and φ(x) ∈ ℝ^d the
attribution vector produced by an explainer (after one-hot aggregation). Let `rank(x)` be the
ranking of features by |φ(x)|, descending, ties broken by average rank.

### 5.1 Agreement (RQ1)

- **Spearman ρ** between `rank_shap(x)` and `rank_lime(x)` — full-ranking concordance.
- **Kendall τ_b** — same, robust to ties.
- **Jaccard@k** = |T_k^shap ∩ T_k^lime| / |T_k^shap ∪ T_k^lime| for k ∈ {3, 5}, where T_k is the top-k set.
- **Rank-Biased Overlap (RBO)**, p = 0.9 — top-weighted list similarity; unlike Jaccard it does not
  require choosing a single k, and unlike Spearman it weights the head of the list more heavily.

Computed **locally** (per instance, then summarised as median and IQR over instances) and **globally**
(on rankings derived from mean |φ| over the explained instances).

### 5.2 Stability (RQ2)

- **2a — Re-run stability.** Explain the same instance *R = 10* times with different RNG seeds.
  Report mean pairwise Spearman ρ across the R(R−1)/2 repeat pairs, and the instance-level
  coefficient of variation CV(x) = mean over features j of σ_r(φ_j) / (|μ_r(φ_j)| + ε), ε = 1e-12,
  where σ_r and μ_r are taken across the R repeats. H2a's unit of analysis is CV(x), one value per
  explained instance.
  *Pre-registered expectation:* SHAP with TreeExplainer and LinearExplainer is deterministic, so
  ρ = 1.000 and CV = 0.000 exactly for `dtree`, `rf`, `xgb`, `lgbm`, `logreg`. This is stated a
  priori as a property of the method, **not** presented as an empirical discovery. The informative
  measurements are (i) LIME's variance and how it shrinks with `num_samples`, and (ii) SHAP's
  variance for `stack`, where the backend is sampling-based.
- **2b — Local robustness.** For instance *x*, draw *m = 5* neighbours x′ ~ x + N(0, (0.1·σ_train)²)
  in standardised space, keeping categorical columns fixed. Estimate a local Lipschitz constant
  L̂(x) = max_{x′} ‖φ(x) − φ(x′)‖₂ / ‖x − x′‖₂ (Alvarez-Melis & Jaakkola, 2018). Lower is more robust.
- **2c — Retraining stability.** Refit the model on *B = 10* stratified bootstrap resamples of the
  train partition; recompute global rankings on the fixed test sample; report mean pairwise Spearman ρ
  across resamples. Run for the three ensemble models (`rf`, `xgb`, `lgbm`) only — see §8 R3.
  This isolates **model** variance from **explainer** variance: an explanation can be perfectly
  reproducible (2a) yet worthless if a marginally different training sample reorders its top features.

### 5.3 Fidelity (RQ3)

With features removed in ranked order and replaced by a reference value (primary: train median for
numeric, train mode for categorical):

- **AOPC (deletion)** = (1/K) Σ_{k=1..K} [ f_c(x) − f_c(x_{\setminus top-k}) ], where *c* is the
  model's own predicted class and K = min(d, 10). Higher = more faithful.
- **AUC-ins (insertion)** — area under the curve of f_c as top-k features are restored onto a
  fully-baseline instance. Higher = more faithful.
- **Comprehensiveness** = f_c(x) − f_c(x_{\setminus top-k}) at k = 5 (DeYoung et al., 2020).
- **Sufficiency** = f_c(x) − f_c(x_{top-5 only}) at k = 5. Lower = the top-5 alone suffice.

**Random-ranking control is mandatory.** AOPC values are meaningless in isolation — deleting *any* 5
of 8 features moves a prediction. Every fidelity number is reported alongside `random` (20 random
rankings, averaged), and an explainer that does not significantly exceed the control is reported as
*not demonstrably faithful* on that cell.

**Baseline-imputation sensitivity.** Median replacement creates off-manifold inputs, which can
inflate AOPC. Secondary variant: replace with a random draw from the training marginal of that
feature (10 draws, averaged). Both are reported; if they disagree in direction, the disagreement is
the finding.

### 5.4 Cost (RQ4)

Median and IQR of wall-clock seconds per explanation, measured with `time.perf_counter()` around the
explainer call only (excluding model fitting and background construction, which are amortised and
reported separately). Thread count pinned to 1 for the timed calls via `OMP_NUM_THREADS=1` and
`n_jobs=1` so comparisons are not confounded by parallelism. Hardware and library versions recorded.
Derived: **fidelity per second** = AOPC / median seconds, plotted as an efficiency frontier.

### 5.5 Plausibility (RQ5)

For each dataset a **reference feature set** R_D is compiled from cited prior literature and clinical
guideline sources, frozen in `docs/reference_features.yaml`, and committed **before** any explanation
is computed. Metrics: **Jaccard@5** = |T_5 ∩ R_D| / |T_5 ∪ R_D| and **Precision@5** = |T_5 ∩ R_D| / 5.

**Framing, stated in the paper verbatim.** R_D is a *plausibility reference*, not ground truth. It is
neither complete nor causal, it reflects publication and measurement conventions in the source
literature, and a low overlap score is as consistent with an incomplete reference set as with a poor
explanation. This metric is reported as corroborating evidence only and never as a correctness
criterion. It is the weakest metric in the study and is labelled as such.

### 5.6 Consensus (RQ6)

Given normalised ranks r_shap, r_lime ∈ [0,1]^d (0 = most important):

- **RankAvg**: r_fused = ½ (r_shap + r_lime).
- **RRF**: score_j = 1/(60 + rank_shap,j) + 1/(60 + rank_lime,j), ranked descending.
- **FidelityWeighted**: r_fused = w·r_shap + (1−w)·r_lime, with
  w = AOPC_shap^cal / (AOPC_shap^cal + AOPC_lime^cal).
  **Leakage control:** w is estimated on a *calibration split* — 30 % of the explained instances,
  disjoint from the 70 % evaluation instances on which H6 is tested. The split is fixed by seed and
  emitted to `results/consensus_splits.csv`. Weights are per (dataset, model) cell, never per instance.

The pre-registered bar for H6 is strict: the fused ranking must beat **max(SHAP, LIME)** on the same
cell, not their average. Beating the average is trivial and is not reported as success.

---

## 6. Hypotheses

α = 0.05 throughout. All tests are two-sided unless stated one-sided. All tests are non-parametric:
attribution and fidelity distributions are skewed and bounded, and normality is not assumed anywhere.
Effect sizes accompany every test — with hundreds of instances per cell, statistical significance is
cheap and the effect size is the actual finding.

| ID | Null hypothesis H0 | Alternative H1 | Test | Unit of analysis | Family size |
|----|--------------------|----------------|------|------------------|-------------|
| **H1** | median local Spearman ρ(SHAP, LIME) ≤ 0.50 | > 0.50 | Wilcoxon signed-rank on (ρ_i − 0.50), one-sided | instance, within cell | 18 cells |
| **H1b** | ρ does not differ between linear/single-tree and ensemble models | differs | Mann–Whitney U | cell | 1 |
| **H1c** | ρ is uncorrelated with task difficulty (1 − ROC-AUC) | correlated | Spearman ρ over cells + permutation test | cell | 1 |
| **H2a** | LIME re-run CV = 0 | > 0 | one-sample Wilcoxon, one-sided | instance, within cell | 18 |
| **H2b** | L̂_SHAP = L̂_LIME | ≠ | paired Wilcoxon signed-rank | instance, within cell | 18 |
| **H2c** | retraining rank-agreement equal for SHAP and LIME | ≠ | paired Wilcoxon | resample pair, within cell | 9 |
| **H3a** | AOPC_SHAP = AOPC_LIME | ≠ | paired Wilcoxon | instance, within cell | 18 |
| **H3b** | AOPC_explainer ≤ AOPC_random | > | paired Wilcoxon, one-sided | instance, within cell | 36 |
| **H4** | median runtime SHAP = LIME | SHAP < LIME | paired Wilcoxon, one-sided | instance, within cell | 18 |
| **H5** | Jaccard@5 vs R_D equal for SHAP and LIME | ≠ | paired Wilcoxon | cell | 1 (exploratory) |
| **H6** | max over variants of AOPC_consensus ≤ max(AOPC_SHAP, AOPC_LIME) | > | paired Wilcoxon, one-sided | evaluation instance, within cell | 18 |

**H1's threshold of 0.50 is deliberate.** Testing ρ = 0 would be a straw man — any two competent
explainers of the same model will exceed zero concordance. 0.50 is the conventional boundary between
weak and moderate monotone association, so the test asks something a reader might actually doubt.
The threshold is fixed here, before any ρ is computed.

**H5 is labelled exploratory** because its unit of analysis is the cell (n = 18) and its reference
sets are literature-derived. It is reported with descriptive statistics and effect size; no strong
inference is drawn from it.

---

## 7. Statistical procedures

- **Multiple comparisons.** Holm–Bonferroni correction applied *within each hypothesis family* (the
  family is the set of cells listed in the table above). Both raw *p* and Holm-adjusted *p_adj* are
  reported for every test. Decisions use *p_adj*.
- **Effect sizes.** Rank-biserial correlation *r* for every Wilcoxon test; Cliff's δ for unpaired
  comparisons; point estimates for ρ, τ, Jaccard, RBO, AOPC.
- **Interval estimates.** 95 % BCa bootstrap confidence intervals, B = 2000 resamples, for every
  point estimate reported in a table or figure.
- **Reporting discipline.** Medians and IQRs are reported alongside every *p*-value. No result is
  described as "significant" without its effect size and CI in the same sentence or table row. The
  word "prove" is not used.
- **Zero-difference handling.** Wilcoxon with `zero_method="wilcox"`; cells where all paired
  differences are exactly zero (expected for SHAP determinism) are reported as degenerate rather than
  forced through a test.
- **Hypothesis decision table (T10)** is generated directly from `results/stats_*.csv` and lists, for
  every hypothesis: test, statistic, n, raw p, p_adj, effect size, 95 % CI, and decision.

---

## 8. Pre-registered risks and their documented fallbacks

Recording fallbacks *before* running means a deviation is a disclosed decision rather than an
unexplained inconsistency.

- **R1 — SHAP probability space too slow.** If interventional TreeSHAP in probability space exceeds
  60 s per instance for any cell, that cell falls back to log-odds space. The fallback is recorded
  per cell in `results/explainer_config.csv` and disclosed in the paper. Cross-space comparisons are
  then reported separately, never pooled.
- **R2 — CKD near-separability.** Expected ROC-AUC ≈ 0.99 may yield degenerate explanations (one
  feature dominating, near-zero attributions elsewhere). This is treated as a *finding* about
  explanation behaviour on easy tasks, reported and discussed, not as a dataset defect.
- **R3 — RQ2c compute cost.** Retraining stability with LIME requires B × instances × cells explainer
  calls. Scope is fixed in advance to the three ensemble models (9 cells), B = 10, 25 instances. The
  restriction is stated in the paper's methods and limitations, not buried.
- **R4 — Consensus null result.** If no fusion variant clears the max(SHAP, LIME) bar, H6 is reported
  as not rejected and the paper's contribution 3 becomes a negative result with an analysis of *why*
  (most plausibly: high base agreement leaves no complementary signal to exploit, linking back to
  RQ1). The paper structure does not depend on H6 being rejected.
- **R5 — LIME discretisation.** `discretize_continuous=True` changes attribution semantics to binned
  features. Primary configuration is `True` (LIME's own default and the dominant practice); `False`
  is run as a sensitivity check on one dataset and reported in an appendix.

---

## 9. Reproducibility commitments

1. Every number, table, and figure in the paper, report, and slides is generated by code and written
   to `results/`, `tables/`, or `figures/`. **No value is typed by hand into any document.** LaTeX
   `\input`s generated `.tex` fragments.
2. `results/MANIFEST.json` records the SHA-256 of every produced artefact, the git commit, the seed
   set, the wall-clock duration, and the resolved library versions.
3. Dependencies are pinned to exact versions in `requirements.lock.txt`.
4. `make all` reproduces the entire study from an empty `results/` directory. `make smoke` runs a
   reduced configuration in a few minutes to validate the pipeline end to end.
5. Metric implementations are unit-tested against analytically known cases (identical rankings → ρ=1;
   reversed → ρ=−1; published RBO worked examples; AOPC of a known-linear model).

## 10. Amendment log

| ID | Date | Section | Change | Reason |
|----|------|---------|--------|--------|
| A1 | 2026-08-29 | §5.1 | Added **floor-normalised RBO** as the reported agreement measure alongside raw RBO. | Found during implementation, before any result was computed. Because both compared rankings are permutations of the *same* feature set, RBO's extrapolation term `p^k · A_k` is always exactly `p^k`, which is a hard floor that depends on the number of features: a perfectly reversed ranking still scores 0.589 at d = 8 (PIMA), 0.416 at d = 13 (Heart), and 0.193 at d = 24 (CKD). Raw RBO is therefore not comparable across the three datasets — PIMA cannot score below 0.589 no matter how much the explainers disagree. `rbo_normalised = (RBO − floor) / (1 − floor)` restores a common 0–1 scale. Raw RBO and the per-dataset floor are both still reported so the correction is auditable. |
