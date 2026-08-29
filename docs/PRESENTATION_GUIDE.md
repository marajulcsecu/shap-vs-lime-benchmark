# Presentation and Defence Guide

For a 12–15 minute talk from `slides/main.pdf`, plus 25 anticipated questions with
answers, divided so each author owns the block matching their row in
`docs/CONTRIBUTIONS.md`.

---

## Before you walk in

```bash
make docs                                  # rebuild all three PDFs
grep -l QUICK figures/*.png                # must return nothing
python -c "import json;print(json.load(open('results/MANIFEST.json'))['profile'])"
```

That last command must print `FULL`. If any figure carries the `PROFILE=QUICK` watermark
you are presenting a draft. Also check that the commit hash in the paper's reproducibility
statement matches `git rev-parse HEAD`.

Rehearse **slide 6** (the four traps) out loud. It is the slide that establishes you
understood the method rather than ran a library, and it is where the marks are.

---

## Timing

| Slides | Content | Minutes |
|---|---|---|
| 1–3 | Problem, gap, question, three claims | 3:00 |
| 4–6 | Design, datasets, the four traps | 3:30 |
| 7–14 | Results, one per RQ | 6:00 |
| 15–17 | Limitations, takeaways, close | 2:00 |
| | **Total** | **14:30** |

If you are running long, compress RQ5 (slide 13) to one sentence — "the plausibility check
did not discriminate, and we pre-registered it as the weakest measurement" — and skip the
RQ1-vs-difficulty slide. Never cut the four-traps slide or the limitations slide.

---

## Per-slide talking track

**1 — Title.** Name the course and the supervisor. Ten seconds.

**2 — The problem.** "The models that work best on tabular data are tree ensembles, and
you cannot read their reasoning off the weights. Regulation now requires explanations. Two
methods dominate. They rest on different theory, so they can disagree — and if they do, a
practitioner has no principled way to choose."

**3 — The gap.** Land one sentence: *the gap is not a missing algorithm, it is a missing
controlled comparison*. Then list what prior work leaves out — one axis varied, no tests,
no control, output space never stated.

**4 — Question and claims.** Read the three claims. Then say the sentence that protects
you for the rest of the talk: **"We do not claim to have invented explanation fusion. That
is prior work and we cite it. Our contribution is the evaluation."**

**5 — Design diagram.** Trace the two tracks with a finger. "Track A tunes and scores.
Track B explains a model whose test partition never took part in tuning. Conflating those
is how explanation studies leak."

**6 — Datasets.** Emphasise the difficulty gradient is *chosen*, not accidental, and that
the diseases are benchmarks with no medical claim.

**7 — The four traps.** Slow down. Take each in one sentence, and for trap 2 give the
number: "a k-means background broke SHAP's own additivity by about one per cent on
LightGBM, and every other cell looked fine." Close with: "none of these is visible in the
output when it goes wrong. Three were caught by tests we wrote before running."

**8–14 — Results.** For each: state the finding, then point at the figure, then give the
effect size. Never lead with a *p*-value.

**15 — Limitations.** Read them plainly. Do not hedge and do not apologise — stating them
first is what makes the rest credible.

**16 — Takeaways.** Four sentences. Then the reproducibility line: one command, pinned
dependencies, checksummed data, protocol frozen in git.

---

## Anticipated questions

### Block A — framing and contribution *(Samira Oasela Antika)*

**1. What is actually novel here?**
Three things, each verifiable from a named table: a controlled benchmark across 18 cells
with 11 pre-registered hypotheses; a decomposition of stability into three separable
sources measured on the same models; and a controlled evaluation of three fusion rules
against a strict bar. We deliberately do not claim to have invented fusion — that is prior
work, we cite it, and claiming it would invite the one question we could not answer.

**2. Why not just propose a new explanation method?**
Because a new method we could build in a semester would not be evaluable against the
existing ones, since the evaluation framework itself is not standardised. Fixing the
evaluation is the prerequisite, and it is the part that was actually missing.

**3. Isn't this just running two libraries?**
The libraries produce attributions. Everything that makes those attributions *comparable*
is the work: forcing both into probability space and verifying it, aggregating one-hot
columns back to features, perturbing whole feature groups, and adding a control so the
fidelity numbers have a scale. Slide 6 is that answer.

**4. Why these three datasets?**
Public, de-identified, redistributable, small enough to run the full factorial, and
chosen to span a difficulty gradient so we can ask whether agreement depends on how hard
the task is. A single dataset cannot pose that question.

**5. Why does the paper not target a journal?**
It is prepared in IEEE conference format because the assignment asked for a conference
paper. We deliberately do not claim it is ready for a venue that requires novelty, because
the contribution is a benchmark and an evaluation, and we would rather state that
accurately than overclaim.

**6. How did you choose the reference feature sets?**
From cited clinical guidelines and the datasets' originating studies, frozen in
`docs/reference_features.yaml` and committed before any explanation was computed. They are
explicitly *not* ground truth, and we say so in the paper, on the slide, and in the table
note.

### Block B — data and models *(Yeasmin Akter Shuchona)*

**7. Why recode PIMA's zeros as missing?**
A plasma glucose of zero is not a measurement, it is a missing value recorded as zero.
Leaving it in trains the model on an impossible physiology. Zeros in *pregnancies* are
legitimate and we keep them — the distinction is the point.

**8. Why no SMOTE, given class imbalance?**
Imbalance is mild here, 35 % to 62 % positive. More importantly, synthetic oversampling
changes the data manifold that LIME's neighbourhood sampler and SHAP's background both
draw from — it would confound the very comparison we are making. We use balanced class
weights and report MCC and PR-AUC instead.

**9. Your CKD models get near-perfect accuracy. Isn't that a problem?**
It is a finding. On a near-separable task many single features are individually
sufficient, so explanations have latitude to disagree — which is exactly what we observe.
We registered CKD's near-separability as an expected risk before running.

**10. Why nested cross-validation *and* a separate split?**
Because they answer different questions. Nested CV gives an unbiased performance estimate.
Explanation analysis needs *one* fixed model per cell, not five fold-models, and its test
partition must never have been seen during tuning. Doing both with one split is the
standard leak.

**11. How do you know preprocessing does not leak?**
Every transformer lives inside a `Pipeline` fitted within each fold, and there is a
regression test asserting that the imputer's learned medians differ between two disjoint
halves of the data. If preprocessing were fitted once on everything, they would be equal.

**12. Why only six models?**
They span linear, single-tree, bagged, boosted, and stacked families. Stacking is there
specifically because it admits no tree-specific explainer, which makes it the one family
where SHAP's own re-run variance is a real measurement. In hindsight we would trade model
breadth for dataset breadth — that is in the report's "what we would do differently".

**13. Why pin xgboost to an old version?**
XGBoost 3.x serialises `base_score` as a JSON array string, and `shap`'s tree explainer
parses it with a bare `float()` and raises. The alternative was to use a model-agnostic
explainer for XGBoost, which would have forfeited both the determinism result in RQ2a and
a fair cost comparison in RQ4. Pinning the library was the smaller compromise, and it is
recorded in the appendix.

### Block C — method, metrics, statistics *(MD. Marajul Haque)*

**14. Your SHAP stability result is trivially zero.**
Correct, and we said so in the protocol before running, because `TreeExplainer` is
deterministic given a fixed background. Reporting it as a discovery would be the error.
The informative measurements are the other two: perturbation robustness, where SHAP is
better but neither is insensitive, and retraining stability, which is where a deterministic
explainer's guarantee runs out.

**15. How do you know SHAP and LIME are comparable at all?**
We verify it rather than assume it. If `base_value + Σφ` reproduces `predict_proba`, the
attributions are in probability space by construction. Median additivity error across all
cells is around 1e-8. That check is also what caught the background bug.

**16. Isn't median imputation for the deletion test off-manifold?**
Yes — that is Hooker et al.'s objection and we do not dispute it. Every fidelity number is
therefore repeated with baselines drawn from each feature's observed training marginal, and
the ordering did not reverse. Both variants still assume independence; a fully on-manifold
evaluation needs a generative model of the data, which we name as future work.

**17. Did you correct for multiple comparisons?**
Holm–Bonferroni within each hypothesis family, and both raw and adjusted *p* are reported
for every test. Degenerate cells with no variation to test are kept in the table but
excluded from the divisor, because correcting a test that never ran only discards power.

**18. With hundreds of instances, isn't everything significant?**
Yes, which is why we report an effect size and a bootstrap interval beside every
*p*-value and never describe a result as significant without them. The effect size is the
finding.

**19. Why test ρ > 0.50 rather than ρ > 0?**
Testing against zero would be a straw man: any two competent explainers of the same model
clear it. 0.50 is the conventional boundary between weak and moderate association, and we
fixed it before computing a single ρ.

**20. What is rank-biased overlap doing that Spearman does not?**
It weights the head of the list, which is what a practitioner reading a top-five
explanation actually uses, and it needs no choice of *k*. We also found that its raw form
has a floor that scales with feature count — 0.589 at 8 features but 0.193 at 24 — so we
report a floor-normalised version. That is amendment A1 in the protocol.

**21. Why did fusion fail?**
It follows from RQ1. Fusion pays off when the inputs are individually noisy but their
errors are uncorrelated. Here the two agree moderately and one is consistently better, so
averaging ranks moves the result toward the weaker signal. The fidelity-weighted rule was
designed to counter that and was usually the best of the three, but a weight fitted per
cell on a small calibration sample cannot recover the gap.

**22. Isn't a null result a failure?**
It was pre-registered as a valid outcome, and the paper's structure does not depend on
rejecting H6. A study that could only report a positive result would be a study whose
analysis was contingent on the answer.

### Block D — reproducibility and writing *(Gian Juti Tripura)*

**23. Can I reproduce this?**
`make all`. Dependencies are pinned to exact versions, the datasets are vendored with
SHA-256 digests verified on every load, and `results/MANIFEST.json` records a digest for
every artefact plus the commit, the seed set, the library versions, and the hardware.

**24. How do I know you did not adjust the analysis after seeing the results?**
The protocol was committed in its own commit, before any commit containing a result. The
git history is the evidence. Every later change is in the protocol's amendment log with a
date and a reason, and five are recorded — three of them errors that tests caught.

**25. Are any numbers in the paper typed by hand?**
No. Tables are `\input` from generated fragments, figures are generated PDFs, and the
headline numbers in the abstract come from generated `\newcommand` macros. Rerun an
experiment and the documents change on the next build.

---

## Two things to check before submission

1. **Verify the related-work characterisations against the sources.** The paper describes
   prior comparisons in aggregate rather than tabulating per-paper counts, precisely
   because those counts are not reliably recoverable from abstracts. Read at least the
   Salih, ElShawi, and Amparore papers and confirm the aggregate characterisation is fair.
2. **Adjust `docs/CONTRIBUTIONS.md` to what actually happened.** The split there is a
   proposal. An obviously uniform four-way split on a code-heavy project reads as
   boilerplate, and each author should be able to answer their own block above.
