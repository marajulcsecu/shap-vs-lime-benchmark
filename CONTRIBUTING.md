# Contributing

This repository backs a submitted paper, so the constraints below are not style
preferences — breaking one of them makes a published number wrong.

## The two rules that matter

**1. No number is typed by hand into any document.** Tables are `\input` from generated
`.tex` fragments, figures are generated PDFs, and even the headline numbers in the
abstract come from `\newcommand` macros in `tables/findings.tex`. If you need a value in
prose, add it to `findings_macros()` in `src/xaibench/stats/report.py` and reference the
macro. `pytest` enforces this: `test_every_number_enters_through_a_generated_file`.

**2. `docs/PROTOCOL.md` is frozen.** It fixes every research question, hypothesis, metric
formula, statistical test, α, and correction procedure, and it was committed *before* any
commit containing a result. Never edit it silently. If a change is genuinely required,
append a row to its §10 amendment log with the date, what changed, and why — including
whether the change was made before or after seeing results. Seven amendments are recorded;
read them before adding an eighth, because three exist only because a test caught a real
error and that is the standard to hold.

Widening an acceptance threshold because a result fails it is the specific thing
pre-registration exists to prevent. Amendment A6 shows the alternative: disclose the
failure and report a sensitivity analysis.

## Getting set up

```bash
make setup    # virtualenv from requirements.lock.txt (exact pins)
make data     # verify the vendored dataset checksums
make test     # ~300 tests, about two minutes
make smoke    # all five stages at the QUICK profile, about five minutes
```

Never run the pipeline at the `FULL` profile to check a change — `make smoke` exercises
every stage and produces every figure and table at reduced budgets, watermarked
`PROFILE=QUICK — NOT FOR REPORTING`. `FULL` takes about thirty minutes and its output is
what the documents are built from.

## Before you open a pull request

```bash
make test && make docs && ./verify.sh
```

`verify.sh` asserts the things that, if false, would make the documents wrong in a way a
reader could not detect: the run really was `FULL`, the additivity gate passed in every
cell (or the failure is disclosed and sensitivity-checked), no figure carries a draft
watermark, no table contains a leaked missing value, no document body contains a
hand-typed result, all three PDFs build with no undefined references, the paper is within
its 6-page limit, and the protocol freeze is still the first commit in history.

## Tests are the specification

Metric implementations are checked against analytically known answers, not against their
own output — the correct feature ordering must maximise AOPC on a synthetic linear model,
RBO must match a hand-computed three-item case, Holm must match a hand-computed example.
When you add a metric, add that kind of test, not a snapshot.

A test run cannot overwrite reported results: `paths()` honours `XAIB_OUTPUT_ROOT` and a
session-scoped fixture redirects every generated artefact into a temp tree. Do not remove
that — without it `pytest` regenerates `results/` at the QUICK profile and destroys the
`FULL` artefacts the paper is built from.

## Style

Match the surrounding code. Comments explain *why*, especially where a choice looks odd —
several exist because a library behaves in a way the docs do not describe (see the
`xgboost==2.1.4` pin, the real-rows SHAP background, and the `\path`-in-caption note in
`paper/preamble.tex`).

Commit messages state what changed and why it matters. If a commit fixes something that
would have been invisible in the output, say so.
