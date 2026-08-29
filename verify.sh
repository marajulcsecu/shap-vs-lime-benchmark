#!/usr/bin/env bash
# Final integrity check for a reported run. Every assertion here is something that,
# if false, would make the documents wrong in a way a reader could not detect.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
fail=0
check() { if [ "$1" = "0" ]; then echo "  PASS  $2"; else echo "  FAIL  $2"; fail=1; fi; }

echo "=== 1. run integrity ==="
PYTHONPATH=src $PY - <<'PYEOF'
import json, sys
import pandas as pd
from xaibench.config import CELLS, paths
p = paths()
cfg = pd.read_csv(p.results / "explainer_config.csv")
manifest = json.loads((p.results / "MANIFEST.json").read_text())
budget = pd.read_csv(p.results / "instance_budget.csv")
stats = pd.read_csv(p.results / "stats_hypotheses.csv")
problems = []
if len(cfg) != len(CELLS): problems.append(f"{len(cfg)} cells, expected {len(CELLS)}")
failed = cfg[~cfg.additivity_gate_ok.astype(bool)]
if len(failed):
    # R1 firing is allowed; silently ignoring it is not. Require the disclosure.
    disclosed = (p.results / "stats_sensitivity_r1.csv").exists()
    # Documents use human labels ("LightGBM"), results use ids ("lgbm").
    from xaibench.models.zoo import MODEL_LABELS
    text = open("paper/sections/06_limitations.tex").read().lower()
    named = all(r.dataset.lower() in text
                and MODEL_LABELS[r.model].lower() in text
                for r in failed.itertuples())
    if not disclosed: problems.append("R1 fired but no sensitivity analysis was written")
    if not named: problems.append("R1 fired but the affected cell is not named in the limitations")
    print(f"  NOTE: protocol risk R1 fired for {len(failed)} cell(s) -- "
          f"{', '.join(r.dataset + '/' + r.model for r in failed.itertuples())}; "
          "disclosed and sensitivity-checked")
if set(cfg.space) != {"probability"}: problems.append(f"mixed output spaces: {set(cfg.space)}")
if manifest["profile"] != "FULL": problems.append(f"profile is {manifest['profile']}, not FULL")
if not manifest["reportable"]: problems.append("manifest marks the run non-reportable")
if stats.hypothesis.nunique() != 11: problems.append(f"{stats.hypothesis.nunique()} hypotheses, expected 11")
if (stats.p_adj < stats.p_value - 1e-12).any(): problems.append("an adjusted p is below its raw p")
bad = [v for v in manifest["artifacts"].values() if len(v["sha256"]) != 64]
if bad: problems.append(f"{len(bad)} artefacts without a valid digest")
print(f"  cells={len(cfg)}  explained={int(budget.n_explained.sum())}  "
      f"hypotheses={stats.hypothesis.nunique()}  artefacts={len(manifest['artifacts'])}")
print(f"  worst median additivity error={cfg.additivity_median.max():.2e}  "
      f"profile={manifest['profile']}  commit={manifest['git_commit'][:8]}")
for problem in problems: print("  ISSUE:", problem)
sys.exit(1 if problems else 0)
PYEOF
check $? "results are internally consistent and from a FULL run"

echo "=== 2. no draft artefacts reached the documents ==="
n=$(grep -lc QUICK figures/*.png 2>/dev/null | wc -l)
[ "$n" = "0" ]; check $? "no figure carries a QUICK watermark"
! grep -qri "nan\|PROFILE=QUICK" tables/*.tex; check $? "no table contains nan or a draft banner"

echo "=== 3. every number is generated, none typed by hand ==="
PYTHONPATH=src $PY - <<'PYEOF'
import re, sys
from pathlib import Path
allowed = re.compile(r"(\\cite|\\ref|\\label|\\input|\\includegraphics|^%|0\.50|0\.05|"
                     r"1e-|e-0|5\\times10|9\.9\\times|7\\times|1\.5\\times|2024|2019|"
                     r"0\.1\\sigma|\\times10)")
offenders = []
for folder in ("paper/sections", "report/sections", "report/appendices"):
    for path in sorted(Path(folder).glob("*.tex")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            s = line.strip()
            if not s or s.startswith("%") or allowed.search(s):
                continue
            if re.search(r"(?<![\\{A-Za-z0-9.])\d+\.\d{3,}", s):
                offenders.append(f"{path}:{i}: {s[:88]}")
for o in offenders: print("  HAND-TYPED:", o)
sys.exit(1 if offenders else 0)
PYEOF
check $? "no hand-typed multi-decimal result in any document body"

echo "=== 4. documents build and are the right length ==="
for d in paper report slides; do
  (cd $d && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1)
  check $? "$d compiles"
done
PYTHONPATH=src $PY - <<'PYEOF'
import sys
from pypdf import PdfReader
# 6 pages is the supervisor's stated IEEE conference limit, including references.
want = {"paper": (5, 6), "report": (28, 60), "slides": (18, 40)}
bad = []
for name, (low, high) in want.items():
    n = len(PdfReader(f"{name}/main.pdf").pages)
    print(f"  {name}: {n} pages (target {low}-{high})")
    if not low <= n <= high:
        bad.append(f"{name} ({n} pages, allowed {low}-{high})")
if bad: print("  OVER/UNDER LIMIT:", "; ".join(bad))
sys.exit(1 if bad else 0)
PYEOF
check $? "page counts are within target"
! grep -qE "Reference .* undefined|Citation .* undefined|There were undefined (references|citations)" \
    paper/main.log report/main.log
check $? "no undefined references or citations"
# The rendered artefact is what a reader sees, so check for "??" there too. The log
# grep alone was matching harmless "Font shape ... undefined" warnings and missing
# nothing; this catches a broken cross-reference even if the log wording changes.
broken=$(for d in paper report; do pdftotext $d/main.pdf - 2>/dev/null; done \
         | grep -cE "Table \?\?|Fig\. \?\?|Section \?\?|Chapter \?\?|\[\?\]")
[ "$broken" = "0" ]; check $? "no ?? or [?] placeholders in the rendered PDFs"

echo "=== 5. reproducibility surface ==="
[ -s requirements.lock.txt ]; check $? "dependency lockfile present"
PYTHONPATH=src $PY -m xaibench.data.fetch >/dev/null 2>&1; check $? "source data checksums verify"
git log --reverse --format=%s | grep -m1 . | grep -q "freeze pre-registered protocol"
check $? "protocol freeze is the first commit in history"

echo
if [ "$fail" = "0" ]; then echo "ALL CHECKS PASSED"; else echo "SOME CHECKS FAILED"; fi
exit $fail
