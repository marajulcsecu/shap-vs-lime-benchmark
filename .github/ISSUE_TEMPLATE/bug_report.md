---
name: Bug report
about: Something in the pipeline, metrics, or documents is wrong
labels: bug
---

**What is wrong**

<!-- If a reported number looks wrong, say which table or figure and what you expected. -->

**How to reproduce**

```bash
# the exact command, including the profile
XAIB_PROFILE=QUICK python -m xaibench.pipeline.stage3_metrics --force
```

**Environment**

Paste the relevant lines from `results/MANIFEST.json` (`profile`, `git_commit`,
`library_versions`) so the run can be identified.

**Does it affect a published number?**

If yes, note which hypothesis or figure, because that determines whether the documents
need rebuilding.
