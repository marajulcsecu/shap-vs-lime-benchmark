# Dataset licences

The MIT licence in [`LICENSE`](LICENSE) covers **the code in this repository only**. The
three benchmark datasets vendored under `data/raw/` retain their own terms and are
redistributed under them.

| Dataset | File | Licence | Source |
|---|---|---|---|
| PIMA Indian Diabetes | `data/raw/pima.csv` | CC0 1.0 (public domain) | Originally collected by the US National Institute of Diabetes and Digestive and Kidney Diseases; mirrored via `plotly/datasets`. Cited as Smith et al., *Proc. Annu. Symp. Comput. Appl. Med. Care*, 1988. |
| Heart Disease (Cleveland) | `data/raw/heart.csv` | CC BY 4.0 | UCI Machine Learning Repository, dataset 45, [doi:10.24432/C52P4X](https://doi.org/10.24432/C52P4X). Cited as Detrano et al., *Am. J. Cardiol.*, 1989. |
| Chronic Kidney Disease | `data/raw/ckd.csv` | CC BY 4.0 | UCI Machine Learning Repository, dataset 336, [doi:10.24432/C5G020](https://doi.org/10.24432/C5G020). Cited as Rubini, Soundarapandian & Eswaran, 2015. |

All three are public and de-identified. SHA-256 digests are recorded in
`data/raw/checksums.json` and verified on every run — a mismatch is a hard failure, because
a silently changed source would invalidate every number in the paper.

The diseases are used as **benchmarks for a methodological question in computer science**.
No model trained in this repository is fit for diagnostic use, and no feature ranking
reported here should be read as a clinical or causal statement.
