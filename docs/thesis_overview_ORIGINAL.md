# Thesis Overview — XAI Ensemble Comparison Study

---

## 1. TL;DR (30-second read)

We systematically compare two popular explainable-AI (XAI) methods — **SHAP** and **LIME** — on three modern ensemble machine-learning models (Random Forest, XGBoost, LightGBM) across three public medical tabular datasets. We measure how much the two explanation methods agree, how stable they are, how model-faithful they are, how fast they run, and how well their explanations match established domain knowledge. **Building on that comparison, we also propose a novel hybrid consensus explanation method** that combines SHAP and LIME outputs — this is our methodological contribution. All experiments, code, datasets, and results will be open-source and reproducible.

---

## 2. Title & Aim

**Title**:
> *Comparative Interpretability Analysis of Ensemble Machine Learning Models Using SHAP and LIME: A Multi-Disease Tabular Data Study (with Hybrid Consensus Method)*

**One-paragraph aim**:
Modern machine-learning models (especially ensembles like XGBoost) achieve high predictive accuracy but are often criticised as "black boxes" — their decisions cannot easily be explained to humans. Two post-hoc explanation methods — **SHAP** (game-theoretic attributions) and **LIME** (local surrogate models) — are widely used to "open" these black boxes, but they don't always agree with each other, and we don't fully know when each one is trustworthy. This thesis **empirically compares** SHAP and LIME on three tree-ensemble architectures over three public datasets, measuring agreement, stability, fidelity, runtime, and domain-overlap, and then **proposes a novel hybrid consensus method** that combines the strengths of both. The intended deliverable is a publishable IEEE Access paper plus a fully reproducible open-source pipeline.

---

## 3. The Problem (Motivation)

### Why does this matter?

1. **Black-box models are everywhere in high-stakes domains** (healthcare, finance, criminal justice). Regulatory bodies (EU AI Act, US Algorithmic Accountability Act) increasingly require *explainable* decisions — not just accurate ones.
2. **SHAP and LIME are the two most-cited post-hoc explanation methods** for tabular data. Yet they were developed with different theory (Shapley values vs local linear surrogates) and have different failure modes.
3. **Most existing XAI comparison studies** either focus on a single model, single dataset, or use metrics that are not standardised — making their results hard to generalise.
4. **Almost no studies propose new XAI methods** — most just compare what exists. Our hybrid consensus method is a step toward bridging that gap.

### What gap does our thesis fill?

A **multi-model × multi-dataset × multi-metric × multi-explainer** systematic comparison with named statistical tests, plus a **novel hybrid explanation method** evaluated against the same framework.

---

## 4. What We Propose

### Scope at a glance

| Dimension | Locked Decision |
|---|---|
| **Research questions** | 6 RQs (Agreement, Stability, Fidelity, Runtime, Domain-overlap, **Hybrid consensus**) |
| **Hypotheses** | 6 (each with H0, H1, named statistical test, α = 0.05) |
| **Datasets** | 3 (PIMA Diabetes, Heart Disease/Cleveland, Chronic Kidney Disease) — all CC0/public domain |
| **Models** | 6 (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, Stacking) |
| **Metrics** | 10 (4 performance + 6 interpretability) |
| **XAI methods** | SHAP (TreeExplainer / LinearExplainer / KernelExplainer) + LIME (LimeTabularExplainer) + **Novel: Hybrid Consensus Method** |
| **Total dataset size** | ≈ 1,471 rows across the three tabular datasets |
| **Framing** | Computer Science methodology study — diseases are *benchmarks*, no medical claims |

### The 6 research questions, in plain English

| # | Question | Simple meaning |
|---|---|---|
| RQ1 | **Agreement** | Do SHAP and LIME produce the same global feature rankings? (Spearman ρ) |
| RQ2 | **Stability** | If we re-run an explanation, does the answer stay the same? (CV across re-runs) |
| RQ3 | **Fidelity** | Does the explanation actually explain the model's real behaviour? (delete-and-predict) |
| RQ4 | **Runtime** | How long does each explainer take? (seconds per explanation) |
| RQ5 | **Domain overlap** | Do top features match what prior papers said matters for each disease? (Jaccard) |
| **RQ6** | **Hybrid consensus** | Does a hybrid method (SHAP + LIME averaged) outperform either alone? (Jaccard vs best single) |

### Our novel contribution — RQ6 / Hybrid Consensus Method

We propose a **hybrid consensus explanation** that combines SHAP and LIME outputs (e.g., by rank-averaging their top-features). Hypothesis: combining two explainers reduces single-method bias, similar to how ensemble ML models outperform single learners. This is tested against the same metrics as RQ5, with the bar being **does the hybrid beat the better of SHAP-only or LIME-only overlap**.

If the hybrid performs better, it's a publishable novel method. If not, we have a publishable finding about the limits of consensus.

---

## 5. The XAI Methods in Scope

### SHAP (SHapley Additive exPlanations)
- **Theory**: Game-theoretic — assigns each feature a Shapley value indicating its marginal contribution to the prediction.
- **Variants used**: TreeExplainer (for RF/XGB/LGBM), LinearExplainer (for LR), KernelExplainer (for Stacking if needed).
- **Property**: Theoretically grounded; deterministic for trees.

### LIME (Local Interpretable Model-agnostic Explanations)
- **Theory**: Fits a simple linear surrogate model locally around each prediction.
- **Variant used**: LimeTabularExplainer (for tabular data).
- **Property**: Model-agnostic; inherently stochastic (sampling-based) → explains run-to-run variance.

### Hybrid Consensus Method (our proposal)
- **Idea**: Take SHAP and LIME attributions on the same instance → rank-normalise → average ranks → extract top-k.
- **Rationale**: Inherits SHAP's theoretical grounding + LIME's model-agnostic coverage; combines two views of "what matters".
- **Risk**: Could dilute signal if the two methods fundamentally disagree. That's why RQ1 (agreement) is upstream of RQ6.


## 6. Team

| Person | ID | Email |
|---|---|---|
| Samira Oasela Antika | 22701043 | 22701043@std.cu.ac.bd |
| Yeasmin Akter Shuchona | 22701052 | 22701052@std.cu.ac.bd |
| MD. Marajul Haque | 23701070 | 23701070@std.cu.ac.bd |
| Gian Juti Tripura | 22701073 | 22701073@std.cu.ac.bd |


Advisor: Dr. M. Jamshed Alam Patwary
Lecturer, Computer Science & Engineering, University of Chittagong, Chittagong-4331