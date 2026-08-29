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
    alternative: str = "two-sided"
    degenerate: bool = False


def wilcoxon_with_effect(a, b=None, alternative: str = "two-sided",
                         mu: float = 0.0) -> TestResult:
    """Wilcoxon signed-rank with the rank-biserial correlation (Kerby, 2014).

    `mu` shifts the null so a threshold hypothesis can be tested directly: H1 in
    the protocol is rho > 0.50, which is this test on (rho_i - 0.50).

    All-zero differences return degenerate=True rather than raising. That case is
    *expected* for tree SHAP's determinism, and dropping those cells would leave
    holes in the hypothesis decision table.
    """
    a = np.asarray(a, dtype=float).ravel()
    offset = np.asarray(b, dtype=float).ravel() if b is not None else 0.0
    diff = a - offset - mu
    diff = diff[np.isfinite(diff)]
    nonzero = diff[diff != 0.0]

    if nonzero.size == 0:
        return TestResult(n=int(diff.size), statistic=0.0, p_value=1.0, effect_size=0.0,
                          effect_name="rank_biserial", median_diff=0.0,
                          alternative=alternative, degenerate=True)

    result = stats.wilcoxon(nonzero, alternative=alternative, zero_method="wilcox")
    ranks = stats.rankdata(np.abs(nonzero))
    w_pos = float(ranks[nonzero > 0].sum())
    w_neg = float(ranks[nonzero < 0].sum())
    total = w_pos + w_neg
    effect = (w_pos - w_neg) / total if total > 0 else 0.0
    return TestResult(n=int(diff.size), statistic=float(result.statistic),
                      p_value=float(result.pvalue), effect_size=float(effect),
                      effect_name="rank_biserial", median_diff=float(np.median(diff)),
                      alternative=alternative)


def cliffs_delta(a, b) -> float:
    """(#a>b - #a<b) / (n_a * n_b). Nonparametric, in [-1, 1]."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.sign(a[:, None] - b[None, :]).sum() / (a.size * b.size))


def mannwhitney_with_effect(a, b, alternative: str = "two-sided") -> TestResult:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0 or np.ptp(np.concatenate([a, b])) == 0:
        return TestResult(n=int(a.size + b.size), statistic=0.0, p_value=1.0,
                          effect_size=0.0, effect_name="cliffs_delta",
                          median_diff=0.0, alternative=alternative, degenerate=True)
    result = stats.mannwhitneyu(a, b, alternative=alternative)
    return TestResult(n=int(a.size + b.size), statistic=float(result.statistic),
                      p_value=float(result.pvalue), effect_size=cliffs_delta(a, b),
                      effect_name="cliffs_delta", alternative=alternative,
                      median_diff=float(np.median(a) - np.median(b)))


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size <= 1:
        return p.copy()
    return np.asarray(multipletests(p, method="holm")[1], dtype=float)


def percentile_ci(data, statistic, n_resamples: int, seed: int,
                  confidence: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval. Used for medians and other non-smooth statistics.

    BCa estimates its acceleration term by jackknife, which assumes the statistic
    responds smoothly to leaving one observation out. A median does not: it either
    does not move or jumps. On small samples that makes BCa's corrected quantiles
    unreliable, and they can land on the same side of the observed value -- producing
    an interval that excludes the very estimate it describes, which is indefensible
    in a results table. See protocol amendment A5.
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
    boots = np.array([float(statistic(x[rng.integers(0, n, n)]))
                      for _ in range(int(n_resamples))])
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return (theta_hat, theta_hat)
    alpha = 1.0 - confidence
    low = float(np.quantile(boots, alpha / 2.0))
    high = float(np.quantile(boots, 1.0 - alpha / 2.0))
    # A bootstrap distribution can sit entirely to one side of the observed value on
    # a lumpy small sample; widen rather than report an interval that omits its point.
    return (min(low, theta_hat), max(high, theta_hat))


def bca_ci(data, statistic, n_resamples: int, seed: int,
           confidence: float = 0.95) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap interval (Efron, 1987).

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
    boots = np.array([float(statistic(x[rng.integers(0, n, n)]))
                      for _ in range(int(n_resamples))])
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return (theta_hat, theta_hat)

    alpha = 1.0 - confidence
    proportion = float(np.mean(boots < theta_hat))
    proportion = min(max(proportion, 1.0 / (2 * boots.size)), 1.0 - 1.0 / (2 * boots.size))
    z0 = float(stats.norm.ppf(proportion))

    jackknife = np.array([float(statistic(np.delete(x, i))) for i in range(n)])
    deviation = jackknife.mean() - jackknife
    denominator = 6.0 * (float(np.sum(deviation ** 2)) ** 1.5)
    acceleration = float(np.sum(deviation ** 3) / denominator) if denominator > 0 else 0.0

    def adjusted(z_alpha: float) -> float:
        numerator = z0 + z_alpha
        return float(stats.norm.cdf(z0 + numerator / (1.0 - acceleration * numerator)))

    low_q = adjusted(float(stats.norm.ppf(alpha / 2.0)))
    high_q = adjusted(float(stats.norm.ppf(1.0 - alpha / 2.0)))
    if not (np.isfinite(low_q) and np.isfinite(high_q)) or low_q >= high_q:
        low_q, high_q = alpha / 2.0, 1.0 - alpha / 2.0
    return (float(np.quantile(boots, low_q)), float(np.quantile(boots, high_q)))
