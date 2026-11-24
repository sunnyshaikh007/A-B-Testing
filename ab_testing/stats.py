"""User-level A/B testing statistics; all rates are on a 0–1 scale."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm


@dataclass(frozen=True)
class ExperimentResult:
    n_control: int
    n_treatment: int
    conversion_control: float
    conversion_treatment: float
    absolute_lift: float
    relative_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    srm_p_value: float
    revenue_control: float
    revenue_treatment: float
    revenue_diff: float
    revenue_ci_low: float
    revenue_ci_high: float


def validate_data(df: pd.DataFrame) -> None:
    required = {"user_id", "group", "converted", "revenue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty or df.user_id.isna().any() or df.user_id.duplicated().any():
        raise ValueError("Input must contain distinct, non-null user IDs")
    if df.group.isna().any() or set(df.group.unique()) != {"control", "treatment"}:
        raise ValueError("Both control and treatment groups are required")
    if not df.converted.isin([0, 1, False, True]).all():
        raise ValueError("converted must be binary (0 or 1)")
    if (not pd.api.types.is_numeric_dtype(df.revenue)
            or not np.isfinite(df.revenue.to_numpy(dtype=float)).all()
            or (df.revenue < 0).any()):
        raise ValueError("revenue must be finite and nonnegative")
    if ((df.converted == 0) & (df.revenue != 0)).any():
        raise ValueError("Non-converters must have zero revenue")


def required_sample_size(baseline: float, mde: float, alpha: float = 0.05,
                         power: float = 0.80) -> int:
    """Approximate equal-sized groups for a two-sided z-test; MDE is absolute."""
    if not (0 < baseline < 1 and 0 < baseline + mde < 1 and mde != 0):
        raise ValueError("Baseline and baseline + MDE must be in (0,1); MDE nonzero")
    if not (0 < alpha < 1 and 0 < power < 1):
        raise ValueError("alpha and power must be in (0,1)")
    p2 = baseline + mde
    pooled = (baseline + p2) / 2
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    numerator = (z_alpha * math.sqrt(2 * pooled * (1 - pooled))
                 + z_beta * math.sqrt(baseline * (1 - baseline) + p2 * (1 - p2))) ** 2
    return math.ceil(numerator / mde ** 2)


def analyze_experiment(df: pd.DataFrame, alpha: float = 0.05,
                       expected_control_share: float = 0.5,
                       bootstrap_samples: int = 2000, seed: int = 42) -> ExperimentResult:
    if not (0 < alpha < 1 and 0 < expected_control_share < 1):
        raise ValueError("alpha and expected_control_share must be in (0,1)")
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    validate_data(df)
    c = df.loc[df.group == "control"]
    t = df.loc[df.group == "treatment"]
    nc, nt = len(c), len(t)
    xc, xt = int(c.converted.sum()), int(t.converted.sum())
    pc, pt = xc / nc, xt / nt
    diff = pt - pc
    pooled = (xc + xt) / (nc + nt)
    pooled_se = math.sqrt(pooled * (1 - pooled) * (1 / nc + 1 / nt))
    p_value = 2 * norm.sf(abs(diff / pooled_se)) if pooled_se else 1.0
    # Unpooled Wald interval is interpretable for sufficiently large samples.
    se = math.sqrt(pc * (1 - pc) / nc + pt * (1 - pt) / nt)
    margin = norm.ppf(1 - alpha / 2) * se
    srm = binomtest(nc, nc + nt, expected_control_share).pvalue

    cr = c.revenue.to_numpy(dtype=float)
    tr = t.revenue.to_numpy(dtype=float)
    revenue_diff = float(tr.mean() - cr.mean())
    rng = np.random.default_rng(seed)
    # Bounded batches keep memory stable for larger experiments.
    deltas = np.empty(bootstrap_samples)
    for start in range(0, bootstrap_samples, 100):
        count = min(100, bootstrap_samples - start)
        cs = cr[rng.integers(nc, size=(count, nc))].mean(axis=1)
        ts = tr[rng.integers(nt, size=(count, nt))].mean(axis=1)
        deltas[start:start + count] = ts - cs
    rev_low, rev_high = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return ExperimentResult(
        nc, nt, pc, pt, diff, diff / pc if pc else math.nan,
        diff - margin, diff + margin, float(p_value), float(srm),
        float(cr.mean()), float(tr.mean()), revenue_diff,
        float(rev_low), float(rev_high),
    )
