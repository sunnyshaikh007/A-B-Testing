"""Analyze a user-level A/B experiment CSV from the command line."""

import argparse

import pandas as pd

from .stats import analyze_experiment, required_sample_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/experiment.csv")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--baseline", type=float, default=0.10)
    parser.add_argument("--mde", type=float, default=0.015, help="Absolute lift (e.g. 0.015)")
    parser.add_argument("--power", type=float, default=0.80)
    args = parser.parse_args()
    result = analyze_experiment(pd.read_csv(args.input), alpha=args.alpha)
    print("A/B TEST RESULTS (synthetic data unless your CSV is real)")
    print(f"Sample sizes: control={result.n_control:,}, treatment={result.n_treatment:,}")
    print(f"Conversion: control={result.conversion_control:.2%}, treatment={result.conversion_treatment:.2%}")
    print(f"Absolute lift: {result.absolute_lift:+.2%} (CI: {result.ci_low:+.2%} to {result.ci_high:+.2%})")
    print(f"Relative lift: {result.relative_lift:+.2%}")
    print(f"Two-sided p-value: {result.p_value:.4g}; SRM p-value: {result.srm_p_value:.4g}")
    print(f"Revenue / visitor: control={result.revenue_control:.2f}, treatment={result.revenue_treatment:.2f}")
    print(f"Revenue difference: {result.revenue_diff:+.2f} (bootstrap CI: {result.revenue_ci_low:+.2f} to {result.revenue_ci_high:+.2f})")
    print(f"Planned sample size / group: {required_sample_size(args.baseline, args.mde, args.alpha, args.power):,}")
    if result.srm_p_value < 0.001:
        print("CAUTION: SRM detected; investigate assignment and instrumentation first.")
    elif result.p_value < args.alpha and result.ci_low > 0:
        print("Conversion lift is statistically significant; consider revenue and practical impact before rollout.")
    else:
        print("Insufficient evidence of positive conversion lift at this threshold.")


if __name__ == "__main__":
    main()
