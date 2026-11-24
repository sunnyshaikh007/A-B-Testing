"""Generate a synthetic randomized checkout experiment."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_data(n_per_group: int = 5000, control_rate: float = 0.10,
                  treatment_rate: float = 0.115, seed: int = 42) -> pd.DataFrame:
    if n_per_group < 1 or not all(0 <= p <= 1 for p in (control_rate, treatment_rate)):
        raise ValueError("Need positive group sizes and conversion probabilities in [0,1]")
    rng = np.random.default_rng(seed)
    groups = np.repeat(["control", "treatment"], n_per_group)
    probabilities = np.where(groups == "control", control_rate, treatment_rate)
    converted = rng.binomial(1, probabilities)
    # Lognormal basket values mimic positive, right-skewed order revenue.
    baskets = rng.lognormal(mean=3.6, sigma=0.55, size=len(groups))
    revenue = np.where(converted, np.round(baskets, 2), 0.0)
    df = pd.DataFrame({"user_id": np.arange(1, len(groups) + 1), "group": groups,
                       "converted": converted, "revenue": revenue})
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/experiment.csv")
    parser.add_argument("--n-per-group", type=int, default=5000)
    parser.add_argument("--control-rate", type=float, default=0.10)
    parser.add_argument("--treatment-rate", type=float, default=0.115)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_data(args.n_per_group, args.control_rate, args.treatment_rate, args.seed)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} simulated users to {path}")


if __name__ == "__main__":
    main()
