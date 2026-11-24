"""Export reproducible, GitHub-renderable SVG experiment charts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ab_testing.generate import generate_data
from ab_testing.stats import analyze_experiment


def main() -> None:
    output = Path("visualizations")
    output.mkdir(exist_ok=True)
    data = generate_data(n_per_group=5000, seed=42)
    result = analyze_experiment(data, bootstrap_samples=200, seed=42)
    colors = ["#64748b", "#2563eb"]
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "figure.facecolor": "white"})

    fig, ax = plt.subplots(figsize=(8, 4))
    values = [result.conversion_control * 100, result.conversion_treatment * 100]
    bars = ax.bar(["Control", "Treatment"], values, color=colors, width=0.52)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=12)
    ax.set(ylabel="Conversion rate (%)", title="Checkout A/B test: conversion rates", ylim=(0, max(values) * 1.25))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output / "conversion_rates.svg", metadata={"Date": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for group, color in zip(["control", "treatment"], colors):
        subset = data[data.group == group].sort_values("user_id")
        cum = subset.converted.to_numpy().cumsum() / np.arange(1, len(subset) + 1)
        # Plot a stable, interpretable region after the first 100 participants.
        steps = np.arange(99, len(subset), 20)
        ax.plot(steps + 1, cum[steps] * 100, label=group.title(), color=color, linewidth=2)
    ax.set(xlabel="Participants per group", ylabel="Cumulative conversion (%)",
           title="How estimated conversion evolves with sample size")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output / "cumulative_conversion.svg", metadata={"Date": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3))
    effect = result.absolute_lift * 100
    ax.errorbar([effect], [0], xerr=[[max(0, (result.absolute_lift - result.ci_low)*100)],
                                     [max(0, (result.ci_high - result.absolute_lift)*100)]],
                fmt="o", color=colors[1], markersize=9, capsize=8, linewidth=2)
    ax.axvline(0, linestyle="--", color="#dc2626", linewidth=1.5)
    ax.set(xlabel="Treatment − control conversion (percentage points)", yticks=[], ylim=(-0.4, 0.4),
           title="Absolute conversion lift with 95% confidence interval")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(output / "lift_confidence_interval.svg", metadata={"Date": None})
    plt.close(fig)
    print(f"Generated 3 SVG charts using {len(data):,} simulated users")


if __name__ == "__main__":
    main()
