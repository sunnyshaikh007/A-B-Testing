"""Streamlit dashboard for a reproducible A/B checkout experiment."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ab_testing.generate import generate_data
from ab_testing.stats import analyze_experiment, required_sample_size

st.set_page_config(page_title="A/B Experiment Lab", page_icon="🧪", layout="wide")
st.title("🧪 A/B Experiment Lab")
st.caption("A statistically grounded checkout experiment · simulated data by default")

with st.sidebar:
    st.header("Experiment settings")
    source = st.radio("Data source", ["Simulated data", "Upload CSV"])
    alpha = st.select_slider("Significance level (α)", [0.01, 0.05, 0.10], value=0.05)
    baseline = st.slider("Baseline conversion (%)", 1.0, 50.0, 10.0) / 100
    mde = st.slider("Minimum detectable absolute lift (percentage points)", 0.5, 10.0, 1.5) / 100
    power = st.slider("Target power (%)", 70, 95, 80, step=5) / 100
    if source == "Simulated data":
        n = st.slider("Users per variant", 500, 10000, 5000, step=500)
        treatment = st.slider("Treatment conversion (%)", 1.0, 50.0, 11.5) / 100
        seed = st.number_input("Random seed", min_value=0, value=42)
        df = generate_data(n, baseline, treatment, int(seed))
    else:
        upload = st.file_uploader("Upload user-level experiment CSV", type="csv")
        if upload is None:
            st.info("Upload a CSV with user_id, group, converted and revenue columns.")
            st.stop()
        df = pd.read_csv(upload)

try:
    result = analyze_experiment(df, alpha=alpha, bootstrap_samples=500)
    needed = required_sample_size(baseline, mde, alpha, power)
except (ValueError, TypeError) as exc:
    st.error(f"Cannot analyze experiment: {exc}")
    st.stop()

st.subheader("Conversion performance")
a, b, c, d = st.columns(4)
a.metric("Control conversion", f"{result.conversion_control:.2%}", f"{result.n_control:,} users")
b.metric("Treatment conversion", f"{result.conversion_treatment:.2%}", f"{result.n_treatment:,} users")
c.metric("Absolute lift", f"{result.absolute_lift * 100:+.2f} pp")
d.metric("Two-sided p-value", f"{result.p_value:.4f}")

left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["Control", "Treatment"], [result.conversion_control * 100,
                                      result.conversion_treatment * 100], color=["#64748b", "#2563eb"])
    ax.set_ylabel("Conversion rate (%)")
    ax.set_ylim(bottom=0)
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("#### Statistical readout")
    st.write(f"Relative lift: **{result.relative_lift:+.2%}**")
    st.write(f"{(1-alpha):.0%} CI for absolute lift: **{result.ci_low*100:+.2f} to {result.ci_high*100:+.2f} pp**")
    st.write(f"Sample-ratio mismatch p-value: **{result.srm_p_value:.4f}**")
    st.write(f"Required sample size: **{needed:,} users per group** for the planning assumptions")

st.subheader("Revenue guardrail")
e, f, g = st.columns(3)
e.metric("Control revenue / visitor", f"{result.revenue_control:.2f}")
f.metric("Treatment revenue / visitor", f"{result.revenue_treatment:.2f}")
g.metric("Revenue change / visitor", f"{result.revenue_diff:+.2f}")
st.write(f"Bootstrap {(1-alpha):.0%} CI for change: **{result.revenue_ci_low:+.2f} to {result.revenue_ci_high:+.2f}** (same monetary units as input)")

if result.srm_p_value < 0.001:
    st.error("Sample-ratio mismatch detected: investigate randomization and tracking before interpreting effects.")
elif result.p_value < alpha and result.ci_low > 0:
    st.success("Positive statistically significant conversion lift. Check revenue, risks, and practical impact before rollout.")
else:
    st.info("No conclusive positive conversion lift at this significance threshold. Consider power and the full interval.")

with st.expander("Inspect data and assumptions"):
    st.dataframe(df.head(100), use_container_width=True)
    st.download_button("Download analyzed data", df.to_csv(index=False), file_name="experiment.csv")
    st.caption("Independent user-level randomization assumed. Normal-approximation interval; bootstrap revenue CI. Avoid repeated unplanned peeking.")
