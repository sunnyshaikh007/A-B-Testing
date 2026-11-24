"""Regression and edge-case tests for experiment analytics."""

import math
import unittest

import numpy as np
import pandas as pd

from ab_testing.generate import generate_data
from ab_testing.stats import analyze_experiment, required_sample_size, validate_data


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.df = generate_data(n_per_group=400, seed=7)

    def test_generation_is_reproducible_and_unique(self):
        pd.testing.assert_frame_equal(self.df, generate_data(n_per_group=400, seed=7))
        self.assertEqual(self.df.user_id.nunique(), 800)
        self.assertTrue((self.df.revenue >= 0).all())

    def test_analysis_matches_direct_conversion_rates(self):
        r = analyze_experiment(self.df, bootstrap_samples=100, seed=5)
        c = self.df[self.df.group == "control"]
        t = self.df[self.df.group == "treatment"]
        self.assertAlmostEqual(r.conversion_control, c.converted.mean())
        self.assertAlmostEqual(r.absolute_lift, t.converted.mean() - c.converted.mean())
        self.assertLessEqual(r.ci_low, r.ci_high)
        self.assertTrue(0 <= r.p_value <= 1)

    def test_null_identical_rates_p_one(self):
        df = pd.DataFrame({"user_id": range(40), "group": ["control"] * 20 + ["treatment"] * 20,
                           "converted": [0, 1] * 20, "revenue": [0., 10.] * 20})
        r = analyze_experiment(df, bootstrap_samples=100)
        self.assertEqual(r.p_value, 1.0)
        self.assertEqual(r.absolute_lift, 0.0)

    def test_zero_conversion_control_relative_lift_nan(self):
        df = pd.DataFrame({"user_id": range(20), "group": ["control"] * 10 + ["treatment"] * 10,
                           "converted": [0] * 10 + [1] * 10, "revenue": [0.] * 10 + [10.] * 10})
        self.assertTrue(math.isnan(analyze_experiment(df, bootstrap_samples=100).relative_lift))

    def test_rejects_duplicate_user_and_bad_revenue(self):
        df = self.df.copy()
        df.loc[1, "user_id"] = df.loc[0, "user_id"]
        with self.assertRaises(ValueError):
            validate_data(df)
        df = self.df.copy()
        df.loc[0, "revenue"] = np.nan
        with self.assertRaises(ValueError):
            validate_data(df)

    def test_srm_flags_extreme_allocation(self):
        df = generate_data(300, seed=1)
        df = pd.concat([df[df.group == "control"].head(30), df[df.group == "treatment"]])
        self.assertLess(analyze_experiment(df, bootstrap_samples=100).srm_p_value, 0.001)

    def test_planning_validity_and_monotonicity(self):
        small_effect = required_sample_size(0.1, 0.01)
        large_effect = required_sample_size(0.1, 0.03)
        self.assertGreater(small_effect, large_effect)
        self.assertGreater(required_sample_size(0.1, 0.01, power=0.9), small_effect)
        with self.assertRaises(ValueError):
            required_sample_size(0.99, 0.02)


if __name__ == "__main__":
    unittest.main()
