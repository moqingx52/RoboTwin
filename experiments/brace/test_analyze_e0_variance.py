import unittest

from experiments.brace.analyze_e0_variance import (
    analyze_point,
    exact_randomization_test,
    seed_cluster_bootstrap_ci,
)


class AnalyzePointTest(unittest.TestCase):
    def test_unclipped_delta_is_primary_and_can_be_negative(self):
        # Two actions, R = 3, one success each in different slots: the ANOVA
        # estimate is negative (within-action noise exceeds between-action
        # spread) and the unclipped delta must preserve the sign while the
        # clipped diagnostic floors at zero.
        point = analyze_point({0: [True, False, False], 1: [True, False, False]}, [3.0])
        self.assertLess(point["sigma2_q_hat"], 0.0)
        self.assertLess(point["delta_c3"], 0.0)
        self.assertEqual(point["sigma2_q_hat_clipped"], 0.0)
        self.assertEqual(point["delta_c3_clipped"], 0.0)

    def test_known_values(self):
        # x = (2, 0), R = 3: v = 1/3, S2_b = 2/9, S2_w = 1/6,
        # sigma2 = 2/9 - 1/18 = 1/6, delta_c3 = 3*(1/6)/(1 + 1) = 0.25.
        point = analyze_point({0: [True, True, False], 1: [False, False, False]}, [3.0])
        self.assertAlmostEqual(point["v_hat"], 1.0 / 3.0)
        self.assertAlmostEqual(point["sigma2_q_hat"], 1.0 / 6.0)
        self.assertAlmostEqual(point["delta_c3"], 0.25)
        self.assertAlmostEqual(point["delta_c3_clipped"], 0.25)


class ExactRandomizationTest(unittest.TestCase):
    def test_strong_action_effect_yields_small_p(self):
        # 10 points, each with one always-succeeding and one always-failing
        # action: under H0 each point has probability 2/C(6,3) = 0.1 of
        # reproducing this split, so the joint p-value must be tiny.
        groups = [[(3, 3), (0, 3)] for _ in range(10)]
        result = exact_randomization_test(groups, iterations=2000, seed=0)
        self.assertEqual(result["informative_points"], 10)
        self.assertLess(result["p_one_sided"], 0.01)

    def test_saturated_points_are_uninformative(self):
        groups = [[(3, 3), (3, 3)], [(0, 3), (0, 3)]]
        result = exact_randomization_test(groups, iterations=100, seed=0)
        self.assertEqual(result["informative_points"], 0)
        self.assertIsNone(result["p_one_sided"])

    def test_null_data_yields_large_p(self):
        # Balanced outcomes spread evenly across actions: no extra-binomial
        # variation, so the test must not reject.
        groups = [[(1, 3), (1, 3)] for _ in range(10)]
        result = exact_randomization_test(groups, iterations=2000, seed=0)
        self.assertGreater(result["p_one_sided"], 0.2)


class SeedClusterBootstrapTest(unittest.TestCase):
    def test_degenerate_clusters_give_point_ci(self):
        points = [{"env_seed": s, "delta": 0.5} for s in range(4)]
        ci = seed_cluster_bootstrap_ci(points, "delta", iterations=200, seed=0)
        self.assertEqual(ci["lower"], 0.5)
        self.assertEqual(ci["upper"], 0.5)
        self.assertEqual(ci["clusters"], 4)

    def test_clusters_stay_together(self):
        # Two seeds with opposite point values: every bootstrap mean is a
        # mixture of whole clusters, so with equal cluster sizes the CI stays
        # within the cluster means and reflects between-seed spread.
        points = [
            {"env_seed": 1, "delta": 0.0},
            {"env_seed": 1, "delta": 0.0},
            {"env_seed": 2, "delta": 1.0},
            {"env_seed": 2, "delta": 1.0},
        ]
        ci = seed_cluster_bootstrap_ci(points, "delta", iterations=500, seed=0)
        self.assertIn(ci["lower"], (0.0, 0.5, 1.0))
        self.assertIn(ci["upper"], (0.0, 0.5, 1.0))
        self.assertLess(ci["lower"], ci["upper"])

    def test_missing_key_returns_none(self):
        self.assertIsNone(seed_cluster_bootstrap_ci([{"env_seed": 1}], "delta", iterations=10, seed=0))


if __name__ == "__main__":
    unittest.main()
