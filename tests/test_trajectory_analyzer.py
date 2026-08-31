"""
Unit Tests for Trajectory Analyzer Module
"""

import unittest
from solidforge.core.trajectory_analyzer import TrajectoryAnalyzer


class TestTrajectoryAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = TrajectoryAnalyzer(num_sectors=36)

    def test_full_coverage(self):
        self.analyzer.generate_simulated_trajectory(count=36, create_gap=False)
        res = self.analyzer.analyze_coverage()

        self.assertTrue(res.is_coverage_sufficient)
        self.assertGreaterEqual(res.coverage_percentage, 90.0)
        self.assertEqual(len(res.missing_angle_ranges), 0)

    def test_gap_detection(self):
        self.analyzer.generate_simulated_trajectory(count=36, create_gap=True)
        res = self.analyzer.analyze_coverage()

        self.assertFalse(res.is_coverage_sufficient)
        self.assertLess(res.coverage_percentage, 95.0)
        self.assertGreater(len(res.missing_angle_ranges), 0)
        self.assertIn("撮影ギャップ検知", res.guidance_message)


if __name__ == "__main__":
    unittest.main()
