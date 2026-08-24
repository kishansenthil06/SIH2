"""Unit tests verifying backend integration with `src/` and `data/` prototype dataset."""
import unittest
from pathlib import Path

from src.detector import PowerThresholdDetector
from src.environment import RFEnvironment
from src.evaluator import Evaluator
from src.receiver import Receiver
from src.scanner import AdaptiveScanner, BaselineScanner, RandomScanner, make_scanner

DATA_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "prototype" / "temporary_rf_dataset.csv"


class TestPrototypeBackendIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = RFEnvironment(DATA_FILE)

    def test_environment_loads_data(self):
        self.assertGreater(len(self.env.data), 0)
        self.assertEqual(len(self.env.frequency_bands), 20)
        self.assertIn(0, self.env.time_slots)

    def test_environment_stats(self):
        stats = self.env.get_stats()
        self.assertIn("total_observations", stats)
        self.assertEqual(stats["num_frequency_bands"], 20)
        self.assertIn("active_ratio", stats)

    def test_time_slot_observations(self):
        obs = self.env.get_time_slot_observations(0)
        self.assertEqual(len(obs), 20)
        for o in obs:
            self.assertEqual(o["time_slot"], 0)
            self.assertIn("signal_power", o)
            self.assertIn("hit", o)

    def test_receiver_and_scanners(self):
        rx = Receiver(self.env)
        
        # Test sequential scanner
        seq_scanner = BaselineScanner(rx, num_bands=20)
        obs0 = seq_scanner.scan(0)
        obs1 = seq_scanner.scan(1)
        self.assertEqual(obs0["frequency_band"], 1)
        self.assertEqual(obs1["frequency_band"], 2)

        # Test random scanner
        rand_scanner = RandomScanner(rx, num_bands=20, seed=123)
        r_obs = rand_scanner.scan(0)
        self.assertTrue(1 <= r_obs["frequency_band"] <= 20)

        # Test adaptive scanner
        adapt_scanner = AdaptiveScanner(rx, num_bands=20)
        a_obs = adapt_scanner.scan(0)
        self.assertTrue(1 <= a_obs["frequency_band"] <= 20)

        # Test factory
        s = make_scanner("adaptive", rx, num_bands=20)
        self.assertIsInstance(s, AdaptiveScanner)

    def test_detector_and_evaluator(self):
        rx = Receiver(self.env)
        scanner = BaselineScanner(rx, num_bands=20)
        detector = PowerThresholdDetector(threshold=5.0)
        evaluator = Evaluator(self.env)

        for i in range(50):
            obs = scanner.scan(i)
            pred = detector.predict(obs)
            res = evaluator.evaluate(obs["time_slot"], obs["frequency_band"], pred)
            self.assertIn(res["result"], ["true_positive", "false_positive", "true_negative", "false_negative"])

        metrics = evaluator.metrics()
        self.assertEqual(metrics["total_scans"], 50)
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= metrics["detection_rate"] <= 1.0)
        self.assertTrue(0.0 <= metrics["false_alarm_rate"] <= 1.0)
        self.assertTrue(0.0 <= metrics["precision"] <= 1.0)
        self.assertTrue(0.0 <= metrics["f1_score"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
