import unittest
from .monitor_api import history_from_logs


class TestMonitor(unittest.TestCase):

    def test_training_history(self):
        logs = [
            "Training 5 Linear layers using 2 GPU stages",
            "Epoch 1/15 | MSE: 0.171604",
            "Epoch 2/15 | MSE: 0.066658"
        ]

        history = history_from_logs(logs)

        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["epoch"], 2)
        self.assertAlmostEqual(
            history[1]["mse"], 0.066658
        )

    def test_no_history(self):
        self.assertEqual(
            history_from_logs(["No logs"]), []
        )


if __name__ == "__main__":
    unittest.main()
