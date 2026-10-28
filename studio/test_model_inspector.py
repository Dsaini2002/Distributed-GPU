import tempfile
import unittest
from pathlib import Path

from .model_inspector import inspect_model, inspect_dataset


class TestModelInspection(unittest.TestCase):

    def test_supported_model_candidate(self):
        code = """
import torch.nn as nn

def get_model():
    return nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "example.py"
            path.write_text(code)

            info = inspect_model(path)

            self.assertTrue(
                info["current_engine_candidate"]
            )

    def test_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "example.csv"
            path.write_text(
                "f1,f2,target\n1,2,3\n4,5,6\n"
            )

            info = inspect_dataset(path)

            self.assertTrue(
                info["current_engine_candidate"]
            )

    def test_unsupported_model(self):
        code = """
import torch.nn as nn

def get_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU()
    )
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cnn.py"
            path.write_text(code)

            info = inspect_model(path)

            self.assertFalse(
                info["current_engine_candidate"]
            )


if __name__ == "__main__":
    unittest.main()
