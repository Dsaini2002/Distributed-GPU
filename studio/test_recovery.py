import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from . import recovery


class TestRecovery(unittest.TestCase):

    def test_epoch_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                recovery, "BASE", Path(tmp)
            ):
                stage = {
                    "stage_id": "stage_0",
                    "address": "192.168.50.2",
                    "hostname": "Laptop-B",
                    "gpu_index": 0,
                    "gpu": "GPU",
                    "blocks": []
                }

                def fake_fetch(
                    stage, route, payload, binary=False
                ):
                    return b"example checkpoint bytes"

                recovery.save_epoch(
                    "abcdef123456",
                    3,
                    [stage],
                    [],
                    fake_fetch
                )

                saved = recovery.latest_checkpoint(
                    "abcdef123456"
                )

                self.assertIsNotNone(saved)
                self.assertEqual(saved["epoch"], 3)

    def test_invalid_job_id(self):
        with self.assertRaises(ValueError):
            recovery.latest_checkpoint("../other")


if __name__ == "__main__":
    unittest.main()
