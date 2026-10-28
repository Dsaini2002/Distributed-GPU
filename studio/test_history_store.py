import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from . import history_store


class TestHistoryStore(unittest.TestCase):

    def test_save_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                history_store, "JOBS_DIR", Path(directory)
            ):
                job = {
                    "job_id": "abcdef123456",
                    "status": "completed",
                    "epoch": 15,
                    "history": [
                        {"epoch": 1, "mse": 0.17},
                        {"epoch": 15, "mse": 0.0002}
                    ]
                }

                history_store.save_job(job)
                restored = history_store.load_job(
                    "abcdef123456"
                )

                self.assertEqual(
                    restored["status"], "completed"
                )
                self.assertEqual(
                    len(restored["history"]), 2
                )

    def test_reject_invalid_id(self):
        with self.assertRaises(ValueError):
            history_store.load_job("../private")

    def test_list_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                history_store, "JOBS_DIR", Path(directory)
            ):
                history_store.save_job({
                    "job_id": "abcdef123456",
                    "status": "completed"
                })

                self.assertEqual(
                    len(history_store.list_jobs()), 1
                )


if __name__ == "__main__":
    unittest.main()
