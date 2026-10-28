import unittest
from .planner import create_plan


class TestPlanner(unittest.TestCase):
    def test_two_gpus(self):
        nodes = []

        for i in range(2):
            nodes.append({
                "hostname": f"Laptop-{i}",
                "gpus": [{
                    "index": 0,
                    "name": f"GPU-{i}",
                    "free_mib": 3300,
                    "total_mib": 4096
                }]
            })

        plan = create_plan(nodes, 16, 256)

        self.assertEqual(plan["stage_count"], 2)
        self.assertEqual(
            [s["blocks"] for s in plan["stages"]],
            [8, 8]
        )

    def test_insufficient_memory(self):
        with self.assertRaises(ValueError):
            create_plan([], 16, 256)


if __name__ == "__main__":
    unittest.main()
