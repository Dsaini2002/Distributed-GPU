import unittest

from .model_partition import plan_model


class TestNComputerPartition(unittest.TestCase):
    def test_three_computers(self):
        blocks = [
            {"in": 8, "out": 64, "relu": True},
            {"in": 64, "out": 64, "relu": True},
            {"in": 64, "out": 64, "relu": True},
            {"in": 64, "out": 64, "relu": True},
            {"in": 64, "out": 64, "relu": True},
            {"in": 64, "out": 1, "relu": False}
        ]

        nodes = []
        for i in range(3):
            nodes.append({
                "hostname": f"Laptop-{i}",
                "address": f"192.168.50.{i+1}",
                "online": True,
                "gpus": [{
                    "name": f"GPU-{i}",
                    "index": 0,
                    "free_mib": 3300
                }]
            })

        stages = plan_model(blocks, nodes, 32)

        self.assertEqual(len(stages), 3)
        self.assertEqual(
            sum(len(s["blocks"]) for s in stages), 6
        )


if __name__ == "__main__":
    unittest.main()
