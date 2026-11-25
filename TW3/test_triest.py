"""Check exact small-graph counts, sampling statistics, update invariants, and input errors."""
import itertools
import random
import statistics
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from triest import Reservoir, Triest, exact_triangles, read_graph


class TriestTests(unittest.TestCase):
    """Test the core algorithms with known graphs, independent counts, and fixed seeds."""

    def test_reservoir_uniformity(self):
        """Check reservoir size and per-item inclusion frequency across repeated trials."""
        counts = Counter()
        for seed in range(4000):
            sampler = Reservoir(6, seed)
            for item in range(20):
                sampler.update(item)
            self.assertEqual(len(sampler.items), 6)
            counts.update(sampler.items)

        # Each item is expected in 4000 * 6/20 = 1200 trials. Allow for random variation.
        for item in range(20):
            self.assertAlmostEqual(counts[item], 1200, delta=150)

    def test_exact_clique_and_local(self):
        """The full K6 fits in the sample, with 20 triangles total and 10 per vertex."""
        edges = list(itertools.combinations(range(6), 2))
        self.assertEqual(exact_triangles(edges)[0], 20)
        for variant in ("base", "impr"):
            algorithm = Triest(20, variant=variant, local=True)
            for edge in edges:
                algorithm.update(*edge)
            self.assertEqual(algorithm.estimate(), 20)
            for node in range(6):
                self.assertEqual(algorithm.estimate(node), 10)

    def test_base_sample_counters_after_every_update(self):
        """Check that the adjacency sets and counters describe the same sample after every update."""
        edges = list(itertools.combinations(range(12), 2))
        random.Random(91).shuffle(edges)
        capacity = 10
        algorithm = Triest(capacity, seed=8, local=True)

        for edge in edges:
            algorithm.update(*edge)
            sampled = algorithm.reservoir.items
            total, local = exact_triangles(sampled)
            self.assertEqual(algorithm.counter, total)
            self.assertEqual(algorithm.local, local)
            self.assertLessEqual(len(algorithm.adjacency), 2 * capacity)
            self.assertEqual(sum(map(len, algorithm.adjacency.values())), 2 * len(sampled))

    def test_empirical_unbiasedness(self):
        """Use repeated sampling to check that the mean estimate is close to the 10 triangles in K5."""
        edges = list(itertools.combinations(range(5), 2))
        for variant in ("base", "impr"):
            estimates = []
            for seed in range(3000):
                algorithm = Triest(6, seed, variant)
                for edge in edges:
                    algorithm.update(*edge)
                estimates.append(algorithm.estimate())
            self.assertAlmostEqual(statistics.mean(estimates), 10, delta=0.25)

    def test_empty_triangle_free_and_determinism(self):
        """Check zero estimates for empty streams and paths, and matching samples for equal seeds."""
        for variant in ("base", "impr"):
            first = Triest(6, seed=42, variant=variant)
            second = Triest(6, seed=42, variant=variant)
            self.assertEqual(first.estimate(), 0)
            for node in range(100):
                first.update(node, node + 1)
                second.update(node, node + 1)
            self.assertEqual(first.estimate(), 0)
            self.assertEqual(first.reservoir.items, second.reservoir.items)

    def test_input_cleaning_and_failures(self):
        """Check reversed duplicates, self-loops, and errors for invalid configuration or input."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.txt"
            path.write_text("# test\n1 2\n2 1\n2 2\n2,3\n", encoding="utf-8")
            edges, stats = read_graph(path)
            self.assertEqual(edges, [(1, 2), (2, 3)])
            self.assertEqual((stats["duplicate_edges"], stats["self_loops"]), (1, 1))

            path.write_text("1 bad", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_graph(path)

        with self.assertRaises(ValueError):
            Triest(5)
        with self.assertRaises(ValueError):
            Triest(6, variant="fd")
        with self.assertRaises(ValueError):
            Triest(6).update(1, 1)
        with self.assertRaises(ValueError):
            Triest(6).estimate(1)

        algorithm = Triest(6)
        algorithm.update(1, 2)
        with self.assertRaises(ValueError):
            algorithm.update(2, 1)


if __name__ == "__main__":
    unittest.main()
