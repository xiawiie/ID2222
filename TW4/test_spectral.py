"""Check spectral clustering and input validation on graphs with known answers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from spectral import (
    choose_k,
    load_graph,
    normalized_spectrum,
    partition_metrics,
    spectral_clustering,
)


class SpectralTests(unittest.TestCase):
    """Six checks that run without the assignment datasets."""

    def test_disconnected_triangles(self):
        """Two zero eigenvalues match two components, with no edges crossing the expected clusters."""
        adjacency = np.zeros((6, 6))
        triangle = np.ones((3, 3)) - np.eye(3)
        adjacency[:3, :3] = triangle
        adjacency[3:, 3:] = triangle

        labels, embedding, values = spectral_clustering(adjacency, 2)
        self.assertAlmostEqual(adjusted_rand_score([0, 0, 0, 1, 1, 1], labels), 1)
        np.testing.assert_allclose(values, [0, 0, 1.5, 1.5, 1.5, 1.5], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(embedding, axis=1), 1)
        self.assertEqual(choose_k(values, 4), 2)

        metrics = partition_metrics(adjacency, labels)
        self.assertAlmostEqual(metrics["normalized_cut"], 0)
        self.assertAlmostEqual(metrics["modularity"], 0.5)
        with self.assertRaises(ValueError):
            spectral_clustering(adjacency, 1)

    def test_bridged_cliques(self):
        """Recover two clusters even when a single bridge connects them into one component."""
        adjacency = np.zeros((10, 10))
        clique = np.ones((5, 5)) - np.eye(5)
        adjacency[:5, :5] = clique
        adjacency[5:, 5:] = clique
        adjacency[4, 5] = adjacency[5, 4] = 1

        labels, _, values = spectral_clustering(adjacency, 2)
        self.assertEqual(choose_k(values, 5), 2)
        self.assertAlmostEqual(adjusted_rand_score([0] * 5 + [1] * 5, labels), 1)
        metrics = partition_metrics(adjacency, labels)
        self.assertEqual(metrics["cross_edges"], 1)
        self.assertAlmostEqual(metrics["normalized_cut"], 2 / 21)

    def test_eigenpairs_and_affinity_order(self):
        """Check that the normalized matrices share eigenvectors and use algebraic ordering."""
        # A path is bipartite. Sorting S by magnitude can incorrectly select its eigenvalue -1.
        adjacency = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=float,
        )
        values, vectors = normalized_spectrum(adjacency)
        inverse_sqrt_degrees = 1 / np.sqrt(adjacency.sum(axis=1))
        affinity = inverse_sqrt_degrees[:, None] * adjacency * inverse_sqrt_degrees[None, :]
        np.testing.assert_allclose(affinity @ vectors, vectors * (1 - values), atol=1e-12)
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(4), atol=1e-12)
        self.assertAlmostEqual(values[-1], 2)

    def test_loader_ids_duplicates_weights(self):
        """Keep nonconsecutive IDs, take maximum weights across repeated and reversed records, and remove loops."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "edges.csv"
            path.write_text("10,30,2\n10,30,2\n30,10,1\n30,50,3\n50,50,1\n", encoding="utf-8")
            graph = load_graph(path)

        np.testing.assert_array_equal(graph.nodes, [10, 30, 50])
        np.testing.assert_array_equal(graph.adjacency, [[0, 2, 0], [2, 0, 3], [0, 3, 0]])
        self.assertEqual(graph.stats["duplicate_directed_rows"], 1)
        self.assertEqual(graph.stats["self_loops_removed"], 1)
        self.assertEqual(graph.stats["undirected_edges"], 2)

    def test_invalid_inputs(self):
        """Reject invalid matrices, isolates, and malformed edge lists."""
        invalid_matrices = [
            np.zeros((3, 3)),
            np.ones((3, 2)),
            np.eye(3),
            np.array([[0, 1], [0, 0]]),
            np.array([[0, np.nan], [np.nan, 0]]),
            np.array([[0, -1], [-1, 0]]),
        ]
        for adjacency in invalid_matrices:
            with self.subTest(adjacency=adjacency), self.assertRaises(ValueError):
                normalized_spectrum(adjacency)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            for invalid_input in ("", "1,2,-1", "1,2,nan", "1,2,inf", "1.2,2", "1,2,3,4"):
                path.write_text(invalid_input, encoding="utf-8")
                with self.subTest(invalid_input=invalid_input), self.assertRaises(ValueError):
                    load_graph(path)

    def test_permutation_invariance(self):
        """Reordering nodes preserves cluster membership after mapping back to the original IDs."""
        adjacency = np.zeros((8, 8))
        clique = np.ones((4, 4)) - np.eye(4)
        adjacency[:4, :4] = clique
        adjacency[4:, 4:] = clique
        adjacency[3, 4] = adjacency[4, 3] = 0.1

        original_labels, _, _ = spectral_clustering(adjacency, 2)
        permutation = np.array([3, 7, 0, 5, 2, 6, 1, 4])
        permuted_adjacency = adjacency[np.ix_(permutation, permutation)]
        permuted_labels, _, _ = spectral_clustering(permuted_adjacency, 2)
        self.assertEqual(adjusted_rand_score(original_labels[permutation], permuted_labels), 1)


if __name__ == "__main__":
    unittest.main()
