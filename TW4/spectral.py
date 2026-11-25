"""Ng–Jordan–Weiss spectral clustering, graph construction, and partition metrics."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


@dataclass
class Graph:
    """Store a constructed graph and its input statistics.

    nodes holds the n original IDs in ascending order. Row i and column i of the
    n-by-n adjacency matrix correspond to nodes[i]. stats records input rows,
    duplicates, self-loops, connected components, and isolated nodes.
    """

    nodes: np.ndarray
    adjacency: np.ndarray
    stats: dict


def load_graph(path: str | Path) -> Graph:
    """Build an undirected graph from a headerless two- or three-column CSV.

    Each row of path supplies two integer node IDs and an optional finite positive
    weight. The default weight is 1. Keep the maximum weight across repeated and
    reversed records and discard self-loops. The returned Graph retains the original
    IDs, including nodes introduced only by self-loops, in matrix row and column order.
    """
    edges = {}
    nodes = set()
    rows = 0
    loops = 0
    duplicates = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for line, row in enumerate(csv.reader(handle), 1):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) not in (2, 3):
                raise ValueError(f"{path}:{line}: expected 2 or 3 comma-separated columns")
            try:
                u, v = int(row[0]), int(row[1])
                weight = float(row[2]) if len(row) == 3 else 1.0
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: invalid node ID or weight") from exc
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError(f"{path}:{line}: weight must be finite and positive")
            rows += 1
            nodes.update((u, v))
            if u == v:
                loops += 1
                continue
            if (u, v) in edges:
                duplicates += 1
            edges[u, v] = max(edges.get((u, v), 0.0), weight)
    if not nodes:
        raise ValueError(f"{path}: no vertices found")
    node_ids = np.array(sorted(nodes), dtype=np.int64)
    index = {node: i for i, node in enumerate(node_ids)}
    adjacency = np.zeros((len(node_ids), len(node_ids)), dtype=float)
    for (u, v), weight in edges.items():
        adjacency[index[u], index[v]] = weight
    asymmetric = int(np.count_nonzero(adjacency != adjacency.T) // 2)
    # One undirected edge can occur in both directions. The maximum preserves its weight.
    adjacency = np.maximum(adjacency, adjacency.T)
    component_count, component_labels = connected_components(adjacency, directed=False)
    stats = {
        "input_rows": rows,
        "duplicate_directed_rows": duplicates,
        "self_loops_removed": loops,
        "asymmetric_pairs": asymmetric,
        "nodes": len(node_ids),
        "undirected_edges": int(np.count_nonzero(adjacency) // 2),
        "components": int(component_count),
        "component_sizes": np.bincount(component_labels).tolist(),
        "isolates": int(np.count_nonzero(adjacency.sum(axis=1) == 0)),
    }
    return Graph(node_ids, adjacency, stats)


def normalized_spectrum(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute sorted eigenpairs from a symmetric, nonnegative adjacency matrix with no loops or isolates.

    adjacency is an n-by-n matrix with n >= 2. Return n eigenvalues in ascending
    order and an n-by-n eigenvector matrix. Column j matches eigenvalue j.
    The smallest algebraic eigenvalues of L_sym = I - S correspond to the largest
    algebraic eigenvalues of the paper's normalized affinity matrix S.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1] or len(adjacency) < 2:
        raise ValueError("adjacency must be square with at least two vertices")
    if not np.isfinite(adjacency).all() or np.any(adjacency < 0):
        raise ValueError("adjacency must be finite and nonnegative")
    if not np.allclose(adjacency, adjacency.T) or np.any(np.diag(adjacency) != 0):
        raise ValueError("adjacency must be symmetric with zero diagonal")
    degrees = adjacency.sum(axis=1)
    if np.any(degrees == 0):
        raise ValueError("isolated vertices have undefined normalization; handle them separately")
    # All degrees are positive, so scale rows and columns directly to apply the diagonal factors.
    inverse_sqrt_degrees = 1.0 / np.sqrt(degrees)
    affinity = inverse_sqrt_degrees[:, None] * adjacency * inverse_sqrt_degrees[None, :]
    normalized_laplacian = np.eye(len(adjacency)) - affinity
    return np.linalg.eigh(normalized_laplacian)


def choose_k(eigenvalues: np.ndarray, max_k: int = 10) -> int:
    """Select K by the largest eligible gap in the sorted Laplacian spectrum.

    eigenvalues is the one-dimensional ascending spectrum of L_sym. max_k is the
    search limit. The number of near-zero eigenvalues sets the lower bound, with
    a minimum of 1. Choose the gap between eigenvalues K and K+1, with ties going
    to the smallest K. Raise ValueError if no candidate is available.
    """
    values = np.asarray(eigenvalues)
    components = int(np.count_nonzero(np.abs(values) < 1e-8))
    lower = max(1, components)
    upper = min(max_k, len(values) - 1)
    if upper < lower:
        raise ValueError("max_k must accommodate the number of connected components")
    gaps = np.diff(values)
    return max(range(lower, upper + 1), key=lambda k: gaps[k - 1])


def cluster_eigenvectors(
    eigenvectors: np.ndarray,
    k: int,
    seed: int = 0,
    n_init: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize the rows of the first k eigenvectors and cluster them with K-means.

    Each row of eigenvectors represents a node. Columns follow ascending L_sym
    eigenvalues. k selects both the column count and the cluster count. seed
    controls randomness, and n_init sets the number of initializations per seed.
    Return n cluster labels and an n-by-k embedding with unit-length rows. The input
    must have enough columns, no zero-length rows, and enough distinct points.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer within the available eigenvector count")
    if not 1 <= k <= eigenvectors.shape[1]:
        raise ValueError("k must be an integer within the available eigenvector count")
    if n_init < 1:
        raise ValueError("n_init must be positive")
    # The first k columns give each node k coordinates. Unit row lengths let K-means compare directions.
    coordinates = np.asarray(eigenvectors[:, :k], dtype=float)
    norms = np.linalg.norm(coordinates, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("zero embedding row; k may be smaller than the component count")
    embedding = coordinates / norms[:, None]
    # Small graphs need little work per K-means iteration. One thread avoids startup costs.
    with threadpool_limits(limits=1):
        kmeans = KMeans(n_clusters=int(k), n_init=n_init, random_state=seed)
        labels = kmeans.fit_predict(embedding)
    if len(np.unique(labels)) != k:
        raise ValueError("embedding does not support k distinct clusters")
    return labels, embedding


def spectral_clustering(
    adjacency: np.ndarray,
    k: int,
    seed: int = 0,
    n_init: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run K-eigenvector spectral clustering on an adjacency matrix.

    adjacency must meet the requirements of normalized_spectrum, and k must be at
    least the component count. seed and n_init control K-means initialization.
    Return the labels, unit-row embedding, and ascending eigenvalues. Labels and
    embedding rows follow the input node order.
    """
    values, vectors = normalized_spectrum(adjacency)
    if k < np.count_nonzero(np.abs(values) < 1e-8):
        raise ValueError("k must be at least the number of connected components")
    labels, embedding = cluster_eigenvectors(vectors, k, seed, n_init)
    return labels, embedding, values


def partition_metrics(adjacency: np.ndarray, labels: np.ndarray) -> dict:
    """Compute four partition metrics for an undirected graph and its node labels.

    labels[i] is the cluster of the node in row i of adjacency. Return sorted
    cluster sizes, the undirected cross-edge count, normalized cut, and weighted
    modularity. The cross-edge count counts edges, and Ncut and modularity use
    edge weights. At least one edge and positive volume in every cluster are required.
    """
    adjacency = np.asarray(adjacency)
    labels = np.asarray(labels)
    if labels.shape != (len(adjacency),):
        raise ValueError("one label is required per vertex")
    degrees = adjacency.sum(axis=1)
    total_volume = float(degrees.sum())
    if total_volume <= 0:
        raise ValueError("metrics need at least one edge")
    sizes = []
    normalized_cut = 0.0
    modularity = 0.0
    for label in np.unique(labels):
        inside = labels == label
        volume = float(degrees[inside].sum())
        if volume == 0:
            raise ValueError("a cluster has zero volume")
        # Each internal undirected edge occupies two matrix entries and contributes twice to the degree sum.
        internal_volume = float(adjacency[np.ix_(inside, inside)].sum())
        normalized_cut += (volume - internal_volume) / volume
        modularity += internal_volume / total_volume - (volume / total_volume) ** 2
        sizes.append(int(inside.sum()))
    different_clusters = labels[:, None] != labels[None, :]
    return {
        "cluster_sizes": sorted(sizes),
        "normalized_cut": normalized_cut,
        "modularity": modularity,
        "cross_edges": int(np.count_nonzero(adjacency * different_clusters) // 2),
    }
