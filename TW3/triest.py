"""Reservoir sampling, TRIÈST insertion-stream estimates, and the offline baseline.

BASE and IMPR correspond to Sections 4.1 and 4.2 of De Stefani et al. (KDD 2016).
"""
from __future__ import annotations

import gzip
import random
from collections import Counter
from pathlib import Path


class Reservoir:
    """Keep a uniform sample of at most M stream items using Algorithm R."""

    def __init__(self, capacity: int, seed: int = 0):
        """Start with an empty sample and a processed-item count of zero.

        capacity is a positive integer. The same seed and input order reproduce replacements.
        """
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.items = []
        self.t = 0

    def update(self, item):
        """Process one item, starting by incrementing the processed-item count.

        Return (accepted, removed_item). Appending or skipping leaves removed_item as None.
        """
        self.t += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return True, None

        # The same uniform draw decides whether to accept (M/t) and which slot to replace.
        slot = self.rng.randrange(self.t)
        if slot >= self.capacity:
            return False, None
        removed = self.items[slot]
        self.items[slot] = item
        return True, removed


class Triest:
    """Estimate global or local triangle counts in a simple undirected insertion stream.

    Global counting uses O(M) space. With local counts enabled, BASE still uses
    O(M) space, while IMPR may retain a count for every vertex seen so far.
    """

    def __init__(self, capacity: int, seed: int = 0, variant="base", local=False):
        """Initialize the reservoir and adjacency sets, and reset the triangle counters.

        capacity is at least 6. variant is base or impr. local enables per-vertex counts.
        """
        if capacity < 6:
            raise ValueError("TRIEST uses capacity >= 6, as in the paper")
        if variant not in ("base", "impr"):
            raise ValueError("variant must be base or impr")
        self.reservoir = Reservoir(capacity, seed)
        self.variant = variant
        self.adjacency = {}
        self.counter = 0
        self.local = Counter() if local else None

    def _count(self, u, v, weight):
        """Add a weight for every triangle that (u, v) closes in the current sample.

        Update the global count and, if enabled, the local counts. A negative weight
        subtracts the contribution of an outgoing edge.
        """
        neighbors_u = self.adjacency.get(u)
        neighbors_v = self.adjacency.get(v)
        if neighbors_u is None or neighbors_v is None:
            return

        common_neighbors = neighbors_u & neighbors_v
        self.counter += weight * len(common_neighbors)
        if self.local is not None:
            for neighbor in common_neighbors:
                for node in (u, v, neighbor):
                    self.local[node] += weight
                    # Drop zero counts after BASE evictions to avoid retaining historical vertices.
                    if self.local[node] == 0:
                        del self.local[node]

    def update(self, u: int, v: int):
        """Process one undirected edge with distinct integer endpoints and return the global estimate.

        The caller supplies each edge once. BASE adjusts counts as the sample changes.
        IMPR adds contributions before sampling.
        """
        if not isinstance(u, int) or not isinstance(v, int) or u == v:
            raise ValueError("endpoints must be distinct integer vertex IDs")
        u, v = min(u, v), max(u, v)
        if v in self.adjacency.get(u, ()):
            raise ValueError("duplicate sampled edge; deduplicate the input first")

        if self.variant == "impr":
            # The closing edge has arrived. Correct for the other two edges being in the old sample.
            previous_size = self.reservoir.t
            capacity = self.reservoir.capacity
            weight = max(1, previous_size * (previous_size - 1) / (capacity * (capacity - 1)))
            self._count(u, v, weight)

        accepted, removed = self.reservoir.update((u, v))
        if not accepted:
            return self.estimate()

        if removed is not None:
            old_u, old_v = removed
            if self.variant == "base":
                self._count(old_u, old_v, -1)
            for node, neighbor in ((old_u, old_v), (old_v, old_u)):
                self.adjacency[node].remove(neighbor)
                if not self.adjacency[node]:
                    del self.adjacency[node]

        if self.variant == "base":
            # Count triangles closed by the new edge after removing the outgoing edge.
            self._count(u, v, 1)
        self.adjacency.setdefault(u, set()).add(v)
        self.adjacency.setdefault(v, set()).add(u)
        return self.estimate()

    def estimate(self, node=None):
        """Return the current estimate as a float. With node=None, return the global estimate.

        A vertex query requires local counting. A vertex with no recorded count returns zero.
        """
        if node is not None and self.local is None:
            raise ValueError("enable local=True to query local counts")

        count = self.counter if node is None else self.local.get(node, 0)
        processed = self.reservoir.t
        capacity = self.reservoir.capacity
        if self.variant == "base" and processed > capacity:
            all_triples = processed * (processed - 1) * (processed - 2)
            sampled_triples = capacity * (capacity - 1) * (capacity - 2)
            # Multiply by the reciprocal of the probability that all three edges are sampled.
            count *= all_triples / sampled_triples
        return float(count)


def read_graph(path: Path):
    """Read a plain or gzip two-column edge file and return simple undirected edges and cleaning statistics.

    Keep edges in order of first appearance. Count vertices from the original endpoints.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    edges = []
    seen = set()
    nodes = set()
    stats = dict(raw_edges=0, self_loops=0, duplicate_edges=0)

    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.replace(",", " ").split()
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected two endpoints")
            try:
                u, v = map(int, fields)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: non-integer endpoint") from error

            stats["raw_edges"] += 1
            nodes.update((u, v))
            if u == v:
                stats["self_loops"] += 1
                continue

            edge = (min(u, v), max(u, v))
            if edge in seen:
                stats["duplicate_edges"] += 1
                continue
            seen.add(edge)
            edges.append(edge)

    stats.update(nodes=len(nodes), edges=len(edges))
    return edges, stats


def exact_triangles(edges):
    """Count triangles offline and return the exact total and a per-vertex count table.

    Omit vertices in no triangles. This check uses the complete edge set.
    """
    forward = {}
    for u, v in edges:
        u, v = min(u, v), max(u, v)
        forward.setdefault(u, set()).add(v)

    total = 0
    local = Counter()
    # Visit endpoints in the order u < v < w to count each triangle exactly once.
    for u, neighbors in forward.items():
        for v in neighbors:
            if v in forward:
                for w in neighbors & forward[v]:
                    total += 1
                    local.update((u, v, w))
    return total, local
