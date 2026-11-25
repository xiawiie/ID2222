"""Run TRIÈST experiments on public graphs and save individual runs and summaries."""
import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

from triest import Triest, exact_triangles, read_graph

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "datasets" / "TW3"
DEFAULT_PATHS = (DATA_DIR / "ca-GrQc.txt.gz", DATA_DIR / "p2p-Gnutella08.txt.gz")
DEFAULT_CAPACITIES = (500, 2000, 5000)
DEFAULT_SEEDS = tuple(range(10))
DEFAULT_ORDER_SEED = 42
DEFAULT_OUTPUT_DIR = HERE / "results"


def run_experiments(
    paths=DEFAULT_PATHS,
    capacities=DEFAULT_CAPACITIES,
    seeds=DEFAULT_SEEDS,
    output_dir=DEFAULT_OUTPUT_DIR,
    order_seed=DEFAULT_ORDER_SEED,
):
    """Run both variants on each file with the given capacities, seeds, and edge order.

    Write CSV results and environment details to output_dir. Return individual runs,
    summaries, and dataset statistics. For each graph, add a capacity large enough to
    hold every edge and check each full-capacity estimate against the exact count.
    """
    paths = list(paths)
    capacities = list(capacities)
    seeds = list(seeds)
    if not paths or not capacities or not seeds or any(capacity < 6 for capacity in capacities):
        raise ValueError("provide datasets, capacities >= 6 and at least one seed")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = []
    metadata = []

    for path in map(Path, paths):
        started = time.perf_counter()
        edges, stats = read_graph(path)
        preprocess_seconds = time.perf_counter() - started

        started = time.perf_counter()
        truth, _ = exact_triangles(edges)
        exact_seconds = time.perf_counter() - started

        # Both variants see the same stream. Edge order and sampling use separate seeds.
        random.Random(order_seed).shuffle(edges)
        metadata.append(dict(
            dataset=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            **stats,
            exact_triangles=truth,
            preprocess_seconds=preprocess_seconds,
            exact_seconds=exact_seconds,
            order_seed=order_seed,
        ))

        # Both variants must return the exact count when the sample can hold every edge.
        full_capacity = max(6, len(edges))
        for capacity in sorted(set(capacities + [full_capacity])):
            for variant in ("base", "impr"):
                group = []
                for seed in seeds:
                    algorithm = Triest(capacity, seed, variant)
                    started = time.perf_counter()
                    for u, v in edges:
                        algorithm.update(u, v)
                    seconds = time.perf_counter() - started

                    estimate = algorithm.estimate()
                    sampled_edges = len(algorithm.reservoir.items)
                    assert sampled_edges <= capacity
                    if capacity >= len(edges):
                        assert estimate == truth

                    absolute_error = abs(estimate - truth)
                    group.append(dict(
                        dataset=path.name,
                        variant=variant,
                        capacity=capacity,
                        seed=seed,
                        edges=len(edges),
                        exact=truth,
                        estimate=estimate,
                        absolute_error=absolute_error,
                        relative_error=absolute_error / truth if truth else "",
                        seconds=seconds,
                        sampled_edges=sampled_edges,
                        sampled_vertices=len(algorithm.adjacency),
                    ))

                rows.extend(group)
                estimates = [row["estimate"] for row in group]
                # Relative error is undefined when the exact count is zero. Leave the CSV field blank.
                summary.append(dict(
                    dataset=path.name,
                    variant=variant,
                    capacity=capacity,
                    runs=len(group),
                    exact=truth,
                    mean_estimate=statistics.mean(estimates),
                    std_estimate=statistics.stdev(estimates) if len(estimates) > 1 else 0,
                    mean_relative_error=(
                        statistics.mean(row["relative_error"] for row in group) if truth else ""
                    ),
                    mean_seconds=statistics.mean(row["seconds"] for row in group),
                ))

    summary.sort(key=lambda row: (row["dataset"], row["variant"], row["capacity"]))
    for name, records in (
        ("runs.csv", rows),
        ("summary.csv", summary),
        ("datasets.csv", metadata),
    ):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "timing": "perf_counter; streaming loop only; excludes loading, shuffling, "
                  "exact baseline and output",
        "memory": "global-only estimator O(M); offline preprocessing and baseline O(E+V)",
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    return rows, summary, metadata


def parse_args():
    """Parse command-line arguments for the experiment runner.

        The notebook shares these defaults. argparse displays usage when given --help.
        """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", nargs="+", type=Path, default=DEFAULT_PATHS,
        help="two-column edge files, optionally compressed with gzip",
    )
    parser.add_argument(
        "--capacities", nargs="+", type=int, default=DEFAULT_CAPACITIES,
        help="reservoir capacities, each at least 6",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="sampling seeds, one run per seed",
    )
    parser.add_argument(
        "--order-seed", type=int, default=DEFAULT_ORDER_SEED,
        help="seed for the edge order shared by both algorithms",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="directory for generated results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runs, summary, metadata = run_experiments(
        args.data, args.capacities, args.seeds, args.output_dir, args.order_seed
    )
    for dataset in metadata:
        print(
            f"{dataset['dataset']}: {dataset['edges']:,} edges, "
            f"{dataset['exact_triangles']:,} exact triangles"
        )
    print(f"{len(runs)} runs, {len(summary)} summary rows. Results: {args.output_dir.resolve()}")
