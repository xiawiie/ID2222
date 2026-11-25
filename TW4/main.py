"""Run spectral clustering on the two assigned graphs and save results for inspection."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score

from spectral import (
    choose_k,
    cluster_eigenvectors,
    load_graph,
    normalized_spectrum,
    partition_metrics,
)

# Both the CLI and notebook save plots to files. Agg needs no graphical window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_csv(path, rows):
    """Write rows to the UTF-8 CSV at path.

    rows is a nonempty list of dictionaries with the same fields. The first row sets
    the column order.
    """
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_partitions(graph, graph_name, values, vectors, selected_k, max_k, seeds, n_init):
    """Compare candidate values of K and K-means seeds using one set of sorted eigenpairs.

    graph supplies the adjacency matrix in node order, and graph_name identifies it.
    values and vectors come from the same eigendecomposition. max_k bounds the
    experiment, and selected_k is the value to export. Run n_init initializations
    for every seed in the nonempty seeds sequence. Return all experiment rows and
    the first seed's labels for selected_k, in graph node order.
    """
    # Ncut is always zero for K=1. Include K=1 only when the eigengap selects it.
    first_k = 1 if selected_k == 1 else max(2, graph.stats["components"])
    last_k = min(max_k, len(values) - 1)
    records = []
    selected_labels = None
    for k in range(first_k, last_k + 1):
        reference_labels = None
        for seed in seeds:
            started = time.perf_counter()
            labels, _ = cluster_eigenvectors(vectors, k, seed, n_init)
            seconds = time.perf_counter() - started
            if reference_labels is None:
                reference_labels = labels
            if k == selected_k and seed == seeds[0]:
                selected_labels = labels

            # ARI compares cluster memberships even if different runs use different labels.
            metrics = partition_metrics(graph.adjacency, labels)
            records.append({
                "graph": graph_name,
                "k": k,
                "seed": seed,
                **metrics,
                "ari_to_first_seed": adjusted_rand_score(reference_labels, labels),
                "clustering_seconds": seconds,
            })
    if selected_labels is None:
        raise ValueError("at least one seed is required")
    return records, selected_labels


def save_analysis_plot(path, graph, values, vectors, labels, records, selected_k):
    """Save five diagnostic panels as a PNG at path.

    values and vectors are graph's sorted eigenpairs. labels reorder the adjacency
    matrix by cluster. records supply the metric averages per K. selected_k marks
    the eigengap and chosen cluster count. The adjacency plot and the sorted
    eigenvector plot each use their own node order.
    """
    figure, axes = plt.subplot_mosaic(
        [
            ["spectrum", "adjacency"],
            ["fiedler", "adjacency"],
            ["cut", "modularity"],
        ],
        figsize=(10, 8),
        layout="constrained",
    )

    spectrum = axes["spectrum"]
    shown = min(16, len(values))
    spectrum.plot(np.arange(1, shown + 1), values[:shown], "o-", markersize=4)
    spectrum.axvline(
        selected_k + 0.5, color="tab:red", linestyle="--", label=f"Gap after K={selected_k}"
    )
    spectrum.set(
        title="Smallest eigenvalues of L_sym",
        xlabel="Eigenvalue index (1-based)",
        ylabel="Eigenvalue",
    )
    spectrum.legend(loc="best")

    # Keep original node order within each cluster. Red lines mark its row and column boundaries.
    adjacency = axes["adjacency"]
    order = np.argsort(labels, kind="stable")
    adjacency.spy(graph.adjacency[np.ix_(order, order)], markersize=1.15, color="#27486b")
    for boundary in np.flatnonzero(np.diff(labels[order])) + 0.5:
        adjacency.axvline(boundary, color="tab:red", linewidth=0.75)
        adjacency.axhline(boundary, color="tab:red", linewidth=0.75)
    adjacency.xaxis.tick_bottom()
    adjacency.xaxis.set_label_position("bottom")
    adjacency.set(
        title="Adjacency ordered by cluster",
        xlabel="Reordered node",
        ylabel="Reordered node",
    )

    fiedler = axes["fiedler"]
    title = "Sorted second eigenvector"
    if graph.stats["components"] > 1:
        title += " (non-unique)"
    fiedler.plot(np.arange(1, len(values) + 1), np.sort(vectors[:, 1]), ".", markersize=3)
    fiedler.axhline(0, color="#687480", linewidth=0.7)
    fiedler.set(title=title, xlabel="Sorted rank (1-based)", ylabel="Entry")

    # Separate vertical scales keep the Ncut range from flattening the modularity curve.
    k_values = sorted({row["k"] for row in records})
    for panel, metric, title, marker in (
        ("cut", "normalized_cut", "Mean normalized cut", "o"),
        ("modularity", "modularity", "Mean modularity", "s"),
    ):
        means = [
            np.mean([row[metric] for row in records if row["k"] == k])
            for k in k_values
        ]
        axes[panel].plot(k_values, means, marker=marker, markersize=4)
        axes[panel].axvline(selected_k, color="tab:red", linestyle="--", linewidth=0.9)
        axes[panel].set(title=title, xlabel="K", xticks=k_values)
        axes[panel].grid(axis="y", alpha=0.25)

    figure.suptitle(
        f"{path.name.removesuffix('_analysis.png')}: {len(labels)} nodes, K={selected_k}"
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def analyze_graph(path, output_dir, max_k=10, seeds=(0, 1, 2, 3, 4), n_init=20):
    """Read an edge list, run the experiments, save CSV, PNG, and JSON files, and return a summary.

    path is the input edge list, and output_dir receives the tables and figures.
    max_k is the candidate limit, seeds is nonempty, and n_init is the number
    of K-means initializations per seed. Return and save the input hash, graph
    statistics, selected K, partition metrics, and timings.
    """
    started = time.perf_counter()
    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = load_graph(path)
    load_seconds = time.perf_counter() - started
    before_eigendecomposition = time.perf_counter()
    values, vectors = normalized_spectrum(graph.adjacency)
    eigen_seconds = time.perf_counter() - before_eigendecomposition

    selected_k = choose_k(values, max_k)
    records, selected_labels = compare_partitions(
        graph, path.stem, values, vectors, selected_k, max_k, seeds, n_init
    )
    analysis_seconds = time.perf_counter() - started

    save_csv(output_dir / f"{path.stem}_experiments.csv", records)
    save_csv(output_dir / f"{path.stem}_labels.csv", [
        {"node_id": int(node), "cluster": int(label)}
        for node, label in zip(graph.nodes, selected_labels)
    ])
    save_csv(output_dir / f"{path.stem}_spectrum.csv", [
        {
            "index": i + 1,
            "laplacian_eigenvalue": float(value),
            "normalized_affinity_eigenvalue": float(1 - value),
        }
        for i, value in enumerate(values)
    ])
    save_analysis_plot(
        output_dir / f"{path.stem}_analysis.png",
        graph,
        values,
        vectors,
        selected_labels,
        records,
        selected_k,
    )

    _, component_labels = connected_components(graph.adjacency, directed=False)
    summary = {
        "graph": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **graph.stats,
        "selected_k": selected_k,
        "selected_eigengap": float(values[selected_k] - values[selected_k - 1]),
        "k_search_max": max_k,
        "seeds": list(seeds),
        "n_init": n_init,
        "selected_metrics": partition_metrics(graph.adjacency, selected_labels),
        "ari_to_connected_components": adjusted_rand_score(component_labels, selected_labels),
        "load_seconds": load_seconds,
        "eigendecomposition_seconds": eigen_seconds,
        "analysis_seconds_excluding_output": analysis_seconds,
        "total_seconds_including_output": time.perf_counter() - started,
    }
    summary_path = output_dir / f"{path.stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args():
    """Parse CLI arguments with default input and output paths relative to this script."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=root.parent / "datasets" / "TW4",
        help="Directory containing example1.dat and example2.dat",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results", help="Output directory for tables and plots"
    )
    parser.add_argument("--max-k", type=int, default=10, help="Upper bound for the search over K")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Random seeds for K-means",
    )
    parser.add_argument("--n-init", type=int, default=20, help="Number of K-means initializations per seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for filename in ("example1.dat", "example2.dat"):
        result = analyze_graph(
            args.data_dir / filename,
            args.output_dir,
            args.max_k,
            args.seeds,
            args.n_init,
        )
        print(json.dumps(result, indent=2))
