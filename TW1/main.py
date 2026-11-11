"""Homework 1 的命令行实验入口。

最常用的运行方式：

    python main.py --sizes 10 100 1000 --lsh

默认读取项目上一级的 twenty_newsgroups/mini/mini_newsgroups。
"""

from __future__ import annotations

import argparse
import csv
import time
from itertools import combinations
from pathlib import Path
from typing import Any

from similarity import (
    CompareSets,
    CompareSignatures,
    LSH,
    MinHashing,
    Shingling,
    balanced_sample,
    load_documents,
)


def parse_args() -> argparse.Namespace:
    """定义命令行参数，并给出适合本项目的默认值。"""

    project_dir = Path(__file__).resolve().parent
    default_data_dir = project_dir.parent / "twenty_newsgroups" / "mini" / "mini_newsgroups"

    parser = argparse.ArgumentParser(
        description="Find textually similar documents with Jaccard, MinHash and LSH."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="只读取指定新闻组。不提供时读取全部新闻组。",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[10, 100, 1000],
        help="需要测试的文档数量。",
    )
    parser.add_argument("--k", type=int, default=5, help="字符 shingle 的长度。")
    parser.add_argument("--num-hashes", type=int, default=100)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="判断文档相似的 Jaccard 阈值。",
    )
    parser.add_argument(
        "--body-only",
        action="store_true",
        help="只使用正文，不使用 Subject。默认保留 Subject 和正文。",
    )
    parser.add_argument("--lsh", action="store_true", help="运行可选的 LSH 阶段。")
    parser.add_argument("--lsh-bands", type=int, default=20)
    parser.add_argument("--lsh-rows", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results",
        help="保存 CSV 结果的目录。",
    )
    return parser.parse_args()


def run_experiment(
    documents,
    shingling: Shingling,
    minhashing: MinHashing,
    threshold: float,
    run_lsh: bool,
    lsh_bands: int,
    lsh_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """对一组文档运行完整实验。

    返回一个汇总字典和按 exact Jaccard 排序后的文档对记录。
    每个阶段单独计时，便于作业报告分析性能。
    """

    total_start = time.perf_counter()

    shingle_start = time.perf_counter()
    shingle_sets = {
        document.name: shingling.shingles(document.text) for document in documents
    }
    shingle_seconds = time.perf_counter() - shingle_start

    minhash_start = time.perf_counter()
    signatures = {
        name: minhashing.signature(values) for name, values in shingle_sets.items()
    }
    minhash_seconds = time.perf_counter() - minhash_start

    pair_start = time.perf_counter()
    pair_records: list[dict[str, Any]] = []
    for first, second in combinations(sorted(shingle_sets), 2):
        exact = CompareSets.jaccard(shingle_sets[first], shingle_sets[second])
        estimated = CompareSignatures.similarity(signatures[first], signatures[second])
        pair_records.append(
            {
                "first": first,
                "second": second,
                "exact_jaccard": exact,
                "minhash_estimate": estimated,
                "absolute_error": abs(exact - estimated),
            }
        )
    pair_seconds = time.perf_counter() - pair_start

    true_similar_pairs = {
        tuple(sorted((row["first"], row["second"])))
        for row in pair_records
        if row["exact_jaccard"] >= threshold
    }

    lsh_seconds = 0.0
    lsh_pairs: set[tuple[str, str]] = set()
    if run_lsh:
        lsh_start = time.perf_counter()
        lsh_pairs = LSH.candidates(signatures, bands=lsh_bands, rows=lsh_rows)
        lsh_seconds = time.perf_counter() - lsh_start

    true_positives = len(lsh_pairs & true_similar_pairs)
    lsh_recall = (
        true_positives / len(true_similar_pairs) if true_similar_pairs else 0.0
    )

    total_seconds = time.perf_counter() - total_start
    summary = {
        "documents": len(documents),
        "pairs": len(pair_records),
        "similar_pairs": len(true_similar_pairs),
        "lsh_candidates": len(lsh_pairs),
        "lsh_true_positives": true_positives,
        "lsh_recall": lsh_recall,
        "shingling_seconds": shingle_seconds,
        "minhash_seconds": minhash_seconds,
        "pair_comparison_seconds": pair_seconds,
        "lsh_seconds": lsh_seconds,
        "total_seconds": total_seconds,
    }
    pair_records.sort(key=lambda row: row["exact_jaccard"], reverse=True)
    return summary, pair_records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """将字典列表写入 CSV。"""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")

    documents = load_documents(
        args.data_dir,
        categories=args.categories,
        include_subject=not args.body_only,
    )
    if not documents:
        raise SystemExit("No documents found. Check --data-dir and --categories.")

    shingling = Shingling(args.k)
    minhashing = MinHashing(args.num_hashes)
    print(f"Dataset: {args.data_dir}")
    print(f"Available documents: {len(documents)}")
    print(f"k={args.k}, signature length={args.num_hashes}, threshold={args.threshold}")
    print(f"Subject included: {not args.body_only}")
    if args.categories:
        print(f"Categories: {', '.join(args.categories)}")

    summaries: list[dict[str, Any]] = []
    top_pairs: list[dict[str, Any]] = []
    for requested_size in args.sizes:
        if requested_size <= 0:
            raise SystemExit("Every value in --sizes must be positive")

        actual_size = min(requested_size, len(documents))
        selected = balanced_sample(documents, actual_size)
        summary, pair_records = run_experiment(
            selected,
            shingling,
            minhashing,
            args.threshold,
            args.lsh,
            args.lsh_bands,
            args.lsh_rows,
        )
        summary["requested_documents"] = requested_size
        summaries.append(summary)

        for row in pair_records[:10]:
            top_pairs.append(
                {
                    "documents": actual_size,
                    "first": row["first"],
                    "second": row["second"],
                    "exact_jaccard": row["exact_jaccard"],
                    "minhash_estimate": row["minhash_estimate"],
                    "absolute_error": row["absolute_error"],
                }
            )

        print(
            f"\nDocuments: {actual_size} | pairs: {summary['pairs']} | "
            f"total time: {summary['total_seconds']:.4f}s"
        )
        print(f"Exact pairs >= threshold: {summary['similar_pairs']}")
        print(
            f"Stage time: shingling={summary['shingling_seconds']:.4f}s, "
            f"minhash={summary['minhash_seconds']:.4f}s, "
            f"pair comparison={summary['pair_comparison_seconds']:.4f}s"
        )
        if args.lsh:
            print(
                f"LSH candidates: {summary['lsh_candidates']}, "
                f"recall={summary['lsh_recall']:.4f}"
            )
        if actual_size <= 10:
            print("Top exact-Jaccard pairs:")
            for row in pair_records[:5]:
                print(
                    f"  {row['first']} <-> {row['second']}: "
                    f"exact={row['exact_jaccard']:.4f}, "
                    f"minhash={row['minhash_estimate']:.4f}"
                )

    write_csv(args.output_dir / "experiment_summary.csv", summaries)
    write_csv(args.output_dir / "top_pairs.csv", top_pairs)
    print(f"\nResults written to: {args.output_dir}")


if __name__ == "__main__":
    main()
