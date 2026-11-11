"""Homework 1 的命令行实验入口。

最常用的运行方式：

    python main.py --sizes 10 100 1000

默认读取项目上一级的 twenty_newsgroups/mini/mini_newsgroups。
全部参数和默认值可以用 python main.py --help 查看。
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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="数据集目录，每个子目录是一个新闻组。默认 ../twenty_newsgroups/mini/mini_newsgroups。",
    )
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
        help="需要测试的文档数量。默认 10 100 1000。",
    )
    parser.add_argument("--k", type=int, default=5, help="字符 shingle 的长度。默认 5。")
    parser.add_argument(
        "--num-hashes",
        type=int,
        default=100,
        help="MinHash signature 的长度。默认 100。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="判断文档相似的 Jaccard 阈值 s。默认 0.3。",
    )
    parser.add_argument(
        "--body-only",
        action="store_true",
        help="只使用正文，不使用 Subject。默认保留 Subject 和正文。",
    )
    parser.add_argument("--lsh", action="store_true", help="运行可选的 LSH 阶段。")
    parser.add_argument(
        "--lsh-bands",
        type=int,
        default=None,
        help="LSH 的 band 数量。默认根据阈值和 signature 长度自动选择。",
    )
    parser.add_argument(
        "--lsh-rows",
        type=int,
        default=None,
        help="每个 band 的行数。必须和 --lsh-bands 一起提供。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results",
        help="保存 CSV 结果的目录。默认 results。",
    )
    return parser.parse_args()


def run_experiment(
    documents,
    shingling: Shingling,
    minhashing: MinHashing,
    threshold: float,
    run_lsh: bool,
    lsh_bands: int | None = None,
    lsh_rows: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """对一组文档运行完整实验。

    返回一个汇总字典和按 exact Jaccard 排序后的文档对记录。
    每个阶段单独计时，便于作业报告分析性能。

    全部文档对的 exact Jaccard 是准确基准。LSH 阶段只使用 signatures，
    不看基准结果，之后再用基准计算它的 precision 和 recall。
    lsh_bands 和 lsh_rows 为 None 时根据阈值自动选择。
    """

    total_start = time.perf_counter()

    # 阶段 1：每篇文档转成 shingle 哈希值的有序集合。
    # 同时保存一份 frozenset，后面比较文档对时直接复用，不用每一对都重建集合。
    shingle_start = time.perf_counter()
    ordered_shingles = {
        document.name: shingling.shingles(document.text) for document in documents
    }
    shingle_sets = {name: frozenset(values) for name, values in ordered_shingles.items()}
    shingle_seconds = time.perf_counter() - shingle_start

    # 阶段 2：每篇文档生成一个 MinHash signature。
    minhash_start = time.perf_counter()
    signatures = {
        name: minhashing.signature(values) for name, values in ordered_shingles.items()
    }
    minhash_seconds = time.perf_counter() - minhash_start

    # 阶段 3：比较全部 n(n-1)/2 个文档对，得到准确基准和 MinHash 估计误差。
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

    # 名称已排序，所以 (first, second) 和 LSH 返回的对使用同一种顺序。
    true_similar_pairs = {
        (row["first"], row["second"])
        for row in pair_records
        if row["exact_jaccard"] >= threshold
    }

    # 阶段 4（可选）：LSH。先用 banding 找候选对，再只对候选对比较 signature。
    bands: int | None = None
    rows: int | None = None
    lsh_seconds = 0.0
    candidate_pairs: set[tuple[str, str]] = set()
    lsh_pairs: set[tuple[str, str]] = set()
    if run_lsh:
        if lsh_bands is None and lsh_rows is None:
            bands, rows = LSH.choose_bands(minhashing.signature_length, threshold)
        elif lsh_bands is None or lsh_rows is None:
            raise ValueError("lsh_bands and lsh_rows must be given together")
        else:
            bands, rows = lsh_bands, lsh_rows

        lsh_start = time.perf_counter()
        candidate_pairs = LSH.candidates(signatures, bands=bands, rows=rows)
        lsh_pairs = {
            pair
            for pair in candidate_pairs
            if CompareSignatures.similarity(signatures[pair[0]], signatures[pair[1]])
            >= threshold
        }
        lsh_seconds = time.perf_counter() - lsh_start

    # 没有运行 LSH，或者分母为 0 时，比例没有意义，用 None 表示。
    candidate_recall = None
    lsh_precision = None
    lsh_recall = None
    true_positives = len(lsh_pairs & true_similar_pairs)
    if run_lsh:
        if true_similar_pairs:
            candidate_recall = len(candidate_pairs & true_similar_pairs) / len(
                true_similar_pairs
            )
            lsh_recall = true_positives / len(true_similar_pairs)
        if lsh_pairs:
            lsh_precision = true_positives / len(lsh_pairs)

    total_seconds = time.perf_counter() - total_start
    summary = {
        "documents": len(documents),
        "pairs": len(pair_records),
        "similar_pairs": len(true_similar_pairs),
        "lsh_bands": bands,
        "lsh_rows": rows,
        "lsh_candidates": len(candidate_pairs) if run_lsh else None,
        "lsh_candidate_recall": candidate_recall,
        "lsh_similar_pairs": len(lsh_pairs) if run_lsh else None,
        "lsh_true_positives": true_positives if run_lsh else None,
        "lsh_precision": lsh_precision,
        "lsh_recall": lsh_recall,
        "shingling_seconds": shingle_seconds,
        "minhash_seconds": minhash_seconds,
        "pair_comparison_seconds": pair_seconds,
        "lsh_seconds": lsh_seconds,
        "total_seconds": total_seconds,
    }
    pair_records.sort(key=lambda row: row["exact_jaccard"], reverse=True)
    return summary, pair_records


def format_ratio(value: float | None) -> str:
    """把比例格式化为四位小数。None 表示分母为 0，显示为 n/a。"""

    return "n/a" if value is None else f"{value:.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """将字典列表写入 CSV。None 会写成空单元格。"""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        # 固定使用 LF 换行，Windows 和 macOS 生成的文件完全一致。
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    if args.k <= 0 or args.num_hashes <= 0:
        raise SystemExit("--k and --num-hashes must be positive")
    if any(size <= 0 for size in args.sizes):
        raise SystemExit("Every value in --sizes must be positive")
    if (args.lsh_bands is None) != (args.lsh_rows is None):
        raise SystemExit("--lsh-bands and --lsh-rows must be given together")
    if args.lsh_bands is not None:
        if args.lsh_bands <= 0 or args.lsh_rows <= 0:
            raise SystemExit("--lsh-bands and --lsh-rows must be positive")
        if args.lsh_bands * args.lsh_rows != args.num_hashes:
            raise SystemExit("--lsh-bands * --lsh-rows must equal --num-hashes")
    if not args.data_dir.is_dir():
        raise SystemExit(
            f"Dataset directory does not exist: {args.data_dir}\n"
            "See README.md for how to download the dataset."
        )

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
        actual_size = min(requested_size, len(documents))
        if actual_size < requested_size:
            print(
                f"\nRequested {requested_size} documents, "
                f"but only {actual_size} are available."
            )
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
        if not args.lsh:
            # 没有运行 LSH 时不输出 LSH 的列，避免 CSV 里出现一排空单元格。
            summary = {
                key: value for key, value in summary.items() if not key.startswith("lsh_")
            }
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
                f"LSH (bands={summary['lsh_bands']}, rows={summary['lsh_rows']}): "
                f"candidates={summary['lsh_candidates']}, "
                f"candidate recall={format_ratio(summary['lsh_candidate_recall'])}, "
                f"time={summary['lsh_seconds']:.4f}s"
            )
            print(
                f"LSH pairs with signature similarity >= threshold: "
                f"{summary['lsh_similar_pairs']}, "
                f"precision={format_ratio(summary['lsh_precision'])}, "
                f"recall={format_ratio(summary['lsh_recall'])}"
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
