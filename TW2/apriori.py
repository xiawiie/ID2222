"""一个适合学习和现场演示的 A-Priori 算法实现。

作业要求我们在交易数据库中寻找频繁项集。本文件只使用普通 Python，
让数据结构、候选生成、剪枝和支持度统计都可以直接阅读和解释。

The main public functions are useful from both the command line and the
Jupyter notebook:

* ``read_transactions`` reads one transaction per line.
* ``run_apriori`` finds frequent itemsets and records timing information.
* ``generate_association_rules`` implements the optional bonus task.
* ``write_results`` saves reproducible CSV and JSON result files.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence


Item = str
Itemset = frozenset[Item]
Transactions = list[Itemset]
FrequentItemsets = dict[int, dict[Itemset, int]]


def _validate_support(min_support: int) -> None:
    """Reject invalid support counts before reading a potentially large file."""

    if isinstance(min_support, bool) or not isinstance(min_support, int):
        raise ValueError("min_support must be a positive integer")
    if min_support <= 0:
        raise ValueError("min_support must be a positive integer")


def _validate_confidence(min_confidence: float) -> None:
    """Check that confidence is a probability in the closed interval [0, 1]."""

    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")


def read_transactions(dataset_path: str | Path) -> Transactions:
    """读取交易文件，每一行表示一个 basket。

    T10I4D100K 文件使用空格分隔项目。转换为 set 很重要，因为支持度统计
    的是包含该 itemset 的交易数量。同一行中重复出现的项目不能重复计数。
    """

    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    transactions: Transactions = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            items = line.split()
            if not items:
                # Empty lines do not represent a useful basket.
                continue
            transaction = frozenset(items)
            transactions.append(transaction)

    if not transactions:
        raise ValueError(f"Dataset contains no non-empty transactions: {path}")
    return transactions


def _sorted_items(itemset: Iterable[Item]) -> tuple[Item, ...]:
    """Return a stable representation for output and candidate generation."""

    return tuple(sorted(itemset))


def format_itemset(itemset: Itemset) -> str:
    """Format an itemset in a compact, deterministic form for CSV output."""

    return "{" + ", ".join(_sorted_items(itemset)) + "}"


def _count_single_items(transactions: Transactions) -> dict[Itemset, int]:
    """Count each item once per transaction."""

    counts: dict[Itemset, int] = {}
    for transaction in transactions:
        for item in transaction:
            singleton = frozenset((item,))
            counts[singleton] = counts.get(singleton, 0) + 1
    return counts


def generate_candidates(
    previous_frequent: Mapping[Itemset, int],
    itemset_size: int,
) -> set[Itemset]:
    """生成下一层候选项集，并使用 A-Priori 性质进行剪枝。

    两个上一层频繁项集先进行合并。如果候选项集的任意一个直接子集不频繁，
    这个候选项集就不可能频繁，因此可以在真正计数前删除。
    """

    previous = set(previous_frequent)
    if itemset_size < 2 or not previous:
        return set()

    previous_list = sorted(previous, key=_sorted_items)
    candidates: set[Itemset] = set()

    for index, left in enumerate(previous_list):
        for right in previous_list[index + 1 :]:
            union = left | right
            if len(union) != itemset_size:
                continue

            all_subsets_are_frequent = all(
                frozenset(subset) in previous
                for subset in combinations(union, itemset_size - 1)
            )
            if all_subsets_are_frequent:
                candidates.add(union)

    return candidates


def count_candidate_support(
    transactions: Transactions,
    candidates: set[Itemset],
    itemset_size: int,
) -> dict[Itemset, int]:
    """通过枚举每个 basket 中的组合来统计候选项集支持度。

    交易在进入本函数前已经转换为 set，所以一个候选项集在同一条交易中
    最多计数一次。把生成的组合放入候选集合查找，可以避免对每条交易逐一
    扫描全部候选项集。
    """

    counts = {candidate: 0 for candidate in candidates}
    if not candidates:
        return counts

    for transaction in transactions:
        if len(transaction) < itemset_size:
            continue
        for combination_items in combinations(_sorted_items(transaction), itemset_size):
            candidate = frozenset(combination_items)
            if candidate in candidates:
                counts[candidate] += 1

    return counts


def run_apriori(
    transactions: Transactions,
    min_support: int,
) -> dict[str, object]:
    """运行 A-Priori，并返回频繁项集、支持度和运行时间。

    返回普通字典，方便 Notebook 查看，也方便命令行程序保存为 JSON。
    ``frequent_itemsets`` 按 itemset 大小保存各层结果，``support_counts``
    把所有层的支持度合并到一个字典中。
    """

    _validate_support(min_support)
    if not transactions:
        raise ValueError("transactions must contain at least one transaction")

    overall_start = time.perf_counter()
    frequent_by_size: FrequentItemsets = {}
    support_counts: dict[Itemset, int] = {}
    level_timings: list[dict[str, float | int]] = []

    level_start = time.perf_counter()
    singleton_counts = _count_single_items(transactions)
    frequent_singletons = {
        itemset: count
        for itemset, count in singleton_counts.items()
        if count >= min_support
    }
    frequent_singletons = dict(sorted(frequent_singletons.items(), key=lambda pair: _sorted_items(pair[0])))
    frequent_by_size[1] = frequent_singletons
    support_counts.update(frequent_singletons)
    level_timings.append(
        {
            "itemset_size": 1,
            "candidate_count": len(singleton_counts),
            "frequent_count": len(frequent_singletons),
            "seconds": time.perf_counter() - level_start,
        }
    )

    previous_frequent = frequent_singletons
    itemset_size = 2
    while previous_frequent:
        level_start = time.perf_counter()
        candidates = generate_candidates(previous_frequent, itemset_size)
        candidate_counts = count_candidate_support(transactions, candidates, itemset_size)
        current_frequent = {
            itemset: count
            for itemset, count in candidate_counts.items()
            if count >= min_support
        }
        current_frequent = dict(sorted(current_frequent.items(), key=lambda pair: _sorted_items(pair[0])))

        if current_frequent:
            frequent_by_size[itemset_size] = current_frequent
            support_counts.update(current_frequent)

        level_timings.append(
            {
                "itemset_size": itemset_size,
                "candidate_count": len(candidates),
                "frequent_count": len(current_frequent),
                "seconds": time.perf_counter() - level_start,
            }
        )

        previous_frequent = current_frequent
        itemset_size += 1

    return {
        "frequent_itemsets": frequent_by_size,
        "support_counts": support_counts,
        "level_timings": level_timings,
        "transaction_count": len(transactions),
        "min_support": min_support,
        "total_seconds": time.perf_counter() - overall_start,
    }


def generate_association_rules(
    frequent_itemsets: FrequentItemsets,
    support_counts: Mapping[Itemset, int],
    min_support: int,
    min_confidence: float,
) -> list[dict[str, object]]:
    """Generate association rules for the optional bonus task."""

    _validate_support(min_support)
    _validate_confidence(min_confidence)

    rules: list[dict[str, object]] = []
    for itemset_size, itemsets in frequent_itemsets.items():
        if itemset_size < 2:
            continue

        for itemset, itemset_support in itemsets.items():
            sorted_itemset = _sorted_items(itemset)
            for antecedent_size in range(1, itemset_size):
                for antecedent_tuple in combinations(sorted_itemset, antecedent_size):
                    antecedent = frozenset(antecedent_tuple)
                    consequent = itemset - antecedent
                    antecedent_support = support_counts.get(antecedent)
                    if not antecedent_support:
                        # This should not happen for a valid A-Priori result,
                        # but the guard avoids division by zero for bad input.
                        continue

                    confidence = itemset_support / antecedent_support
                    if itemset_support < min_support or confidence < min_confidence:
                        continue

                    rules.append(
                        {
                            "antecedent": antecedent,
                            "consequent": consequent,
                            "support": itemset_support,
                            "confidence": confidence,
                        }
                    )

    rules.sort(
        key=lambda rule: (
            -float(rule["confidence"]),
            -int(rule["support"]),
            _sorted_items(rule["antecedent"]),
            _sorted_items(rule["consequent"]),
        )
    )
    return rules


def write_results(
    output_dir: str | Path,
    result: Mapping[str, object],
    rules: Sequence[Mapping[str, object]] | None = None,
) -> None:
    """Write human-readable result files for the report and presentation."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frequent_by_size = result["frequent_itemsets"]
    if not isinstance(frequent_by_size, dict):
        raise TypeError("result['frequent_itemsets'] must be a dictionary")

    with (output_path / "frequent_itemsets.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["itemset_size", "itemset", "support"])
        for itemset_size in sorted(frequent_by_size):
            itemsets = frequent_by_size[itemset_size]
            for itemset, support in itemsets.items():
                writer.writerow([itemset_size, format_itemset(itemset), support])

    association_rules_path = output_path / "association_rules.csv"
    if rules is not None:
        with association_rules_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["antecedent", "consequent", "support", "confidence"])
            for rule in rules:
                writer.writerow(
                    [
                        format_itemset(rule["antecedent"]),
                        format_itemset(rule["consequent"]),
                        rule["support"],
                        f"{float(rule['confidence']):.6f}",
                    ]
                )
    elif association_rules_path.exists():
        # A reused output directory must not keep rules from an older run.
        association_rules_path.unlink()

    summary = {
        "transaction_count": result["transaction_count"],
        "min_support": result["min_support"],
        "total_seconds": result["total_seconds"],
        "level_timings": result["level_timings"],
        "frequent_itemset_count": sum(len(level) for level in frequent_by_size.values()),
        "association_rule_count": None if rules is None else len(rules),
    }
    with (output_path / "run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def _positive_integer(value: str) -> int:
    """argparse converter with a useful error message."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("support must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("support must be a positive integer")
    return parsed


def _confidence_value(value: str) -> float:
    """argparse converter for a confidence probability."""

    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("confidence must be a number") from error
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("confidence must be between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser in one place for easy inspection."""

    parser = argparse.ArgumentParser(
        description="Find frequent itemsets with the A-Priori algorithm."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a transaction file with one space-separated basket per line.",
    )
    parser.add_argument(
        "--support",
        required=True,
        type=_positive_integer,
        help="Minimum support count measured in transactions.",
    )
    parser.add_argument(
        "--confidence",
        type=_confidence_value,
        help="Optional minimum confidence for association rules, from 0 to 1.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for CSV and JSON output files. Default: results",
    )
    return parser


def main() -> None:
    """Run the complete command-line workflow."""

    args = build_parser().parse_args()
    try:
        transactions = read_transactions(args.dataset)
        result = run_apriori(transactions, args.support)

        rules = None
        if args.confidence is not None:
            rules = generate_association_rules(
                result["frequent_itemsets"],
                result["support_counts"],
                args.support,
                args.confidence,
            )

        write_results(args.output_dir, result, rules)
    except (FileNotFoundError, ValueError) as error:
        # User input errors should be understandable without reading a traceback.
        raise SystemExit(f"Error: {error}") from error

    frequent_count = sum(
        len(level) for level in result["frequent_itemsets"].values()
    )
    print(f"Transactions: {result['transaction_count']}")
    print(f"Minimum support: {args.support}")
    print(f"Frequent itemsets: {frequent_count}")
    if rules is not None:
        print(f"Minimum confidence: {args.confidence}")
        print(f"Association rules: {len(rules)}")
    print(f"Total runtime: {result['total_seconds']:.3f} seconds")
    print(f"Results written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
