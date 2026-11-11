"""Small correctness checks for the A-Priori implementation.

Run with:  python test_apriori.py

The checks do not need the T10I4D100K dataset.  They use a tiny in-memory
dataset with manually calculable results, and random small datasets whose
results are compared with a brute-force enumeration of every itemset.
"""

import random
import tempfile
from itertools import combinations
from pathlib import Path

from apriori import (
    count_candidate_support,
    format_itemset,
    generate_association_rules,
    generate_candidates,
    read_transactions,
    run_apriori,
)


def check_manual_example() -> None:
    """支持度可以人工计算的四条交易。"""

    transactions = [
        frozenset({"A", "B", "C"}),
        frozenset({"A", "B"}),
        frozenset({"A", "C"}),
        frozenset({"B", "C"}),
    ]

    result = run_apriori(transactions, min_support=2)
    frequent = result["frequent_itemsets"]
    supports = result["support_counts"]

    assert supports[frozenset({"A"})] == 3
    assert supports[frozenset({"B"})] == 3
    assert supports[frozenset({"C"})] == 3
    assert supports[frozenset({"A", "B"})] == 2
    assert supports[frozenset({"A", "C"})] == 2
    assert supports[frozenset({"B", "C"})] == 2
    assert 3 not in frequent, "The triple should be removed because support is 1"

    rules = generate_association_rules(
        frequent,
        supports,
        min_support=2,
        min_confidence=0.6,
    )
    # 每个 2-项集给出两条规则，置信度都是 2/3。
    assert len(rules) == 6
    for rule in rules:
        assert rule["support"] == 2
        assert abs(rule["confidence"] - 2 / 3) < 1e-12
        assert not rule["antecedent"] & rule["consequent"]
    rule_pairs = {(rule["antecedent"], rule["consequent"]) for rule in rules}
    assert (frozenset({"A"}), frozenset({"B"})) in rule_pairs

    # 置信度阈值高于 2/3 时不应该有任何规则。
    assert generate_association_rules(frequent, supports, 2, 0.7) == []


def check_candidate_generation() -> None:
    """Join 和 prune 两步都要正确。"""

    def itemsets(*groups: str) -> dict[frozenset[str], int]:
        return {frozenset(group): 1 for group in groups}

    # AB、AC、BC 都频繁，所以 ABC 是候选。
    assert generate_candidates(itemsets("AB", "AC", "BC"), 3) == {frozenset("ABC")}
    # BC 不频繁，ABC 在计数前就被剪掉。
    assert generate_candidates(itemsets("AB", "AC"), 3) == set()
    # 1-项集两两组合成 2-项集候选。
    assert generate_candidates(itemsets("A", "B", "C"), 2) == {
        frozenset("AB"),
        frozenset("AC"),
        frozenset("BC"),
    }
    assert generate_candidates({}, 2) == set()


def check_counting_paths() -> None:
    """枚举组合和逐个检查候选两条计数路径必须给出相同结果。"""

    transactions = [frozenset("ABCDE"), frozenset("ABC"), frozenset("ABD"), frozenset("E")]
    # 只有一个候选时，长交易的组合数多于候选数，走逐个检查的路径。
    assert count_candidate_support(transactions, {frozenset("AB")}, 2) == {
        frozenset("AB"): 3
    }
    # 候选很多时走枚举组合的路径。
    many = {frozenset(pair) for pair in combinations("ABCDE", 2)}
    counts = count_candidate_support(transactions, many, 2)
    for candidate in many:
        assert counts[candidate] == sum(candidate <= t for t in transactions)


def brute_force_frequent_itemsets(
    transactions: list[frozenset[str]], min_support: int
) -> dict[frozenset[str], int]:
    """枚举所有可能的 itemset 并直接计数，作为独立的参考答案。"""

    items = sorted(set().union(*transactions))
    result: dict[frozenset[str], int] = {}
    for size in range(1, len(items) + 1):
        for combination_items in combinations(items, size):
            itemset = frozenset(combination_items)
            support = sum(itemset <= transaction for transaction in transactions)
            if support >= min_support:
                result[itemset] = support
    return result


def check_against_brute_force() -> None:
    """随机小数据集上，A-Priori 和规则生成都要和暴力枚举一致。"""

    generator = random.Random(2222)
    items = [str(number) for number in range(1, 9)]
    for _ in range(200):
        transactions = [
            frozenset(generator.sample(items, generator.randint(1, 6)))
            for _ in range(generator.randint(5, 30))
        ]
        min_support = generator.randint(1, 6)
        min_confidence = generator.choice([0.0, 0.3, 0.5, 0.8, 1.0])

        expected = brute_force_frequent_itemsets(transactions, min_support)
        result = run_apriori(transactions, min_support)
        assert result["support_counts"] == expected
        by_size = result["frequent_itemsets"]
        assert all(len(itemset) == size for size, level in by_size.items() for itemset in level)
        assert sum(len(level) for level in by_size.values()) == len(expected)

        # 参考规则：对每个频繁项集枚举所有非空真子集作为前件。
        expected_rules = set()
        for itemset, support in expected.items():
            for size in range(1, len(itemset)):
                for antecedent_items in combinations(sorted(itemset), size):
                    antecedent = frozenset(antecedent_items)
                    confidence = support / expected[antecedent]
                    if confidence >= min_confidence:
                        expected_rules.add((antecedent, itemset - antecedent, support, confidence))
        rules = generate_association_rules(
            by_size, result["support_counts"], min_support, min_confidence
        )
        actual_rules = {
            (rule["antecedent"], rule["consequent"], rule["support"], rule["confidence"])
            for rule in rules
        }
        assert len(actual_rules) == len(rules), "rules must not contain duplicates"
        assert actual_rules == expected_rules


def check_input_handling() -> None:
    """读取文件和参数检查。"""

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "baskets.dat"
        # 空行被忽略，同一条交易中重复的项目只计一次。
        path.write_text("1 2 2 3\n\n2 3\n   \n10 2\n", encoding="utf-8")
        transactions = read_transactions(path)
        assert transactions == [frozenset({"1", "2", "3"}), frozenset({"2", "3"}), frozenset({"10", "2"})]
        assert run_apriori(transactions, 2)["support_counts"] == {
            frozenset({"2"}): 3,
            frozenset({"3"}): 2,
            frozenset({"2", "3"}): 2,
        }

        empty = Path(directory) / "empty.dat"
        empty.write_text("\n\n", encoding="utf-8")
        for bad_call in (
            lambda: read_transactions(empty),
            lambda: read_transactions(Path(directory) / "missing.dat"),
            lambda: run_apriori(transactions, 0),
            lambda: run_apriori(transactions, -1),
            lambda: run_apriori(transactions, 1.5),
            lambda: run_apriori([], 1),
            lambda: generate_association_rules({}, {}, 1, 1.5),
            lambda: generate_association_rules({}, {}, 1, float("nan")),
        ):
            try:
                bad_call()
            except (ValueError, FileNotFoundError):
                continue
            raise AssertionError("invalid input was accepted")

    # 数字编号按数值排序，输出更容易阅读。
    assert format_itemset(frozenset({"100", "9", "25"})) == "{9, 25, 100}"


def main() -> None:
    check_manual_example()
    check_candidate_generation()
    check_counting_paths()
    check_against_brute_force()
    check_input_handling()
    print("All small-dataset A-Priori checks passed.")


if __name__ == "__main__":
    main()
