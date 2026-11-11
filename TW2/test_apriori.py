"""Small correctness checks for the A-Priori implementation.

The checks use a tiny in-memory dataset with manually calculable results.  The
full T10I4D100K dataset is used separately by the command-line smoke test.
"""

from apriori import generate_association_rules, run_apriori


def main() -> None:
    # The expected support counts are easy to verify by hand.
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
    rule_pairs = {
        (rule["antecedent"], rule["consequent"])
        for rule in rules
    }
    assert (
        frozenset({"A"}),
        frozenset({"B"}),
    ) in rule_pairs
    assert all(0 <= rule["confidence"] <= 1 for rule in rules)

    print("All small-dataset A-Priori checks passed.")


if __name__ == "__main__":
    main()
