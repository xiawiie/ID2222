"""不依赖第三方库的最小自检。运行：python -m unittest -v"""

import unittest

from similarity import (
    CompareSets,
    CompareSignatures,
    LSH,
    MinHashing,
    Shingling,
    preprocess_text,
)


class SimilarityTests(unittest.TestCase):
    def test_shingling_removes_duplicate_shingles(self):
        shingling = Shingling(k=3)
        result = shingling.shingles("AAAA")
        self.assertEqual(len(result), 1)

    def test_jaccard(self):
        self.assertAlmostEqual(CompareSets.jaccard({1, 2}, {2, 3}), 1 / 3)

    def test_minhash_signature_length(self):
        signature = MinHashing(signature_length=20).signature({1, 2, 3})
        self.assertEqual(len(signature), 20)

    def test_identical_signatures_have_similarity_one(self):
        signature = [1, 2, 3]
        self.assertEqual(CompareSignatures.similarity(signature, signature), 1.0)

    def test_lsh_finds_identical_signatures(self):
        signatures = {"a": [1] * 10, "b": [1] * 10, "c": [2] * 10}
        candidates = LSH.candidates(signatures, bands=2, rows=5)
        self.assertIn(("a", "b"), candidates)
        self.assertNotIn(("a", "c"), candidates)

    def test_preprocessing_keeps_subject_but_removes_other_headers(self):
        raw = "Subject: A test\nMessage-ID: <unique>\n\nThis is the body."
        cleaned = preprocess_text(raw)
        self.assertIn("a test", cleaned)
        self.assertIn("this is the body", cleaned)
        self.assertNotIn("unique", cleaned)

    def test_lsh_rejects_unused_signature_components(self):
        signatures = {"a": [1] * 10, "b": [1] * 10}
        with self.assertRaises(ValueError):
            LSH.candidates(signatures, bands=2, rows=4)

    def test_lsh_rejects_different_signature_lengths(self):
        signatures = {"a": [1] * 10, "b": [1] * 9}
        with self.assertRaises(ValueError):
            LSH.candidates(signatures, bands=2, rows=5)

    def test_shingles_are_an_ordered_set(self):
        result = Shingling(k=2).shingles("abcabc")
        # "ab"、"bc"、"ca" 三个不同的 shingle，升序且没有重复。
        self.assertEqual(len(result), 3)
        self.assertEqual(result, sorted(set(result)))

    def test_text_shorter_than_k_has_no_shingles(self):
        self.assertEqual(Shingling(k=5).shingles("abc"), [])

    def test_jaccard_accepts_lists_and_sets(self):
        self.assertAlmostEqual(CompareSets.jaccard([1, 2, 2], frozenset({2, 3})), 1 / 3)
        self.assertEqual(CompareSets.jaccard({1, 2}, {3, 4}), 0.0)
        self.assertEqual(CompareSets.jaccard(set(), set()), 1.0)

    def test_minhash_is_deterministic_and_order_independent(self):
        first = MinHashing(signature_length=30, seed=7).signature([5, 9, 12])
        second = MinHashing(signature_length=30, seed=7).signature([12, 5, 9])
        self.assertEqual(first, second)

    def test_minhash_estimate_is_close_to_exact_jaccard(self):
        # 两个集合共享 500 个元素，并集为 1500，Jaccard 等于 1/3。
        first = set(range(0, 1000))
        second = set(range(500, 1500))
        minhashing = MinHashing(signature_length=500, seed=1)
        estimate = CompareSignatures.similarity(
            minhashing.signature(first), minhashing.signature(second)
        )
        # 500 个哈希函数时标准差约为 0.021，0.1 的容差非常宽松。
        self.assertAlmostEqual(estimate, 1 / 3, delta=0.1)

    def test_empty_document_signature_never_matches_non_empty_one(self):
        minhashing = MinHashing(signature_length=10)
        empty = minhashing.signature([])
        self.assertEqual(CompareSignatures.similarity(empty, empty), 1.0)
        self.assertEqual(
            CompareSignatures.similarity(empty, minhashing.signature([1, 2])), 0.0
        )

    def test_signature_similarity_is_fraction_of_equal_components(self):
        self.assertEqual(CompareSignatures.similarity([1, 2, 3, 4], [1, 2, 0, 0]), 0.5)
        with self.assertRaises(ValueError):
            CompareSignatures.similarity([1, 2], [1])

    def test_choose_bands_follows_the_threshold(self):
        # 阈值越高，每个 band 可以用越多行，候选对越少。
        self.assertEqual(LSH.choose_bands(100, 0.8), (20, 5))
        self.assertEqual(LSH.choose_bands(100, 0.3), (50, 2))
        for threshold in (0.1, 0.3, 0.5, 0.8):
            bands, rows = LSH.choose_bands(100, threshold)
            self.assertEqual(bands * rows, 100)
            self.assertGreaterEqual(
                LSH.detection_probability(threshold, bands, rows), 0.95
            )

    def test_lsh_similar_pairs_applies_the_threshold(self):
        # a 和 b 有 8/10 的分量相同，并且第一个 band 完全相同。
        # a 和 c 第二个 band 相同，所以是候选对，但只有 5/10 的分量相同。
        signatures = {
            "a": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "b": [1, 1, 1, 1, 1, 2, 2, 2, 9, 9],
            "c": [7, 7, 7, 7, 7, 2, 2, 2, 2, 2],
        }
        self.assertIn(("a", "c"), LSH.candidates(signatures, bands=2, rows=5))
        pairs = LSH.similar_pairs(signatures, threshold=0.8, bands=2, rows=5)
        self.assertEqual(pairs, {("a", "b"): 0.8})

    def test_lsh_similar_pairs_chooses_bands_automatically(self):
        signatures = {"a": [1] * 100, "b": [1] * 100, "c": [2] * 100}
        self.assertEqual(LSH.similar_pairs(signatures, threshold=0.8), {("a", "b"): 1.0})
        with self.assertRaises(ValueError):
            LSH.similar_pairs(signatures, threshold=0.8, bands=20)


if __name__ == "__main__":
    unittest.main()
