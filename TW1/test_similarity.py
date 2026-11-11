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


if __name__ == "__main__":
    unittest.main()
