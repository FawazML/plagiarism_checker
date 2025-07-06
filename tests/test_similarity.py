"""
Unit tests for the similarity module of the plagiarism checker.
"""
import unittest
import numpy as np
from plagcheck.modules.similarity import SimilarityCalculator, SimilarityMetric


class TestSimilarityCalculator(unittest.TestCase):
    """Test cases for the SimilarityCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a default similarity calculator with cosine metric
        self.calculator = SimilarityCalculator(SimilarityMetric.COSINE)
        
        # Test vectors for different scenarios
        # Identical vectors
        self.identical_vectors = (
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5])
        )
        
        # Similar vectors
        self.similar_vectors = (
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 4])
        )
        
        # Different vectors
        self.different_vectors = (
            np.array([1, 2, 3, 4, 5]),
            np.array([5, 4, 3, 2, 1])
        )
        
        # Orthogonal vectors (should have 0 cosine similarity)
        self.orthogonal_vectors = (
            np.array([1, 0, 0]),
            np.array([0, 1, 0])
        )
        
        # Zero vector (edge case)
        self.zero_vector = np.zeros(5)
        
        # Binary vectors for Jaccard
        self.binary_vectors = (
            np.array([1, 1, 0, 0, 1]),
            np.array([1, 0, 0, 1, 1])
        )

    def test_init_with_string_metric(self):
        """Test initializing with string metric name."""
        calc = SimilarityCalculator("cosine")
        self.assertEqual(calc.metric, SimilarityMetric.COSINE)

    def test_init_with_enum_metric(self):
        """Test initializing with enum metric."""
        calc = SimilarityCalculator(SimilarityMetric.JACCARD)
        self.assertEqual(calc.metric, SimilarityMetric.JACCARD)

    def test_init_with_invalid_metric(self):
        """Test initializing with invalid metric."""
        with self.assertRaises(ValueError):
            SimilarityCalculator("invalid_metric")

    def test_cosine_similarity_identical(self):
        """Test cosine similarity between identical vectors (should be 1.0)."""
        v1, v2 = self.identical_vectors
        similarity = self.calculator._cosine_similarity(v1, v2)
        self.assertAlmostEqual(similarity, 1.0, places=6)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity between orthogonal vectors (should be 0.0)."""
        v1, v2 = self.orthogonal_vectors
        similarity = self.calculator._cosine_similarity(v1, v2)
        self.assertAlmostEqual(similarity, 0.0, places=6)

    def test_cosine_similarity_with_zero_vector(self):
        """Test cosine similarity with a zero vector."""
        v1 = self.identical_vectors[0]
        similarity = self.calculator._cosine_similarity(v1, self.zero_vector)
        # With sklearn's implementation, this should handle the edge case
        self.assertEqual(similarity, 0.0)

    def test_jaccard_similarity(self):
        """Test Jaccard similarity with binary vectors."""
        calculator = SimilarityCalculator(SimilarityMetric.JACCARD)
        v1, v2 = self.binary_vectors
        similarity = calculator._jaccard_similarity(v1, v2)
        # Expected: |intersection| / |union| = 2 / 4 = 0.5
        self.assertAlmostEqual(similarity, 0.5, places=6)

    def test_euclidean_similarity_identical(self):
        """Test Euclidean similarity between identical vectors (should be 1.0)."""
        calculator = SimilarityCalculator(SimilarityMetric.EUCLIDEAN)
        v1, v2 = self.identical_vectors
        similarity = calculator._euclidean_similarity(v1, v2)
        self.assertAlmostEqual(similarity, 1.0, places=6)

    def test_euclidean_similarity_different(self):
        """Test Euclidean similarity between different vectors."""
        calculator = SimilarityCalculator(SimilarityMetric.EUCLIDEAN)
        v1, v2 = self.different_vectors
        similarity = calculator._euclidean_similarity(v1, v2)
        # Value should be between 0 and 1, closer to 0 for more different vectors
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)

    def test_calculate_similarity_method(self):
        """Test the higher-level calculate_similarity method."""
        v1, v2 = self.similar_vectors
        similarity = self.calculator.calculate_similarity(v1, v2)
        # Should be close to 1.0 but not exactly 1.0
        self.assertLess(similarity, 1.0)
        self.assertGreater(similarity, 0.9)

    def test_similarity_matrix(self):
        """Test calculation of similarity matrix from multiple documents."""
        # Create a list of (name, vector) tuples
        documents = [
            ("doc1", np.array([1, 2, 3])),
            ("doc2", np.array([2, 3, 4])),
            ("doc3", np.array([0, 0, 0]))
        ]
        
        # Calculate similarity matrix as dictionary
        matrix_dict = self.calculator.calculate_similarity_matrix_as_dict(documents)
        
        # Verify structure and some values
        self.assertIn("doc1", matrix_dict)
        self.assertIn("doc2", matrix_dict["doc1"])
        self.assertAlmostEqual(matrix_dict["doc1"]["doc1"], 1.0, places=6)
        self.assertGreater(matrix_dict["doc1"]["doc2"], 0.9)  # Should be highly similar
        self.assertEqual(matrix_dict["doc1"]["doc3"], 0.0)    # Zero vector has zero similarity


if __name__ == '__main__':
    unittest.main()
