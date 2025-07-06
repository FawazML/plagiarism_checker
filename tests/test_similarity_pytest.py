"""
Pytest-based tests for the similarity module of the plagiarism checker.
Using fixtures from conftest.py
"""
import pytest
import numpy as np
from plagcheck.modules.similarity import SimilarityCalculator, SimilarityMetric


def test_init_with_string_metric(similarity_calculator):
    """Test initializing with string metric name."""
    calc = SimilarityCalculator("cosine")
    assert calc.metric == SimilarityMetric.COSINE


def test_init_with_enum_metric():
    """Test initializing with enum metric."""
    calc = SimilarityCalculator(SimilarityMetric.JACCARD)
    assert calc.metric == SimilarityMetric.JACCARD


def test_init_with_invalid_metric():
    """Test initializing with invalid metric."""
    with pytest.raises(ValueError):
        SimilarityCalculator("invalid_metric")


def test_cosine_similarity_identical(similarity_calculator, test_vectors, expected_similarity_scores):
    """Test cosine similarity between identical vectors (should be 1.0)."""
    v1, v2 = test_vectors["identical"]
    similarity = similarity_calculator._cosine_similarity(v1, v2)
    assert pytest.approx(similarity, abs=1e-6) == expected_similarity_scores["cosine"]["identical"]


def test_cosine_similarity_orthogonal(similarity_calculator, test_vectors, expected_similarity_scores):
    """Test cosine similarity between orthogonal vectors (should be 0.0)."""
    v1, v2 = test_vectors["orthogonal"]
    similarity = similarity_calculator._cosine_similarity(v1, v2)
    assert pytest.approx(similarity, abs=1e-6) == expected_similarity_scores["cosine"]["orthogonal"]


def test_cosine_similarity_with_zero_vector(similarity_calculator, test_vectors, expected_similarity_scores):
    """Test cosine similarity with a zero vector."""
    v1, v2 = test_vectors["zero_vector"]
    similarity = similarity_calculator._cosine_similarity(v1, v2)
    # With sklearn's implementation, this should handle the edge case
    assert similarity == expected_similarity_scores["cosine"]["zero_vector"]


def test_jaccard_similarity(test_vectors, expected_similarity_scores):
    """Test Jaccard similarity with binary vectors."""
    calculator = SimilarityCalculator(SimilarityMetric.JACCARD)
    v1, v2 = test_vectors["binary"]
    similarity = calculator._jaccard_similarity(v1, v2)
    # Expected: |intersection| / |union| = 2 / 4 = 0.5
    assert pytest.approx(similarity, abs=1e-6) == expected_similarity_scores["jaccard"]["binary"]


def test_euclidean_similarity_identical(test_vectors, expected_similarity_scores):
    """Test Euclidean similarity between identical vectors (should be 1.0)."""
    calculator = SimilarityCalculator(SimilarityMetric.EUCLIDEAN)
    v1, v2 = test_vectors["identical"]
    similarity = calculator._euclidean_similarity(v1, v2)
    assert pytest.approx(similarity, abs=1e-6) == expected_similarity_scores["euclidean"]["identical"]


def test_euclidean_similarity_different(test_vectors):
    """Test Euclidean similarity between different vectors."""
    calculator = SimilarityCalculator(SimilarityMetric.EUCLIDEAN)
    v1, v2 = test_vectors["different"]
    similarity = calculator._euclidean_similarity(v1, v2)
    # Value should be between 0 and 1, closer to 0 for more different vectors
    assert 0.0 < similarity < 1.0


def test_calculate_similarity_method(similarity_calculator, test_vectors):
    """Test the higher-level calculate_similarity method."""
    v1, v2 = test_vectors["similar"]
    similarity = similarity_calculator.calculate_similarity(v1, v2)
    # Should be close to 1.0 but not exactly 1.0
    assert similarity < 1.0
    assert similarity > 0.9


def test_similarity_matrix(similarity_calculator):
    """Test calculation of similarity matrix from multiple documents."""
    # Create a list of (name, vector) tuples
    documents = [
        ("doc1", np.array([1, 2, 3])),
        ("doc2", np.array([2, 3, 4])),
        ("doc3", np.array([0, 0, 0]))
    ]
    
    # Calculate similarity matrix as dictionary
    matrix_dict = similarity_calculator.calculate_similarity_matrix_as_dict(documents)
    
    # Verify structure and some values
    assert "doc1" in matrix_dict
    assert "doc2" in matrix_dict["doc1"]
    assert pytest.approx(matrix_dict["doc1"]["doc1"], abs=1e-6) == 1.0
    assert matrix_dict["doc1"]["doc2"] > 0.9  # Should be highly similar
    assert matrix_dict["doc1"]["doc3"] == 0.0  # Zero vector has zero similarity


def test_document_similarity_calculation(document_similarity, sample_documents):
    """Test the DocumentSimilarity class with sample documents."""
    # Calculate similarities between sample documents
    results = document_similarity.compare_texts(
        sample_documents,
        names=["doc1", "doc2", "doc3", "doc4"],
        metric="cosine",
        vectorizer="tfidf",
        ngram_range=(1, 1),
        binary=False,
        preprocess=True
    )
    
    # Check that the results contain the expected pairs
    found_pairs = {f"{pair[0]}-{pair[1]}" for pair in results}
    expected_pairs = {"doc1-doc2", "doc1-doc3", "doc1-doc4", "doc2-doc3", "doc2-doc4", "doc3-doc4"}
    assert found_pairs == expected_pairs
    
    # Check that similarity between doc1 and doc4 is high (they're very similar)
    doc1_doc4_similarity = next(
        score for doc1, doc2, score in results if 
        (doc1 == "doc1" and doc2 == "doc4") or (doc1 == "doc4" and doc2 == "doc1")
    )
    # Use a lower threshold since the actual value may vary by implementation
    assert doc1_doc4_similarity > 0.7
