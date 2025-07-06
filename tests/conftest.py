"""
Pytest fixtures for the plagiarism checker tests.
These fixtures provide reusable sample documents and expected outputs for tests.
"""

import pytest
import numpy as np
import os
from plagcheck.modules.text_processor import TextProcessor, TextNormalizer, Tokenizer, StopwordRemover
from plagcheck.modules.similarity import SimilarityCalculator, SimilarityMetric, DocumentSimilarity
from pathlib import Path


# Basic document fixtures
@pytest.fixture
def sample_documents():
    """Return a list of simple sample documents for testing."""
    return [
        "This is the first document. It contains simple text.",
        "This is the second document and it is similar to the first one.",
        "This document is completely different from the others.",
        "This is almost identical to the first document. It contains simple text."
    ]


@pytest.fixture
def document_with_features():
    """Return a document with various features for testing preprocessing."""
    return """
    This is a Sample Document with:
    - Uppercase letters
    - Punctuation marks!
    - Numbers like 123 and 456
    - Extra   spaces   and line breaks
    
    It also contains common stopwords like 'a', 'the', 'is', 'and'.
    """


@pytest.fixture
def documents_with_known_similarity():
    """Return a list of documents with known similarity relationships."""
    return {
        "doc1": "Python is a programming language that is widely used in data science.",
        "doc2": "Python programming language is widely adopted for data science applications.",
        "doc3": "Java is another programming language that is used for enterprise applications.",
        "doc4": "Data science involves analyzing and interpreting complex data using statistics.",
        "doc5": "Python is a programming language that is widely used in data science."  # Same as doc1
    }


# Text preprocessing fixtures
@pytest.fixture
def text_normalizer():
    """Return a set of common normalization operations."""
    return [
        TextNormalizer.to_lowercase,
        TextNormalizer.remove_punctuation,
        TextNormalizer.remove_whitespace
    ]


@pytest.fixture
def stopword_remover():
    """Return a stopword remover with default stopwords."""
    return StopwordRemover()


@pytest.fixture
def custom_stopword_remover():
    """Return a stopword remover with custom stopwords."""
    custom_stopwords = {"document", "text", "contains"}
    return StopwordRemover(custom_stopwords, include_default=False)


@pytest.fixture
def text_processor():
    """Return a text processor with default settings."""
    return TextProcessor()


@pytest.fixture
def custom_text_processor(text_normalizer, custom_stopword_remover):
    """Return a text processor with custom settings."""
    return TextProcessor(
        normalize_operations=text_normalizer,
        tokenizer=Tokenizer.tokenize_words,
        stopword_remover=custom_stopword_remover
    )


# Similarity calculation fixtures
@pytest.fixture
def similarity_calculator():
    """Return a similarity calculator with default settings (cosine)."""
    return SimilarityCalculator()


@pytest.fixture
def similarity_calculators():
    """Return similarity calculators for all metrics."""
    metrics = [metric for metric in SimilarityMetric]
    return {metric.value: SimilarityCalculator(metric) for metric in metrics}


@pytest.fixture
def document_similarity():
    """Return a document similarity calculator."""
    return DocumentSimilarity()


@pytest.fixture
def test_vectors():
    """Return a dictionary of test vectors for similarity comparison."""
    return {
        "identical": (
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5])
        ),
        "similar": (
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 4])
        ),
        "different": (
            np.array([1, 2, 3, 4, 5]),
            np.array([5, 4, 3, 2, 1])
        ),
        "orthogonal": (
            np.array([1, 0, 0]),
            np.array([0, 1, 0])
        ),
        "zero_vector": (
            np.array([1, 2, 3, 4, 5]),
            np.zeros(5)
        ),
        "binary": (
            np.array([1, 1, 0, 0, 1]),
            np.array([1, 0, 0, 1, 1])
        )
    }


# Expected results fixtures
@pytest.fixture
def expected_normalizations():
    """Return expected results for normalization operations."""
    return {
        "lowercase": {
            "input": "TEXT WITH UPPERCASE and Mixed Case",
            "output": "text with uppercase and mixed case"
        },
        "punctuation": {
            "input": "Hello, world! This: is a test; with punctuation.",
            "output": "Hello world This is a test with punctuation"
        },
        "whitespace": {
            "input": "  Text   with  extra    spaces   ",
            "output": "Text with extra spaces"
        },
        "numbers": {
            "input": "There are 42 apples and 17 oranges",
            "output": "There are  apples and  oranges"
        },
        "unicode": {
            "input": "Café Résumé",
            "output": "Cafe Resume"
        }
    }


@pytest.fixture
def expected_tokenizations():
    """Return expected results for tokenization operations."""
    return {
        "words": {
            "input": "This is a simple test",
            "output": ["This", "is", "a", "simple", "test"]
        },
        "sentences": {
            "input": "First sentence. Second sentence! Third sentence?",
            "output": ["First sentence.", "Second sentence!", "Third sentence?"]
        },
        "char_bigrams": {
            "input": "abcd",
            "output": ["ab", "bc", "cd"]
        },
        "char_trigrams": {
            "input": "abcde",
            "output": ["abc", "bcd", "cde"]
        },
        "word_bigrams": {
            "input": "This is a simple test",
            "output": ["This is", "is a", "a simple", "simple test"]
        },
        "empty": {
            "input": "",
            "output_words": [],
            "output_sentences": [""]
        }
    }


@pytest.fixture
def expected_similarity_scores():
    """Return expected similarity scores for different metrics and vector pairs."""
    return {
        "cosine": {
            "identical": 1.0,
            "orthogonal": 0.0,
            "zero_vector": 0.0
        },
        "jaccard": {
            "binary": 0.5  # Intersection: 2, Union: 4
        },
        "euclidean": {
            "identical": 1.0
        }
    }


# File-based fixtures
@pytest.fixture
def test_files_dir():
    """Return the path to the test files directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "files")


@pytest.fixture
def sample_file_contents(test_files_dir):
    """Return the contents of the sample files."""
    result = {}
    for filename in os.listdir(test_files_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(test_files_dir, filename)
            with open(filepath, 'r') as file:
                result[filename] = file.read()
    return result


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary directory with test files."""
    # Create test documents
    doc1 = tmp_path / "doc1.txt"
    doc2 = tmp_path / "doc2.txt"
    doc3 = tmp_path / "doc3.txt"
    
    doc1.write_text("This is document one with some unique content.")
    doc2.write_text("This is document two with similar structure but different words.")
    doc3.write_text("Document three has completely different text.")
    
    return tmp_path


# Command line testing fixtures
@pytest.fixture
def cli_parser_args():
    """Return common sets of command line arguments for testing."""
    return {
        "basic": ["--dir", "files", "--ext", ".txt"],
        "advanced": [
            "--dir", "files",
            "--ext", ".txt",
            "--normalize",
            "--remove-stopwords",
            "--similarity", "jaccard",
            "--ngram", "3",
            "--visualization", "heatmap",
            "--output", "results.csv",
            "--output-format", "csv",
            "--color-theme", "traffic_light",
            "--precision", "2"
        ],
        "invalid": ["--similarity", "invalid_metric"],
        "help": ["--help"]
    }


@pytest.fixture
def expected_cli_results():
    """Return expected results for command line argument parsing."""
    return {
        "basic": {
            "dir": "files",
            "ext": ".txt",
            "normalize": False,
            "remove_stopwords": False,
            "similarity": "cosine",
            "ngram": 1,
            "visualization": "none"
        },
        "advanced": {
            "dir": "files",
            "ext": ".txt",
            "normalize": True,
            "remove_stopwords": True,
            "similarity": "jaccard",
            "ngram": 3,
            "visualization": "heatmap",
            "output": "results.csv",
            "output_format": "csv",
            "color_theme": "traffic_light",
            "precision": 2
        }
    }
