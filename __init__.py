"""
Plagiarism Checker - A tool for detecting similarities between text documents.

This package provides tools for text processing, similarity detection,
file handling, and result presentation for plagiarism detection.
"""

__version__ = "0.1.0"

# Import key components to make them available at package level
from plagcheck.modules.similarity import DocumentSimilarity, SimilarityMetric, SimilarityCalculator
from plagcheck.modules.text_processor import TextProcessor, TextNormalizer, Tokenizer, StopwordRemover
from plagcheck.modules.file_handler import FileHandler
from plagcheck.modules.result_presenter import ResultPresenter, ColorTheme, OutputFormat

# Database components
from plagcheck.db.models import SimilarityResult
from plagcheck.db.db import init_db, get_db, shutdown_db
