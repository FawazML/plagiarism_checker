"""
Core modules for the plagiarism checker package.

This package includes components for text processing, similarity calculation,
file handling, and result presentation.
"""

from plagcheck.modules.text_processor import TextProcessor, TextNormalizer, Tokenizer, StopwordRemover
from plagcheck.modules.similarity import SimilarityCalculator, SimilarityMetric, DocumentSimilarity
from plagcheck.modules.file_handler import FileHandler
from plagcheck.modules.result_presenter import ResultPresenter, ColorTheme, OutputFormat
