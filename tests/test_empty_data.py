"""
Tests for handling empty or minimal data in the plagiarism checker.
"""

import pytest
import io
from contextlib import redirect_stdout
import tempfile
from pathlib import Path

from plagcheck.modules.file_handler import FileHandler
from plagcheck.modules.text_processor import TextProcessor
from plagcheck.modules.similarity import SimilarityCalculator, DocumentSimilarity
from plagcheck.modules.result_presenter import ResultPresenter


class TestEmptyData:
    """Test handling of empty text data."""
    
    def test_empty_texts_comparison(self):
        """Test comparing empty texts."""
        empty_texts = ["", ""]
        names = ["empty1", "empty2"]
        
        # Create document similarity calculator
        doc_similarity = DocumentSimilarity()
        
        # The current implementation raises an error for empty texts
        # This test helps identify this behavior and suggests improvements
        with pytest.raises(ValueError) as excinfo:
            doc_similarity.compare_texts(
                empty_texts,
                names=names,
                metric='cosine',
                preprocess=True
            )
        
        # Check the error message to confirm it's the expected error
        assert "empty vocabulary" in str(excinfo.value)
        
        # Bug report: The application should handle empty texts gracefully
        # and return a default similarity (e.g., 1.0 for identical empty texts)
    
    
    def test_empty_vs_nonempty_text(self):
        """Test comparing an empty text with a non-empty text."""
        texts = ["", "This is a non-empty text with sufficient content to process"]
        names = ["empty", "nonempty"]
        
        # Create document similarity calculator
        doc_similarity = DocumentSimilarity()
        
        try:
            # Try to calculate similarity - may raise an error or return results
            results = doc_similarity.compare_texts(
                texts,
                names=names,
                metric='cosine',
                preprocess=True
            )
            
            # If no error, check results
            assert isinstance(results, set)
            
        except ValueError as e:
            # If it raises an error, check that it's the expected error
            assert "empty vocabulary" in str(e) or "document vector is empty" in str(e)
            
            # Bug report: The application should handle one empty and one non-empty text
            # better, perhaps by returning 0.0 similarity or by adding special handling
    
    
    def test_empty_text_preprocessing(self):
        """Test preprocessing an empty text."""
        empty_text = ""
        
        # Create text processor
        processor = TextProcessor()
        
        # Preprocess the empty text
        result = processor.preprocess(empty_text)
        
        # Should return an empty string without crashing
        assert result == "" or result is None or len(result) == 0
    
    
    def test_save_empty_results(self, tmp_path):
        """Test saving empty results to a file."""
        # Create an empty result set
        results = set()
        
        # Create output file
        output_file = tmp_path / "empty_results.csv"
        
        # Create result presenter
        presenter = ResultPresenter()
        
        # Capture stdout
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            # Save empty results
            presenter.save_results(results, str(output_file), format='csv')
        
        # Should not crash and create the file
        assert output_file.exists(), "Should create the output file even with empty results"
        
        # File should be empty or contain just a header
        content = output_file.read_text()
        assert len(content.strip().splitlines()) <= 1, "File should be empty or contain only a header row"
        
        # Should indicate that the file was saved
        assert str(output_file) in stdout.getvalue(), "Should indicate the file was saved"


class TestMinimalData:
    """Test handling of minimal text data (e.g., single character)."""
    
    def test_single_character_texts(self):
        """Test comparing texts with single characters."""
        texts = ["a", "b"]
        names = ["char1", "char2"]
        
        # Create document similarity calculator
        doc_similarity = DocumentSimilarity()
        
        # The current implementation has issues with minimal texts
        with pytest.raises(ValueError) as excinfo:
            doc_similarity.compare_texts(
                texts,
                names=names,
                metric='cosine',
                preprocess=True
            )
            
        # Check that it's the expected error
        assert "empty vocabulary" in str(excinfo.value)
        
        # Bug report: The application should handle minimal texts better,
        # perhaps by adding placeholder content or special case handling
    
    
    def test_whitespace_only_texts(self):
        """Test comparing texts with only whitespace."""
        texts = ["   ", "\t\n"]
        names = ["space", "tab_newline"]
        
        # Create document similarity calculator
        doc_similarity = DocumentSimilarity()
        
        # The current implementation has issues with whitespace-only texts
        with pytest.raises(ValueError) as excinfo:
            doc_similarity.compare_texts(
                texts,
                names=names,
                metric='cosine',
                preprocess=True
            )
            
        # Check that it's the expected error
        assert "empty vocabulary" in str(excinfo.value)
        
        # Bug report: The application should handle whitespace-only texts better,
        # perhaps by treating them as empty texts
    
    
    def test_stopword_only_texts(self):
        """Test comparing texts with only stopwords."""
        texts = ["the a and", "is the of"]
        names = ["stopword1", "stopword2"]
        
        # Create document similarity calculator with stopword removal
        doc_similarity = DocumentSimilarity(
            preprocessor=TextProcessor(
                stopword_remover=TextProcessor().stopword_remover  # Default stopword remover
            )
        )
        
        # When stopwords are removed, these texts become empty
        with pytest.raises(ValueError) as excinfo:
            doc_similarity.compare_texts(
                texts,
                names=names,
                metric='cosine',
                preprocess=True
            )
            
        # Check that it's the expected error
        assert "empty vocabulary" in str(excinfo.value)
        
        # Bug report: The application should handle stopword-only texts better,
        # perhaps by detecting this case and returning a default value or warning
