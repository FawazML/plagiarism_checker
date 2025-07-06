"""
Tests for error handling in the plagiarism checker.
These tests verify that the application handles error conditions gracefully.
"""

import os
import sys
import pytest
import tempfile
import subprocess
from pathlib import Path
import io
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch, mock_open

from plagcheck.modules.file_handler import FileHandler
from plagcheck.modules.text_processor import TextProcessor
from plagcheck.modules.similarity import SimilarityCalculator, DocumentSimilarity
from plagcheck.modules.result_presenter import ResultPresenter

# Import main function for direct calling in tests
try:
    from plagcheck import cli
except ImportError:
    # If main is not directly importable, use a mock
    main = None


class TestFileHandlingErrors:
    """Test error handling related to file operations."""

    def test_nonexistent_directory(self):
        """Test handling of a nonexistent directory."""
        nonexistent_dir = "/path/that/definitely/does/not/exist/12345"
        file_handler = FileHandler(directory=nonexistent_dir)
        
        # Should return an empty dictionary instead of raising an exception
        result = file_handler.load_files()
        assert result == {}, "Should return an empty dictionary for nonexistent directory"


    def test_empty_directory(self, tmp_path):
        """Test handling of an empty directory."""
        # Create an empty directory
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        
        file_handler = FileHandler(directory=str(empty_dir))
        result = file_handler.load_files()
        
        # Should return an empty dictionary
        assert result == {}, "Should return an empty dictionary for empty directory"


    def test_permission_denied(self, tmp_path, monkeypatch):
        """Test handling of permission denied when reading files."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        # Mock open to raise a permission error
        def mock_open_with_perm_error(*args, **kwargs):
            raise PermissionError("Permission denied")
        
        # Patch the built-in open function when called by FileHandler
        with patch('builtins.open', mock_open_with_perm_error):
            file_handler = FileHandler(directory=str(tmp_path))
            
            # Capture stdout to check for error message
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = file_handler.load_files()
            
            # Should print an error message and return an empty dictionary
            assert "Error loading file" in stdout.getvalue()
            assert "Permission denied" in stdout.getvalue()
            assert result == {}, "Should return an empty dictionary when permission is denied"


    def test_invalid_encoding(self, tmp_path):
        """Test handling of files with invalid encoding."""
        # Create a binary file with non-UTF-8 content
        binary_file = tmp_path / "binary.txt"
        with open(binary_file, 'wb') as f:
            f.write(b'\x80\x81\x82')  # Invalid UTF-8 bytes
        
        file_handler = FileHandler(directory=str(tmp_path))
        
        # Capture stdout to check for error message
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = file_handler.load_files()
        
        # Should print an error message but continue processing
        assert "Error loading file" in stdout.getvalue()
        assert result == {}, "Should return an empty dictionary for files with invalid encoding"


class TestCLIErrorHandling:
    """Test error handling in the command-line interface."""

    def test_invalid_similarity_metric(self):
        """Test error handling for invalid similarity metric."""
        cmd = [
            sys.executable,
            "main.py",
            "--similarity", "invalid_metric"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Should exit with non-zero status and print error message
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()
        assert "invalid_metric" in result.stderr


    def test_invalid_visualization_type(self):
        """Test error handling for invalid visualization type."""
        cmd = [
            sys.executable,
            "main.py",
            "--visualization", "invalid_type"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Should exit with non-zero status and print error message
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()
        assert "invalid_type" in result.stderr


    def test_invalid_output_format(self):
        """Test error handling for invalid output format."""
        cmd = [
            sys.executable,
            "main.py",
            "--output-format", "invalid_format"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Should exit with non-zero status and print error message
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()
        assert "invalid_format" in result.stderr


    def test_missing_output_file_with_format(self, tmp_path):
        """Test error handling when output format is specified but output file is not."""
        cmd = [
            sys.executable,
            "main.py",
            "--dir", str(tmp_path),
            "--output-format", "json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Either should warn about missing output file or ignore the format
        if result.returncode != 0:
            assert "requires" in result.stderr.lower() or "requires" in result.stdout.lower()
        else:
            # If it runs successfully, it should not crash
            pass


    def test_missing_required_args_for_metrics(self):
        """Test handling of missing required arguments for Levenshtein similarity."""
        calculator = SimilarityCalculator(metric="levenshtein")
        
        # Levenshtein needs raw text, so this should raise a ValueError
        with pytest.raises(ValueError) as excinfo:
            calculator.calculate_similarity(
                vector1=None, 
                vector2=None
            )
        
        # Check the error message
        assert "Raw texts not available" in str(excinfo.value)


@pytest.mark.skipif(main is None, reason="main function not importable")
class TestMainErrorHandling:
    """Test error handling in the main function."""

    def test_no_files_found_message(self, monkeypatch, tmp_path):
        """Test that appropriate message is shown when no files are found."""
        # Create an empty directory
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        
        # Prepare command-line args
        test_args = [
            "main.py",
            "--dir", str(empty_dir)
        ]
        
        # Patch sys.argv
        monkeypatch.setattr('sys.argv', test_args)
        
        # Capture stdout
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            # Call main
            try:
                main()
            except SystemExit:
                pass
        
        # Check for the "No files found" message
        assert "No files found" in stdout.getvalue()


    def test_empty_file_handling(self, tmp_path):
        """Test handling of empty files."""
        # Create an empty file
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        
        # Create a non-empty file
        nonempty_file = tmp_path / "nonempty.txt"
        nonempty_file.write_text("This file has content")
        
        # Load files
        file_handler = FileHandler(directory=str(tmp_path))
        file_contents = file_handler.load_files()
        
        # Should load both files without errors
        assert "empty.txt" in file_contents
        assert "nonempty.txt" in file_contents
        assert file_contents["empty.txt"] == ""
        
        # Process with document similarity
        doc_similarity = DocumentSimilarity()
        results = doc_similarity.compare_texts(
            list(file_contents.values()),
            names=list(file_contents.keys()),
            preprocess=True
        )
        
        # Should not crash and return results
        assert isinstance(results, set)
        # Results might be empty or have a 0 similarity score for empty file comparison


class TestSimilarityErrorHandling:
    """Test error handling in the similarity module."""

    def test_invalid_similarity_metric_value(self):
        """Test handling of an invalid similarity metric value."""
        with pytest.raises(ValueError) as excinfo:
            SimilarityCalculator("not_a_real_metric")
        
        assert "Unsupported similarity metric" in str(excinfo.value)


    def test_invalid_vectorizer_value(self):
        """Test handling of an invalid vectorizer value."""
        doc_similarity = DocumentSimilarity()
        
        with pytest.raises(ValueError) as excinfo:
            doc_similarity.compare_texts(
                ["Test document"],
                names=["doc1"],
                vectorizer="not_a_real_vectorizer"
            )
        
        assert "Unsupported vectorizer" in str(excinfo.value)


    def test_unimplemented_similarity_metric(self, monkeypatch):
        """Test handling of a similarity metric that is defined but not implemented."""
        # Create a calculator with a mocked metric that isn't implemented
        calculator = SimilarityCalculator("cosine")
        
        # Change the metric to something not in similarity_functions
        monkeypatch.setattr(calculator, 'metric', "fake_metric")
        
        # Try to calculate similarity
        with pytest.raises(ValueError) as excinfo:
            calculator.calculate_similarity(
                vector1=None,
                vector2=None
            )
        
        assert "Similarity metric not implemented" in str(excinfo.value)


class TestResultPresenterErrorHandling:
    """Test error handling in the result presenter module."""

    def test_unknown_output_format(self):
        """Test handling of unknown output format."""
        presenter = ResultPresenter()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.xyz') as temp:
            # Try to save with an unknown format
            results = {('doc1', 'doc2', 0.5)}
            
            # Capture stdout
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                # Should not crash
                presenter.save_results(results, temp.name)
            
            # It should handle the unknown format gracefully (but might not print a warning)
            assert temp.name in stdout.getvalue(), "Should indicate which file was saved"
            

    def test_invalid_visualization_type(self, tmp_path):
        """Test handling of invalid visualization type."""
        presenter = ResultPresenter()
        results = {('doc1', 'doc2', 0.5)}
        output_file = str(tmp_path / "invalid_viz.png")
        
        # Capture stdout for warning message
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            # Should not crash with invalid visualization type
            presenter.visualize(
                results,
                visualization_type="not_a_real_viz",
                output_file=output_file
            )
        
        # It should print a warning about the invalid visualization type
        assert "Unsupported visualization type" in stdout.getvalue() or "Invalid visualization type" in stdout.getvalue()
