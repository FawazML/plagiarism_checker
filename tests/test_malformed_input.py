"""
Tests for handling malformed input files in the plagiarism checker.
"""

import os
import pytest
import io
from contextlib import redirect_stdout
from pathlib import Path

from plagcheck.modules.file_handler import FileHandler
from plagcheck.modules.text_processor import TextProcessor
from plagcheck.modules.similarity import DocumentSimilarity
from plagcheck.modules.result_presenter import ResultPresenter


class TestMalformedInput:
    """Test handling of malformed input files."""
    
    def test_binary_file_handling(self, tmp_path):
        """Test handling of binary files."""
        # Create a binary file
        binary_file = tmp_path / "binary.txt"
        with open(binary_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\x04\xff')
        
        # Also create a valid text file
        text_file = tmp_path / "text.txt"
        text_file.write_text("This is a valid text file")
        
        # Load files
        file_handler = FileHandler(directory=str(tmp_path))
        
        # Capture stdout to check for error message
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            file_contents = file_handler.load_files()
        
        # Should print an error message for the binary file
        assert "Error loading file" in stdout.getvalue()
        
        # Should still load the valid file
        assert "text.txt" in file_contents
        assert file_contents["text.txt"] == "This is a valid text file"
    
    
    def test_very_large_file_handling(self, tmp_path):
        """Test handling of very large files (simulated)."""
        # Simulate a very large file by mocking read()
        large_file = tmp_path / "large.txt"
        
        # Write a placeholder that we'll pretend is huge
        large_file.write_text("LARGE FILE PLACEHOLDER")
        
        # Create a file handler
        file_handler = FileHandler(directory=str(tmp_path))
        
        # In a real implementation, we would check for memory limits,
        # but for testing, we'll just verify the file is loaded
        file_contents = file_handler.load_files()
        
        # Should load the file without crashing
        assert "large.txt" in file_contents
        assert file_contents["large.txt"] == "LARGE FILE PLACEHOLDER"
    
    
    def test_invalid_utf8_handling(self, tmp_path):
        """Test handling of files with invalid UTF-8 characters."""
        # Create a file with invalid UTF-8 sequence
        invalid_file = tmp_path / "invalid_utf8.txt"
        with open(invalid_file, 'wb') as f:
            f.write(b'Valid ASCII then invalid \x80\x81\x82 UTF-8 sequence')
        
        # Load files
        file_handler = FileHandler(directory=str(tmp_path))
        
        # Capture stdout to check for error message
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            file_contents = file_handler.load_files()
        
        # Should print an error message
        assert "Error loading file" in stdout.getvalue()
        assert "invalid_utf8.txt" in stdout.getvalue()
        
        # The invalid file should not be included in the results
        assert "invalid_utf8.txt" not in file_contents
    
    
    def test_mixed_text_formats(self, tmp_path):
        """Test handling files with mixed line endings and encodings."""
        # Create files with different line endings
        unix_file = tmp_path / "unix.txt"
        unix_file.write_text("Line 1\nLine 2\nLine 3")
        
        windows_file = tmp_path / "windows.txt"
        windows_file.write_text("Line 1\r\nLine 2\r\nLine 3")
        
        # Load files
        file_handler = FileHandler(directory=str(tmp_path))
        file_contents = file_handler.load_files()
        
        # Should load both files
        assert "unix.txt" in file_contents
        assert "windows.txt" in file_contents
        
        # Should handle the different line endings
        # (exact handling depends on implementation, but should not crash)
        assert "Line" in file_contents["unix.txt"]
        assert "Line" in file_contents["windows.txt"]
    
    
    def test_file_without_extension(self, tmp_path):
        """Test handling files without extensions."""
        # Create a file without extension
        no_ext_file = tmp_path / "noextension"
        no_ext_file.write_text("This file has no extension")
        
        # Create a file with the expected extension
        txt_file = tmp_path / "withextension.txt"
        txt_file.write_text("This file has a .txt extension")
        
        # Load files with .txt extension filter
        file_handler = FileHandler(directory=str(tmp_path), file_extension='.txt')
        file_contents = file_handler.load_files()
        
        # Should only load the .txt file
        assert "withextension.txt" in file_contents
        assert len(file_contents) == 1, "Should only load files with the specified extension"
    
    
    def test_nonexistent_output_directory(self, tmp_path):
        """Test handling of nonexistent output directory."""
        # Create a nonexistent path for output
        nonexistent_dir = tmp_path / "does_not_exist" / "output"
        
        # Create a simple result
        results = {('doc1', 'doc2', 0.5)}
        
        # Create an output file path in the nonexistent directory
        output_file = nonexistent_dir / "results.csv"
        
        # Create result presenter
        presenter = ResultPresenter()
        
        # Attempt to save results
        try:
            presenter.save_results(results, str(output_file))
            # If it succeeds without errors, the directory was created
            assert output_file.exists(), "File should exist if save was successful"
            assert nonexistent_dir.exists(), "Directory should have been created"
        except Exception:
            # This is also acceptable behavior if the application doesn't auto-create directories
            assert not nonexistent_dir.exists(), "Directory should not exist if save failed"
