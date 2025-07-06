"""
Integration tests for the plagiarism checker.
These tests verify end-to-end flows from file input through processing to output.
"""

import os
import pytest
import tempfile
import json
import csv
import io
import subprocess
import sys
from pathlib import Path
import numpy as np
from contextlib import redirect_stdout

from plagcheck.modules.file_handler import FileHandler
from plagcheck.modules.text_processor import TextProcessor, TextNormalizer, Tokenizer, StopwordRemover
from plagcheck.modules.similarity import SimilarityCalculator, SimilarityMetric, DocumentSimilarity
from plagcheck.modules.result_presenter import ResultPresenter, ColorTheme, OutputFormat

# Import main function for direct calling in tests
try:
    from plagcheck import cli
except ImportError:
    # If main is not directly importable, use a mock
    main = None


class TestEndToEnd:
    """Integration tests that verify the entire pipeline from files to results."""

    def test_simple_plagiarism_detection_with_real_files(self, test_files_dir):
        """Test detecting similarity between real files in the files directory."""
        # Initialize components manually
        file_handler = FileHandler(directory=test_files_dir, file_extension='.txt')
        
        # Load files
        documents = file_handler.load_files()
        assert len(documents) > 0, "No test files found"
        
        # Setup text processor with normalization
        text_processor = TextProcessor(
            normalize_operations=[
                TextNormalizer.to_lowercase,
                TextNormalizer.remove_punctuation,
                TextNormalizer.remove_whitespace
            ],
            stopword_remover=StopwordRemover()
        )
        
        # Setup document similarity calculator
        doc_similarity = DocumentSimilarity(preprocessor=text_processor)
        
        # Get document names and texts
        doc_names = list(documents.keys())
        doc_texts = list(documents.values())
        
        # Calculate similarities
        results = doc_similarity.compare_texts(
            doc_texts,
            names=doc_names,
            metric='cosine',
            vectorizer='tfidf',
            ngram_range=(1, 1),
            binary=False,
            preprocess=True
        )
        
        # Assert we have results
        assert len(results) > 0, "No similarity results generated"
        
        # Check the format of results (set of tuples)
        assert isinstance(results, set), "Results should be a set"
        for result in results:
            assert isinstance(result, tuple), "Each result should be a tuple"
            assert len(result) == 3, "Each result tuple should have 3 elements"
            assert isinstance(result[0], str), "First element should be a document name"
            assert isinstance(result[1], str), "Second element should be a document name"
            assert isinstance(result[2], float), "Third element should be a float similarity score"
        
        # Check that at least one pair has non-zero similarity
        has_non_zero_pairs = any(similarity > 0.01 for _, _, similarity in results)
        assert has_non_zero_pairs, "Expected to find non-zero similarity between some documents"


    def test_cli_invocation_through_subprocess(self, temp_test_dir):
        """Test CLI invocation through subprocess."""
        # Create test files in the temporary directory
        identical_text = "This is a test document that will be identical to another document."
        similar_text = "This is a test document that is quite similar to another document."
        different_text = "This document has completely different content than the others."
        
        # Create test files
        (temp_test_dir / "doc1.txt").write_text(identical_text)
        (temp_test_dir / "doc2.txt").write_text(identical_text)  # Identical to doc1
        (temp_test_dir / "doc3.txt").write_text(similar_text)    # Similar to doc1 and doc2
        (temp_test_dir / "doc4.txt").write_text(different_text)  # Different 
        
        # Run the CLI command
        output_file = temp_test_dir / "results.json"
        cmd = [
            sys.executable, 
            "main.py", 
            "--dir", str(temp_test_dir),
            "--normalize", 
            "--similarity", "cosine",
            "--output", str(output_file),
            "--output-format", "json"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check if execution was successful
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Check if output file was created
            assert output_file.exists(), "Output file was not created"
            
            # Parse output and verify results
            with open(output_file, 'r') as f:
                data = json.load(f)
                
                # Check that the JSON file contains results
                assert len(data) > 0, "No results in the JSON file"
                
                # Verify we have content in the output file that looks like similarity results
                json_str = json.dumps(data)
                # Look for key phrases that suggest similarity results
                has_doc_references = any(f"doc{i}" in json_str for i in range(1, 5))
                has_numbers = any(str(i/10) in json_str for i in range(11))  # Check for numbers like 0.1, 0.2, etc.
                
                assert has_doc_references, "Output should have references to documents"
                assert has_numbers, "Output should have similarity scores"
                
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            pytest.skip(f"Subprocess test skipped: {e}")


class TestOutputFormats:
    """Tests for different output formats of the plagiarism checker."""

    def test_csv_output_format(self, temp_test_dir):
        """Test CSV output format."""
        # Create test files
        (temp_test_dir / "doc1.txt").write_text("File 1 content")
        (temp_test_dir / "doc2.txt").write_text("File 2 content")
        
        # Setup the file handler and load files
        file_handler = FileHandler(directory=str(temp_test_dir), file_extension='.txt')
        documents = file_handler.load_files()
        
        # Setup document similarity calculator with default text processor
        doc_similarity = DocumentSimilarity()
        
        # Calculate similarities
        doc_names = list(documents.keys())
        doc_texts = list(documents.values())
        results = doc_similarity.compare_texts(
            doc_texts,
            names=doc_names,
            metric='cosine',
            preprocess=True
        )
        
        # Initialize result presenter
        presenter = ResultPresenter()
        
        # Create a CSV output file
        output_file = temp_test_dir / "results.csv"
        presenter.save_results(results, str(output_file), format='csv')
        
        # Verify the CSV file
        assert output_file.exists(), "CSV file was not created"
        
        # Read and check the CSV content
        with open(output_file, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        # Check that we have rows (header + data)
        assert len(rows) >= 2, "CSV file should have at least a header and data row"
        
        # Check that rows have at least 3 columns
        for row in rows:
            assert len(row) >= 3, "Each row should have at least 3 columns"


class TestPreprocessingIntegration:
    """Tests for text preprocessing integration with similarity calculation."""

    def test_normalization_effect(self):
        """Test how normalization affects text processing."""
        # Test documents before and after normalization
        original_text = "  This is a TEST document, with Punctuation!  "
        
        # Apply individual normalization steps directly
        lowercase_text = TextNormalizer.to_lowercase(original_text)
        no_punct_text = TextNormalizer.remove_punctuation(original_text)
        no_whitespace_text = TextNormalizer.remove_whitespace(original_text)
        
        # Verify each step has the expected effect
        assert lowercase_text == "  this is a test document, with punctuation!  "
        assert "," not in no_punct_text and "!" not in no_punct_text
        assert no_whitespace_text == "This is a TEST document, with Punctuation!"


    def test_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline."""
        # Test document with various features
        text = "This IS a test document. It contains STOPWORDS like 'the' and 'a'."
        
        # Create a text processor with full preprocessing
        processor = TextProcessor(
            normalize_operations=[
                TextNormalizer.to_lowercase,
                TextNormalizer.remove_punctuation,
                TextNormalizer.remove_whitespace
            ],
            stopword_remover=StopwordRemover()
        )
        
        # Preprocess the text
        processed_text = processor.preprocess(text)
        
        # Check that preprocessing had the expected effects
        assert "IS" not in processed_text, "Text should be lowercase"
        assert "." not in processed_text, "Punctuation should be removed"
        assert "  " not in processed_text, "Extra whitespace should be removed"
        
        # Check if common stopwords were removed
        # Note: This depends on what's in the StopwordRemover's default set
        common_stopwords = ['the', 'a', 'is', 'and']
        for stopword in common_stopwords:
            # If any stopword remains, it should at least be part of another word,
            # not a standalone word
            if stopword in processed_text:
                assert f" {stopword} " not in f" {processed_text} ", f"Stopword '{stopword}' should be removed"


class TestSimilarityMetricsIntegration:
    """Tests for different similarity metrics in an end-to-end context."""

    @pytest.mark.parametrize("metric", [
        'cosine', 'jaccard', 'euclidean', 'manhattan'
    ])
    def test_different_similarity_metrics(self, metric):
        """Test different similarity metrics on the same document pairs."""
        # Very similar documents
        doc1 = "This is a test document for similarity calculation"
        doc2 = "This is also a test document for similarity calculation"
        
        # Calculate similarity using specified metric
        doc_similarity = DocumentSimilarity(
            preprocessor=TextProcessor(
                normalize_operations=[TextNormalizer.to_lowercase]
            )
        )
        
        results = doc_similarity.compare_texts(
            [doc1, doc2],
            names=["doc1", "doc2"],
            metric=metric,
            preprocess=True
        )
        
        # Extract similarity score
        similarity_score = next(score for _, _, score in results)
        
        # For similar documents, all metrics should give scores in [0,1]
        assert 0.0 <= similarity_score <= 1.0, f"{metric} similarity should be between 0 and 1"


class TestComplexFlows:
    """Tests for more complex end-to-end flows."""

    def test_folder_processing_with_multiple_output_formats(self, temp_test_dir):
        """Test processing a folder of documents with multiple output formats."""
        # Create test files
        for i in range(1, 5):
            (temp_test_dir / f"doc{i}.txt").write_text(f"Document {i} content with some unique words {i*i}")
        
        # Create a document with copied content to simulate plagiarism
        (temp_test_dir / "copied.txt").write_text("Document 1 content with some unique words 1")
        
        # Setup output directory
        output_dir = temp_test_dir / "results"
        output_dir.mkdir(exist_ok=True)
        
        # Setup the file handler and load files
        file_handler = FileHandler(directory=str(temp_test_dir), file_extension='.txt')
        documents = file_handler.load_files()
        
        # Setup document similarity calculator
        doc_similarity = DocumentSimilarity(
            preprocessor=TextProcessor(
                normalize_operations=[
                    TextNormalizer.to_lowercase,
                    TextNormalizer.remove_punctuation
                ]
            )
        )
        
        # Calculate similarities
        doc_names = list(documents.keys())
        doc_texts = list(documents.values())
        
        results = doc_similarity.compare_texts(
            doc_texts,
            names=doc_names,
            metric='cosine',
            preprocess=True
        )
        
        # Initialize result presenter
        presenter = ResultPresenter()
        
        # Save results in multiple formats
        formats = ['json', 'csv', 'markdown']
        for fmt in formats:
            output_file = output_dir / f"results.{fmt}"
            presenter.save_results(results, str(output_file), format=fmt)
            
            # Verify the file was created
            assert output_file.exists(), f"{fmt.upper()} file was not created"
        
        # Check for high similarity between copied documents
        found_high_similarity = False
        doc1_name = "doc1.txt"
        copied_name = "copied.txt"
        
        for doc1, doc2, score in results:
            if (doc1 == doc1_name and doc2 == copied_name) or (doc1 == copied_name and doc2 == doc1_name):
                found_high_similarity = score > 0.9
                break
        
        assert found_high_similarity, "Failed to detect high similarity between copied documents"


    def test_different_ngram_configurations(self, temp_test_dir):
        """Test plagiarism detection with different n-gram configurations."""
        # Create test files
        doc1 = "This is a test document for plagiarism detection"
        doc2 = "This is a document for testing plagiarism detection"  # Similar but reworded
        
        (temp_test_dir / "doc1.txt").write_text(doc1)
        (temp_test_dir / "doc2.txt").write_text(doc2)
        
        # Try different n-gram ranges
        ngram_ranges = [(1, 1), (2, 2), (1, 2)]
        
        for ngram_range in ngram_ranges:
            # Setup document similarity calculator
            doc_similarity = DocumentSimilarity(
                preprocessor=TextProcessor(normalize_operations=[TextNormalizer.to_lowercase])
            )
            
            # Calculate similarities
            results = doc_similarity.compare_texts(
                [doc1, doc2],
                names=["doc1", "doc2"],
                metric='cosine',
                vectorizer='tfidf',
                ngram_range=ngram_range,
                preprocess=True
            )
            
            # Extract similarity score
            similarity_score = next(score for _, _, score in results)
            
            # All configurations should produce valid scores
            assert 0.0 <= similarity_score <= 1.0, f"N-gram range {ngram_range} produced invalid score"
            
            # Score should be reasonable for similar documents
            assert similarity_score >= 0.1, f"N-gram range {ngram_range} produced unexpectedly low score"


@pytest.mark.skipif(main is None, reason="main function not importable")
class TestDirectMainCalls:
    """Tests that call the main function directly."""

    def test_main_function_with_args(self, monkeypatch, temp_test_dir):
        """Test the main function directly, passing args through monkeypatch."""
        # Create test files
        (temp_test_dir / "doc1.txt").write_text("First test document")
        (temp_test_dir / "doc2.txt").write_text("Second test document")
        
        # Create a CSV output file path
        output_file = temp_test_dir / "direct_call_results.csv"
        
        # Prepare arguments to pass to the main function
        test_args = [
            "main.py",  # Script name
            "--dir", str(temp_test_dir),
            "--normalize",
            "--output", str(output_file),
            "--output-format", "csv"
        ]
        
        # Patch sys.argv
        monkeypatch.setattr('sys.argv', test_args)
        
        # Capture stdout to prevent output during tests
        output = io.StringIO()
        with redirect_stdout(output):
            # Call the main function
            try:
                main()
            except SystemExit:
                # The main function might call sys.exit, which is fine
                pass
        
        # Verify the output file was created
        assert output_file.exists(), "Output file was not created when calling main directly"
        
        # Verify the file has content
        with open(output_file, 'r') as f:
            content = f.read()
            assert len(content) > 0, "Output file is empty"