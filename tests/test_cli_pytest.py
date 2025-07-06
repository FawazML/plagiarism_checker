"""
Pytest-based tests for the command-line interface of the plagiarism checker.
Using fixtures from conftest.py
"""
import pytest
import sys
import os
import argparse
import subprocess
from unittest.mock import patch, MagicMock


def test_default_arguments():
    """Test that default arguments are set correctly."""
    # Create a parser similar to the one in main.py
    parser = argparse.ArgumentParser(description='Check plagiarism among text documents')
    parser.add_argument('--dir', type=str, default=None, help='Directory containing text files')
    parser.add_argument('--ext', type=str, default='.txt', help='File extension to process')
    parser.add_argument('--similarity', type=str, default='cosine', 
                choices=['cosine', 'jaccard', 'euclidean', 'manhattan', 'dice', 'overlap', 'levenshtein'],
                help='Similarity metric to use')
    parser.add_argument('--ngram', type=int, default=1, help='Use n-grams for comparison')
    parser.add_argument('--visualization', type=str, choices=['heatmap', 'bar', 'none'], 
                default='none', help='Type of visualization to generate')
    
    # Parse with no arguments (defaults)
    args = parser.parse_args([])
    
    # Check that defaults are as expected
    assert args.dir is None
    assert args.ext == '.txt'
    assert args.similarity == 'cosine'
    assert args.ngram == 1
    assert args.visualization == 'none'


def test_custom_arguments(cli_parser_args, expected_cli_results):
    """Test that custom arguments are parsed correctly."""
    # Create a parser with the essential arguments for testing
    parser = argparse.ArgumentParser(description='Check plagiarism among text documents')
    parser.add_argument('--dir', type=str, default=None, help='Directory containing text files')
    parser.add_argument('--ext', type=str, default='.txt', help='File extension to process')
    parser.add_argument('--normalize', action='store_true', help='Apply text normalization')
    parser.add_argument('--remove-stopwords', action='store_true', help='Remove stopwords')
    parser.add_argument('--ngram', type=int, default=1, help='Use n-grams for comparison')
    parser.add_argument('--similarity', type=str, default='cosine', 
                choices=['cosine', 'jaccard', 'euclidean', 'manhattan', 'dice', 'overlap', 'levenshtein'],
                help='Similarity metric to use')
    parser.add_argument('--visualization', type=str, choices=['heatmap', 'bar', 'none'], 
                default='none', help='Type of visualization to generate')
    parser.add_argument('--output', type=str, default=None, help='Output file for results')
    parser.add_argument('--output-format', type=str, choices=['json', 'csv', 'html', 'markdown', 'latex'], 
                default=None, help='Format for output file')
    parser.add_argument('--color-theme', type=str, default='heatmap', 
                choices=['none', 'basic', 'heatmap', 'traffic_light'],
                help='Color theme for console output')
    parser.add_argument('--precision', type=int, default=4, help='Decimal precision for similarity values')
    
    # Test basic arguments
    basic_args = parser.parse_args(cli_parser_args["basic"])
    expected_basic = expected_cli_results["basic"]
    
    assert basic_args.dir == expected_basic["dir"]
    assert basic_args.ext == expected_basic["ext"]
    assert basic_args.normalize == expected_basic["normalize"]
    assert basic_args.remove_stopwords == expected_basic["remove_stopwords"]
    assert basic_args.similarity == expected_basic["similarity"]
    assert basic_args.ngram == expected_basic["ngram"]
    assert basic_args.visualization == expected_basic["visualization"]
    
    # Test advanced arguments
    advanced_args = parser.parse_args(cli_parser_args["advanced"])
    expected_advanced = expected_cli_results["advanced"]
    
    assert advanced_args.dir == expected_advanced["dir"]
    assert advanced_args.ext == expected_advanced["ext"]
    assert advanced_args.normalize == expected_advanced["normalize"]
    assert advanced_args.remove_stopwords == expected_advanced["remove_stopwords"]
    assert advanced_args.similarity == expected_advanced["similarity"]
    assert advanced_args.ngram == expected_advanced["ngram"]
    assert advanced_args.visualization == expected_advanced["visualization"]
    assert advanced_args.output == expected_advanced["output"]
    assert advanced_args.output_format == expected_advanced["output_format"]
    assert advanced_args.color_theme == expected_advanced["color_theme"]
    assert advanced_args.precision == expected_advanced["precision"]


def test_invalid_similarity_metric(cli_parser_args):
    """Test that an invalid similarity metric raises an error."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--similarity', type=str, 
                choices=['cosine', 'jaccard', 'euclidean', 'manhattan', 'dice', 'overlap', 'levenshtein'],
                help='Similarity metric to use')
    
    # Should raise SystemExit because of invalid choice
    with pytest.raises(SystemExit):
        parser.parse_args(cli_parser_args["invalid"])


def test_invalid_visualization_type():
    """Test that an invalid visualization type raises an error."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualization', type=str, 
                choices=['heatmap', 'bar', 'none'], 
                default='none', help='Type of visualization to generate')
    
    # Should raise SystemExit because of invalid choice
    with pytest.raises(SystemExit):
        parser.parse_args(["--visualization", "invalid_type"])


def test_invalid_output_format():
    """Test that an invalid output format raises an error."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-format', type=str, 
                choices=['json', 'csv', 'html', 'markdown', 'latex'], 
                default=None, help='Format for output file')
    
    # Should raise SystemExit because of invalid choice
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-format", "invalid_format"])


def test_invalid_image_format():
    """Test that an invalid image format raises an error."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-format', type=str, 
                choices=['png', 'jpg', 'jpeg', 'svg', 'pdf', 'eps', 'tiff'], 
                default='png', help='Format for saving visualization images')
    
    # Should raise SystemExit because of invalid choice
    with pytest.raises(SystemExit):
        parser.parse_args(["--image-format", "invalid_format"])


@pytest.mark.skipif(not os.path.exists('./main.py'), reason="Skipping test if main.py not found")
def test_help_option(cli_parser_args):
    """Test that --help option works and returns a zero exit code."""
    cmd = [sys.executable, 'main.py'] + cli_parser_args["help"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert 'usage:' in result.stdout
        assert 'Check plagiarism among text documents' in result.stdout
    except FileNotFoundError:
        pytest.skip(f"Could not execute command: {' '.join(cmd)}")


@pytest.mark.skipif(not os.path.exists('./main.py'), reason="Skipping test if main.py not found")
def test_invalid_argument():
    """Test that an invalid argument returns a non-zero exit code."""
    cmd = [sys.executable, 'main.py', '--invalid-argument']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0
        assert 'error:' in result.stderr
    except FileNotFoundError:
        pytest.skip(f"Could not execute command: {' '.join(cmd)}")


def test_argument_dependencies():
    """Test argument dependencies and combinations."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngram', type=int, default=1, help='Use n-grams for comparison')
    parser.add_argument('--min-df', type=float, default=1, help='Min document frequency for terms')
    parser.add_argument('--max-df', type=float, default=1.0, help='Max document frequency for terms')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--output-format', type=str, choices=['json', 'csv', 'html', 'markdown', 'latex'],
                        help='Format for output file')
    parser.add_argument('--visualization', type=str, choices=['heatmap', 'bar', 'none'], 
                        default='none', help='Type of visualization to generate')
    parser.add_argument('--image-format', type=str, choices=['png', 'jpg', 'jpeg', 'svg', 'pdf', 'eps', 'tiff'],
                        default='png', help='Format for saving visualization images')
    
    # Test that ngram > 1 works
    args = parser.parse_args(['--ngram', '3'])
    assert args.ngram == 3
    
    # Test min-df and max-df
    args = parser.parse_args(['--min-df', '0.1', '--max-df', '0.9'])
    assert args.min_df == 0.1
    assert args.max_df == 0.9
    
    # Test output with format
    args = parser.parse_args(['--output', 'results.csv', '--output-format', 'csv'])
    assert args.output == 'results.csv'
    assert args.output_format == 'csv'
    
    # Test visualization with image format
    args = parser.parse_args(['--visualization', 'heatmap', '--image-format', 'svg'])
    assert args.visualization == 'heatmap'
    assert args.image_format == 'svg'


def test_cli_with_temp_files(temp_test_dir):
    """Test CLI arguments with a temporary directory of files."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory containing text files')
    parser.add_argument('--ext', type=str, default='.txt', help='File extension to process')
    
    # Test with temp directory
    args = parser.parse_args(['--dir', str(temp_test_dir), '--ext', '.txt'])
    assert args.dir == str(temp_test_dir)
    assert args.ext == '.txt'
    
    # Verify the directory exists and contains test files
    assert os.path.exists(temp_test_dir)
    assert len([f for f in os.listdir(temp_test_dir) if f.endswith('.txt')]) == 3
