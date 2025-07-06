"""
Tests for the command-line interface of the plagiarism checker.
"""

import unittest
import sys
import os
import argparse
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path

class TestCommandLineInterface(unittest.TestCase):
    """Test cases for the command-line interface."""
    
    def setUp(self):
        """Set up a parser similar to the one in main.py for testing."""
        self.parser = argparse.ArgumentParser(description='Check plagiarism among text documents')
        self.parser.add_argument('--dir', type=str, default=None, help='Directory containing text files')
        self.parser.add_argument('--ext', type=str, default='.txt', help='File extension to process')
        self.parser.add_argument('--output', type=str, default=None, help='Output file for results')
        self.parser.add_argument('--output-format', type=str, choices=['json', 'csv', 'html', 'markdown', 'latex'], 
                        default=None, help='Format for output file')
        self.parser.add_argument('--normalize', action='store_true', help='Apply text normalization')
        self.parser.add_argument('--remove-stopwords', action='store_true', help='Remove stopwords')
        self.parser.add_argument('--ngram', type=int, default=1, 
                        help='Use n-grams for comparison (n=1 for words, n>1 for char n-grams)')
        self.parser.add_argument('--min-df', type=float, default=1, help='Min document frequency for terms')
        self.parser.add_argument('--max-df', type=float, default=1.0, help='Max document frequency for terms')
        self.parser.add_argument('--similarity', type=str, default='cosine', 
                        choices=['cosine', 'jaccard', 'euclidean', 'manhattan', 'dice', 'overlap', 'levenshtein'],
                        help='Similarity metric to use')
        self.parser.add_argument('--binary', action='store_true', 
                        help='Use binary vectorization (0/1 instead of counts/weights)')
        self.parser.add_argument('--compare-all', action='store_true', 
                        help='Compare documents with all similarity metrics')
        self.parser.add_argument('--color-theme', type=str, default='heatmap', 
                        choices=['none', 'basic', 'heatmap', 'traffic_light'],
                        help='Color theme for console output')
        self.parser.add_argument('--output-dir', type=str, default=None, 
                        help='Directory for saving comprehensive report with multiple formats')
        self.parser.add_argument('--visualization', type=str, choices=['heatmap', 'bar', 'none'], 
                        default='none', help='Type of visualization to generate')
        self.parser.add_argument('--table-format', type=str, default='grid', 
                        help='Table format for console output (see tabulate docs for options)')
        self.parser.add_argument('--show-summary', action='store_true', 
                        help='Show summary statistics of similarity results')
        self.parser.add_argument('--precision', type=int, default=4, 
                        help='Decimal precision for similarity values')
        self.parser.add_argument("--image-format", type=str, 
                        choices=["png", "jpg", "jpeg", "svg", "pdf", "eps", "tiff"], 
                        default="png", help="Format for saving visualization images")
    
    def test_default_arguments(self):
        """Test that default arguments are set correctly."""
        # Parse with no arguments (defaults)
        args = self.parser.parse_args([])
            
        # Check that defaults are as expected
        self.assertIsNone(args.dir)
        self.assertEqual(args.ext, '.txt')
        self.assertIsNone(args.output)
        self.assertIsNone(args.output_format)
        self.assertFalse(args.normalize)
        self.assertFalse(args.remove_stopwords)
        self.assertEqual(args.ngram, 1)
        self.assertEqual(args.similarity, 'cosine')
        self.assertFalse(args.binary)
        self.assertFalse(args.compare_all)
        self.assertEqual(args.color_theme, 'heatmap')
        self.assertEqual(args.visualization, 'none')
        self.assertEqual(args.table_format, 'grid')
        self.assertFalse(args.show_summary)
        self.assertEqual(args.precision, 4)
        self.assertEqual(args.image_format, 'png')
    
    def test_custom_arguments(self):
        """Test that custom arguments are parsed correctly."""
        # Test with custom arguments
        args = self.parser.parse_args([
            "--dir", "test_files",
            "--ext", ".md",
            "--similarity", "jaccard",
            "--ngram", "3",
            "--visualization", "heatmap",
            "--normalize",
            "--remove-stopwords",
            "--binary",
            "--output", "results.html",
            "--output-format", "html",
            "--color-theme", "traffic_light",
            "--precision", "2",
            "--image-format", "pdf"
        ])
            
        # Check that arguments are parsed correctly
        self.assertEqual(args.dir, "test_files")
        self.assertEqual(args.ext, ".md")
        self.assertEqual(args.similarity, "jaccard")
        self.assertEqual(args.ngram, 3)
        self.assertEqual(args.visualization, "heatmap")
        self.assertTrue(args.normalize)
        self.assertTrue(args.remove_stopwords)
        self.assertTrue(args.binary)
        self.assertEqual(args.output, "results.html")
        self.assertEqual(args.output_format, "html")
        self.assertEqual(args.color_theme, "traffic_light")
        self.assertEqual(args.precision, 2)
        self.assertEqual(args.image_format, "pdf")
    
    def test_invalid_similarity_metric(self):
        """Test that an invalid similarity metric raises an error."""
        # Should raise SystemExit because of invalid choice
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--similarity", "invalid_metric"])
    
    def test_invalid_visualization_type(self):
        """Test that an invalid visualization type raises an error."""
        # Should raise SystemExit because of invalid choice
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--visualization", "invalid_type"])
    
    def test_invalid_output_format(self):
        """Test that an invalid output format raises an error."""
        # Should raise SystemExit because of invalid choice
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--output-format", "invalid_format"])
    
    def test_invalid_image_format(self):
        """Test that an invalid image format raises an error."""
        # Should raise SystemExit because of invalid choice
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--image-format", "invalid_format"])
    
    @unittest.skipIf(not os.path.exists('./main.py'), "Skipping test if main.py not found")
    def test_help_option(self):
        """Test that --help option works and returns a zero exit code."""
        cmd = [sys.executable, 'main.py', '--help']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertIn('usage:', result.stdout)
            self.assertIn('Check plagiarism among text documents', result.stdout)
        except FileNotFoundError:
            self.skipTest(f"Could not execute command: {' '.join(cmd)}")
    
    @unittest.skipIf(not os.path.exists('./main.py'), "Skipping test if main.py not found")
    def test_invalid_argument(self):
        """Test that an invalid argument returns a non-zero exit code."""
        cmd = [sys.executable, 'main.py', '--invalid-argument']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('error:', result.stderr)
        except FileNotFoundError:
            self.skipTest(f"Could not execute command: {' '.join(cmd)}")
    
    def test_argument_dependencies(self):
        """Test argument dependencies and combinations."""
        # Test that ngram > 1 works
        args = self.parser.parse_args(['--ngram', '3'])
        self.assertEqual(args.ngram, 3)
        
        # Test min-df and max-df
        args = self.parser.parse_args(['--min-df', '0.1', '--max-df', '0.9'])
        self.assertEqual(args.min_df, 0.1)
        self.assertEqual(args.max_df, 0.9)
        
        # Test output with format
        args = self.parser.parse_args(['--output', 'results.csv', '--output-format', 'csv'])
        self.assertEqual(args.output, 'results.csv')
        self.assertEqual(args.output_format, 'csv')
        
        # Test visualization with image format
        args = self.parser.parse_args(['--visualization', 'heatmap', '--image-format', 'svg'])
        self.assertEqual(args.visualization, 'heatmap')
        self.assertEqual(args.image_format, 'svg')

if __name__ == '__main__':
    unittest.main()
