"""
Unit tests for the text processing module of the plagiarism checker.
"""
import unittest
import sys
import os
from typing import List, Set

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plagcheck.modules.text_processor import TextNormalizer, Tokenizer, StopwordRemover, TextProcessor

class TestTextNormalizer(unittest.TestCase):
    """Test cases for the TextNormalizer class."""
    
    def test_to_lowercase(self):
        """Test converting text to lowercase."""
        text = "This Is A SAMPLE TEXT with Mixed Case."
        expected = "this is a sample text with mixed case."
        result = TextNormalizer.to_lowercase(text)
        self.assertEqual(result, expected)
    
    def test_remove_punctuation(self):
        """Test removing punctuation from text."""
        text = "Hello, world! This is a test. How are you?"
        expected = "Hello world This is a test How are you"
        result = TextNormalizer.remove_punctuation(text)
        self.assertEqual(result, expected)
    
    def test_remove_whitespace(self):
        """Test removing extra whitespace from text."""
        text = "  This   has    extra    spaces  \t and \n tabs."
        expected = "This has extra spaces and tabs."
        result = TextNormalizer.remove_whitespace(text)
        self.assertEqual(result, expected)
    
    def test_remove_numbers(self):
        """Test removing numbers from text."""
        text = "This contains 123 numbers and 456.78 decimals."
        expected = "This contains  numbers and . decimals."
        result = TextNormalizer.remove_numbers(text)
        self.assertEqual(result, expected)
    
    def test_normalize_unicode(self):
        """Test normalizing Unicode characters."""
        text = "Café résumé naïve"
        expected = "Cafe resume naive"
        result = TextNormalizer.normalize_unicode(text)
        self.assertEqual(result, expected)
    
    def test_combined_operations(self):
        """Test combining multiple normalization operations."""
        text = "Hello, World! Text with 123 Numbers & Symbols."
        normalized = TextNormalizer.to_lowercase(text)
        normalized = TextNormalizer.remove_punctuation(normalized)
        normalized = TextNormalizer.remove_numbers(normalized)
        expected = "hello world text with  numbers  symbols"
        self.assertEqual(normalized, expected)


class TestTokenizer(unittest.TestCase):
    """Test cases for the Tokenizer class."""
    
    def test_tokenize_words(self):
        """Test tokenizing text into words."""
        text = "This is a simple sentence."
        expected = ["This", "is", "a", "simple", "sentence."]
        tokens = Tokenizer.tokenize_words(text)
        self.assertEqual(tokens, expected)
    
    def test_tokenize_sentences(self):
        """Test tokenizing text into sentences."""
        text = "This is sentence one. This is sentence two! And number three?"
        expected = ["This is sentence one.", "This is sentence two!", "And number three?"]
        sentences = Tokenizer.tokenize_sentences(text)
        self.assertEqual(sentences, expected)
    
    def test_tokenize_character_ngrams(self):
        """Test generating character n-grams from text."""
        text = "hello"
        
        # Test bigrams (n=2)
        bigrams = Tokenizer.tokenize_ngrams(text, 2)
        expected_bigrams = ["he", "el", "ll", "lo"]
        self.assertEqual(bigrams, expected_bigrams)
        
        # Test trigrams (n=3)
        trigrams = Tokenizer.tokenize_ngrams(text, 3)
        expected_trigrams = ["hel", "ell", "llo"]
        self.assertEqual(trigrams, expected_trigrams)
    
    def test_tokenize_word_ngrams(self):
        """Test generating word n-grams from text."""
        text = "this is a test sentence"
        
        # Test word bigrams (n=2)
        bigrams = Tokenizer.tokenize_word_ngrams(text, 2)
        expected_bigrams = ["this is", "is a", "a test", "test sentence"]
        self.assertEqual(bigrams, expected_bigrams)
        
        # Test word trigrams (n=3)
        trigrams = Tokenizer.tokenize_word_ngrams(text, 3)
        expected_trigrams = ["this is a", "is a test", "a test sentence"]
        self.assertEqual(trigrams, expected_trigrams)


class TestStopwordRemover(unittest.TestCase):
    """Test cases for the StopwordRemover class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.stopword_remover = StopwordRemover()
        self.custom_stopwords = {"custom", "stopword", "example"}
        # Include default stopwords plus custom ones
        self.custom_remover = StopwordRemover(custom_stopwords=self.custom_stopwords)
    
    def test_default_stopwords(self):
        """Test removing default English stopwords."""
        text = "This is a test with the and some other stopwords."
        processed = self.stopword_remover.remove(text)
        
        # Check that common stopwords are removed
        self.assertNotIn("is", processed.split())
        self.assertNotIn("a", processed.split())
        self.assertNotIn("the", processed.split())
        self.assertNotIn("and", processed.split())
        self.assertNotIn("with", processed.split())
        
        # Check that "This" is also removed (case-insensitive comparison)
        self.assertNotIn("This", processed.split())
        
        # Check that content words remain
        self.assertIn("test", processed.split())
        self.assertIn("some", processed.split())  # "some" might not be in the default stopwords
        self.assertIn("other", processed.split())
        self.assertIn("stopwords.", processed.split())
    
    def test_custom_stopwords(self):
        """Test removing custom stopwords."""
        text = "This custom example contains stopword to be removed."
        processed = self.custom_remover.remove(text)
        
        expected_words = ["contains", "removed."]
        processed_words = processed.split()
        
        # Check that custom stopwords are removed
        self.assertNotIn("custom", processed_words)
        self.assertNotIn("stopword", processed_words)
        self.assertNotIn("example", processed_words)
        
        # Check that default stopwords are also removed
        self.assertNotIn("This", processed_words)
        self.assertNotIn("to", processed_words)
        self.assertNotIn("be", processed_words)
        
        # Check that remaining words match exactly what we expect
        self.assertEqual(sorted(processed_words), sorted(expected_words))
    
    def test_add_stopwords(self):
        """Test adding new stopwords to the set."""
        remover = StopwordRemover(include_default=False)
        self.assertEqual(remover.remove("test word"), "test word")  # No stopwords yet
        
        remover.add_stopwords(["test"])
        self.assertEqual(remover.remove("test word"), "word")  # "test" is now a stopword
    
    def test_remove_stopwords(self):
        """Test removing words from the stopword set."""
        # Create a remover with only a few stopwords
        remover = StopwordRemover(custom_stopwords={"test", "word", "example"}, include_default=False)
        
        # Initially "test" is a stopword
        self.assertEqual(remover.remove("test example"), "")
        
        # Remove "test" from stopword list
        remover.remove_stopwords(["test"])
        self.assertEqual(remover.remove("test example"), "test")


class TestTextProcessor(unittest.TestCase):
    """Test cases for the TextProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Default text processor
        self.processor = TextProcessor()
        
        # Text processor with custom settings
        self.custom_processor = TextProcessor(
            normalize_operations=[
                TextNormalizer.to_lowercase,
                TextNormalizer.remove_punctuation,
                TextNormalizer.remove_numbers
            ],
            tokenizer=Tokenizer.tokenize_sentences,
            stopword_remover=StopwordRemover(custom_stopwords={"custom", "stopword"}),
            custom_preprocessing=[lambda text: text.replace("example", "EXAMPLE")]
        )
    
    def test_normalize(self):
        """Test text normalization."""
        text = "This is An EXAMPLE text, with Punctuation and   Spaces!"
        normalized = self.processor.normalize(text)
        # Default normalization: lowercase, remove punctuation, remove extra whitespace
        expected = "this is an example text with punctuation and spaces"
        self.assertEqual(normalized, expected)
    
    def test_custom_normalize(self):
        """Test text normalization with custom operations."""
        text = "This is An EXAMPLE text, with 123 numbers!"
        normalized = self.custom_processor.normalize(text)
        # Custom normalization: lowercase, remove punctuation, remove numbers
        expected = "this is an example text with  numbers"
        self.assertEqual(normalized, expected)
    
    def test_preprocess(self):
        """Test full text preprocessing."""
        text = "This is an example with stopwords like the and a."
        processed = self.processor.preprocess(text)
        
        # Check that normalization and stopword removal are applied
        self.assertEqual(processed.lower(), processed)  # Lowercase applied
        self.assertNotIn("the", processed.split())     # Common stopword removed
        self.assertNotIn("and", processed.split())     # Common stopword removed
        self.assertNotIn("a", processed.split())       # Common stopword removed
        self.assertIn("example", processed.split())    # Content word preserved
    
    def test_custom_preprocess(self):
        """Test custom preprocessing pipeline."""
        text = "This is an example with custom stopword."
        processed = self.custom_processor.preprocess(text)
        
        # Check that custom normalization, stopword removal, and preprocessing are applied
        self.assertNotIn("custom", processed.split())  # Custom stopword removed
        self.assertIn("EXAMPLE", processed)            # Custom preprocessing applied
    
    def test_preprocess_documents(self):
        """Test preprocessing multiple documents."""
        documents = [
            "This is document one.",
            "This is document two with more stopwords.",
            "Document three has numbers like 123."
        ]
        
        processed = self.processor.preprocess_documents(documents)
        
        # Check that we have the right number of documents
        self.assertEqual(len(processed), len(documents))
        
        # Check that preprocessing was applied to each document
        for doc in processed:
            self.assertEqual(doc.lower(), doc)  # Lowercase applied
            self.assertNotIn("is", doc.split())  # Common stopword removed


if __name__ == '__main__':
    unittest.main()
