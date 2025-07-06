# text_processor.py
import re
import string
import unicodedata
from typing import List, Dict, Any, Callable, Set, Optional, Union, Tuple
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class TextNormalizer:
    """Class for text normalization operations."""
    
    @staticmethod
    def to_lowercase(text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Remove punctuation from text."""
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    @staticmethod
    def remove_whitespace(text: str) -> str:
        """Remove extra whitespace from text."""
        return ' '.join(text.split())
    
    @staticmethod
    def remove_numbers(text: str) -> str:
        """Remove numbers from text."""
        return re.sub(r'\d+', '', text)
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize Unicode characters to their closest ASCII representation."""
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')


class Tokenizer:
    """Class for text tokenization operations."""
    
    @staticmethod
    def tokenize_words(text: str) -> List[str]:
        """Split text into words."""
        return text.split()
    
    @staticmethod
    def tokenize_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        return re.split(r'(?<=[.!?])\s+', text)
    
    @staticmethod
    def tokenize_ngrams(text: str, n: int = 2) -> List[str]:
        """Generate character n-grams from text."""
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    @staticmethod
    def tokenize_word_ngrams(text: str, n: int = 2) -> List[str]:
        """Generate word n-grams from text."""
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]


class StopwordRemover:
    """Class for stopword removal operations."""
    
    # Common English stopwords
    ENGLISH_STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
        'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
        'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
        'to', 'from', 'in', 'on', 'at', 'by', 'about', 'like', 'with', 'after',
        'between', 'into', 'through', 'during', 'before', 'without', 'under',
        'within', 'along', 'following', 'across', 'behind', 'beyond', 'plus',
        'except', 'but', 'up', 'out', 'around', 'down', 'off', 'above', 'near',
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'am', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
        'doing', 'can', 'could', 'should', 'would', 'ought', 'will', 'shall', 
        'may', 'might', 'must', 'now'
    }
    
    def __init__(self, custom_stopwords: Optional[Set[str]] = None, 
                 include_default: bool = True):
        """
        Initialize stopword remover with custom stopwords.
        
        Args:
            custom_stopwords: Optional set of custom stopwords
            include_default: Whether to include default English stopwords
        """
        self.stopwords = set()
        if include_default:
            self.stopwords.update(self.ENGLISH_STOPWORDS)
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
    
    def remove(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with stopwords removed
        """
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        return ' '.join(filtered_words)
    
    def add_stopwords(self, words: List[str]) -> None:
        """Add additional stopwords to the set."""
        self.stopwords.update(words)
    
    def remove_stopwords(self, words: List[str]) -> None:
        """Remove words from the stopword set."""
        self.stopwords.difference_update(words)


class TextProcessor:
    """Main class for text processing operations."""
    
    def __init__(self, 
                 normalize_operations: List[Callable[[str], str]] = None,
                 tokenizer: Callable[[str], List[str]] = None,
                 stopword_remover: StopwordRemover = None,
                 custom_preprocessing: List[Callable[[str], str]] = None):
        """
        Initialize TextProcessor with configurable processing steps.
        
        Args:
            normalize_operations: List of normalization functions to apply
            tokenizer: Tokenization function to use
            stopword_remover: StopwordRemover instance for stopword removal
            custom_preprocessing: List of custom preprocessing functions
        """
        # Initialize with default operations if not provided
        self.normalize_operations = normalize_operations or [
            TextNormalizer.to_lowercase,
            TextNormalizer.remove_punctuation,
            TextNormalizer.remove_whitespace
        ]
        
        self.tokenizer = tokenizer or Tokenizer.tokenize_words
        self.stopword_remover = stopword_remover or StopwordRemover()
        self.custom_preprocessing = custom_preprocessing or []
    
    def normalize(self, text: str) -> str:
        """
        Apply all normalization operations to text.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        result = text
        for operation in self.normalize_operations:
            result = operation(result)
        return result
    
    def process_text(self, text: str) -> str:
        """
        Apply all preprocessing steps to text.
        
        Args:
            text: Input text
            
        Returns:
            Processed text
        """
        # Apply normalization
        processed_text = self.normalize(text)
        
        # Apply stopword removal
        processed_text = self.stopword_remover.remove(processed_text)
        
        # Apply custom preprocessing
        for step in self.custom_preprocessing:
            processed_text = step(processed_text)
        
        return processed_text
    
    def preprocess_documents(self, documents: List[str]) -> List[str]:
        """
        Preprocess a list of documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of processed document texts
        """
        return [self.preprocess(doc) for doc in documents]
    
    def vectorize_tfidf(self, 
                       texts: List[str], 
                       preprocess: bool = True,
                       min_df: Union[int, float] = 1,
                       max_df: Union[int, float] = 1.0,
                       ngram_range: Tuple[int, int] = (1, 1),
                       max_features: Optional[int] = None) -> Tuple[np.ndarray, TfidfVectorizer]:
        """
        Convert a list of texts into TF-IDF vectors.
        
        Args:
            texts: List of document texts
            preprocess: Whether to preprocess texts before vectorization
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            ngram_range: Range of n-grams to include
            max_features: Maximum number of features
            
        Returns:
            Tuple of document vectors array and vectorizer
        """
        processed_texts = self.preprocess_documents(texts) if preprocess else texts
        
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            max_features=max_features
        )
        
        vectors = vectorizer.fit_transform(processed_texts)
        return vectors.toarray(), vectorizer
    
    def vectorize_count(self, 
                       texts: List[str], 
                       preprocess: bool = True,
                       min_df: Union[int, float] = 1,
                       max_df: Union[int, float] = 1.0,
                       ngram_range: Tuple[int, int] = (1, 1),
                       max_features: Optional[int] = None) -> Tuple[np.ndarray, CountVectorizer]:
        """
        Convert a list of texts into count vectors.
        
        Args:
            texts: List of document texts
            preprocess: Whether to preprocess texts before vectorization
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            ngram_range: Range of n-grams to include
            max_features: Maximum number of features
            
        Returns:
            Tuple of document vectors array and vectorizer
        """
        processed_texts = self.preprocess_documents(texts) if preprocess else texts
        
        vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            max_features=max_features
        )
        
        vectors = vectorizer.fit_transform(processed_texts)
        return vectors.toarray(), vectorizer
    
    def get_document_stats(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about a document.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of document statistics
        """
        # Tokenize text into words and sentences
        words = Tokenizer.tokenize_words(text)
        sentences = Tokenizer.tokenize_sentences(text)
        
        # Calculate word frequencies
        word_freq = Counter(words)
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'unique_words': len(word_freq),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'most_common_words': word_freq.most_common(10)
        }
