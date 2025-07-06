# similarity.py
import numpy as np
import scipy.spatial.distance as dist
from typing import List, Tuple, Set, Dict, Union, Optional, Callable, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from enum import Enum
import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import will be skipped if the module is used standalone without database
try:
    from plagcheck.db.models import SimilarityResult
except ImportError:
    logger.warning("models module not found; database functionality will be disabled")
    SimilarityResult = None

class SimilarityMetric(Enum):
    """Enumeration of supported similarity metrics."""
    COSINE = "cosine"
    JACCARD = "jaccard"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    DICE = "dice"
    OVERLAP = "overlap"
    LEVENSHTEIN = "levenshtein"
    
class SimilarityCalculator:
    """
    Class providing various methods to calculate similarity between document vectors.
    Supports multiple similarity metrics for different use cases.
    """
    
    def __init__(self, metric: Union[str, SimilarityMetric] = SimilarityMetric.COSINE):
        """
        Initialize the SimilarityCalculator with the specified metric.
        
        Args:
            metric: The similarity metric to use (default: cosine)
        """
        if isinstance(metric, str):
            try:
                self.metric = SimilarityMetric(metric.lower())
            except ValueError:
                raise ValueError(f"Unsupported similarity metric: {metric}. "
                                 f"Supported metrics: {[m.value for m in SimilarityMetric]}")
        else:
            self.metric = metric
        
        # Map metrics to their corresponding functions
        self.similarity_functions = {
            SimilarityMetric.COSINE: self._cosine_similarity,
            SimilarityMetric.JACCARD: self._jaccard_similarity,
            SimilarityMetric.EUCLIDEAN: self._euclidean_similarity,
            SimilarityMetric.MANHATTAN: self._manhattan_similarity,
            SimilarityMetric.DICE: self._dice_similarity,
            SimilarityMetric.OVERLAP: self._overlap_similarity,
            SimilarityMetric.LEVENSHTEIN: self._get_levenshtein_similarity
        }
    
    @staticmethod
    def _cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate the cosine similarity between two vectors.
        
        Cosine similarity measures the cosine of the angle between two vectors,
        providing a measure of similarity that is independent of vector magnitude.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Use sklearn's implementation which handles edge cases well
        return cosine_similarity([vector1, vector2])[0][1]
    
    @staticmethod
    def _jaccard_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate the Jaccard similarity between two vectors.
        
        Jaccard similarity is the size of the intersection divided by the size of the union
        of two sets. It's useful for comparing sparse binary vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Jaccard similarity score between 0 and 1
        """
        # For binary vectors, convert to sets
        if np.all(np.logical_or(vector1 == 0, vector1 == 1)) and np.all(np.logical_or(vector2 == 0, vector2 == 1)):
            set1 = set(np.where(vector1 > 0)[0])
            set2 = set(np.where(vector2 > 0)[0])
            
            if not set1 and not set2:  # Both vectors are all zeros
                return 1.0
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union
        else:
            # For non-binary vectors, use weighted Jaccard
            min_sum = np.sum(np.minimum(vector1, vector2))
            max_sum = np.sum(np.maximum(vector1, vector2))
            
            if max_sum == 0:  # Both vectors are all zeros
                return 1.0
                
            return min_sum / max_sum
    
    @staticmethod
    def _euclidean_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate the Euclidean similarity between two vectors.
        
        This transforms Euclidean distance into a similarity measure between 0 and 1,
        where 1 means identical vectors and values approach 0 as differences increase.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Euclidean similarity score between 0 and 1
        """
        distance = np.linalg.norm(vector1 - vector2)
        # Convert distance to similarity (1 for identical vectors, approaches 0 as distance increases)
        return 1.0 / (1.0 + distance)
    
    @staticmethod
    def _manhattan_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate the Manhattan (L1) similarity between two vectors.
        
        This transforms Manhattan distance into a similarity measure between 0 and 1,
        where 1 means identical vectors and values approach 0 as differences increase.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Manhattan similarity score between 0 and 1
        """
        distance = np.sum(np.abs(vector1 - vector2))
        # Convert distance to similarity (1 for identical vectors, approaches 0 as distance increases)
        return 1.0 / (1.0 + distance)
    
    @staticmethod
    def _dice_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate the Dice coefficient between two vectors.
        
        The Dice coefficient is defined as 2*|X∩Y|/(|X|+|Y|), where X and Y are the
        two sets. It's commonly used in information retrieval.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Dice similarity score between 0 and 1
        """
        # For binary vectors, convert to sets
        if np.all(np.logical_or(vector1 == 0, vector1 == 1)) and np.all(np.logical_or(vector2 == 0, vector2 == 1)):
            set1 = set(np.where(vector1 > 0)[0])
            set2 = set(np.where(vector2 > 0)[0])
            
            if not set1 and not set2:  # Both vectors are all zeros
                return 1.0
            
            intersection = len(set1.intersection(set2))
            return (2.0 * intersection) / (len(set1) + len(set2))
        else:
            # For non-binary vectors, use weighted version
            intersection = np.sum(np.minimum(vector1, vector2))
            return (2.0 * intersection) / (np.sum(vector1) + np.sum(vector2))
    
    @staticmethod
    def _overlap_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate the overlap coefficient between two vectors.
        
        The overlap coefficient is defined as |X∩Y|/min(|X|,|Y|), measuring the overlap
        between two sets. It equals 1 if one set is a subset of the other.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Overlap coefficient between 0 and 1
        """
        # For binary vectors, convert to sets
        if np.all(np.logical_or(vector1 == 0, vector1 == 1)) and np.all(np.logical_or(vector2 == 0, vector2 == 1)):
            set1 = set(np.where(vector1 > 0)[0])
            set2 = set(np.where(vector2 > 0)[0])
            
            if not set1 and not set2:  # Both vectors are all zeros
                return 1.0
            
            intersection = len(set1.intersection(set2))
            min_size = min(len(set1), len(set2))
            
            if min_size == 0:
                return 0.0
                
            return intersection / min_size
        else:
            # For non-binary vectors, use weighted version
            intersection = np.sum(np.minimum(vector1, vector2))
            min_sum = min(np.sum(vector1), np.sum(vector2))
            
            if min_sum == 0:
                return 0.0
                
            return intersection / min_sum
    
    def _get_levenshtein_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate the Levenshtein similarity between two texts.
        
        This method ignores the vectors and uses the raw texts directly.
        It normalizes the Levenshtein distance to a similarity measure between 0 and 1.
        
        Args:
            vector1: First vector (ignored)
            vector2: Second vector (ignored)
            
        Returns:
            Normalized Levenshtein similarity between 0 and 1
        """
        if not hasattr(self, 'raw_text1') or not hasattr(self, 'raw_text2'):
            raise ValueError("Raw texts must be set with set_raw_texts() before using Levenshtein similarity")
        
        # Use Levenshtein distance on raw texts
        distance = self._levenshtein_distance(self.raw_text1, self.raw_text2)
        max_len = max(len(self.raw_text1), len(self.raw_text2))
        
        if max_len == 0:
            return 1.0
        
        # Normalize to get similarity (1 = identical, 0 = completely different)
        return 1.0 - (distance / max_len)
    
    def set_raw_texts(self, text1: str, text2: str) -> None:
        """
        Set the raw texts for text-based similarity metrics like Levenshtein.
        
        Args:
            text1: First raw text
            text2: Second raw text
        """
        self.raw_text1 = text1
        self.raw_text2 = text2
    
    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calculate the Levenshtein (edit) distance between two strings.
        
        The Levenshtein distance is the minimum number of single-character edits
        (insertions, deletions, or substitutions) required to change one string into another.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Levenshtein distance as an integer
        """
        if len(s1) < len(s2):
            return SimilarityCalculator._levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
    
    def calculate_similarity(self, 
                            vector1: np.ndarray, 
                            vector2: np.ndarray, 
                            text1: str = None, 
                            text2: str = None) -> float:
        """
        Calculate similarity between two vectors using the specified metric.
        
        Args:
            vector1: First vector
            vector2: Second vector
            text1: Optional raw text of first document (only used for text-based metrics)
            text2: Optional raw text of second document (only used for text-based metrics)
            
        Returns:
            Similarity score between 0 and 1
        """
        if self.metric == SimilarityMetric.LEVENSHTEIN and (text1 is not None and text2 is not None):
            self.set_raw_texts(text1, text2)
            
        # Get the similarity function for the specified metric
        similarity_func = self.similarity_functions.get(self.metric)
        
        if similarity_func is None:
            raise ValueError(f"Unsupported similarity metric: {self.metric}")
            
        return similarity_func(vector1, vector2)
    
    def calculate_similarity_matrix(self, 
                                  document_vectors: List[Tuple[str, np.ndarray]], 
                                  raw_texts: Dict[str, str] = None) -> Dict[Tuple[str, str], float]:
        """
        Calculate similarity between all pairs of document vectors.
        
        Args:
            document_vectors: List of (document_id, vector) tuples
            raw_texts: Optional dictionary mapping document IDs to raw texts (for text-based metrics)
            
        Returns:
            Dictionary mapping (doc_id1, doc_id2) tuples to similarity scores
        """
        results = {}
        n = len(document_vectors)
        
        for i in range(n):
            doc_id1, vector1 = document_vectors[i]
            for j in range(i + 1, n):
                doc_id2, vector2 = document_vectors[j]
                
                if self.metric == SimilarityMetric.LEVENSHTEIN and raw_texts is not None:
                    self.set_raw_texts(raw_texts[doc_id1], raw_texts[doc_id2])
                    similarity = self.calculate_similarity(vector1, vector2)
                else:
                    similarity = self.calculate_similarity(vector1, vector2)
                    
                results[(doc_id1, doc_id2)] = similarity
                
        return results
    
    def calculate_similarity_matrix_as_dict(self, 
                                         document_vectors: List[Tuple[str, np.ndarray]],
                                         raw_texts: Dict[str, str] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate similarity between all pairs of document vectors and return as a nested dictionary.
        
        Args:
            document_vectors: List of (document_id, vector) tuples
            raw_texts: Optional dictionary mapping document IDs to raw texts (for text-based metrics)
            
        Returns:
            Nested dictionary where outer keys are doc_ids and inner keys are other doc_ids with similarity scores as values
        """
        # Get flat similarity results
        flat_results = self.calculate_similarity_matrix(document_vectors, raw_texts)
        
        # Initialize nested dictionary
        nested_results = {doc_id: {} for doc_id, _ in document_vectors}
        
        # Fill in similarities (both directions)
        for (doc_id1, doc_id2), similarity in flat_results.items():
            nested_results[doc_id1][doc_id2] = similarity
            nested_results[doc_id2][doc_id1] = similarity
            
        # Set self-similarity to 1.0
        for doc_id in nested_results:
            nested_results[doc_id][doc_id] = 1.0
            
        return nested_results
    
    def calculate_similarity_matrix_as_array(self, 
                                          document_vectors: List[Tuple[str, np.ndarray]],
                                          raw_texts: Dict[str, str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate similarity between all pairs of document vectors and return as a 2D array.
        
        Args:
            document_vectors: List of (document_id, vector) tuples
            raw_texts: Optional dictionary mapping document IDs to raw texts (for text-based metrics)
            
        Returns:
            Tuple containing:
            - 2D numpy array of similarity scores
            - List of document IDs corresponding to rows/columns in the array
        """
        # Get nested dictionary results
        nested_results = self.calculate_similarity_matrix_as_dict(document_vectors, raw_texts)
        
        # Get document IDs in consistent order
        doc_ids = [doc_id for doc_id, _ in document_vectors]
        
        # Create 2D array
        n = len(doc_ids)
        similarity_matrix = np.zeros((n, n))
        
        for i, doc_id1 in enumerate(doc_ids):
            for j, doc_id2 in enumerate(doc_ids):
                similarity_matrix[i, j] = nested_results[doc_id1][doc_id2]
                
        return similarity_matrix, doc_ids

class DocumentSimilarity:
    """
    High-level class for calculating document similarity using different methods.
    Provides convenient methods for comparing documents directly with various metrics.
    """
    
    def __init__(self, preprocessor=None):
        """
        Initialize the DocumentSimilarity with an optional preprocessor.
        
        Args:
            preprocessor: Optional text preprocessor to apply before vectorization
        """
        self.preprocessor = preprocessor
        
    def compare_texts(self, 
                     texts: List[str], 
                     names: List[str] = None,
                     metric: Union[str, SimilarityMetric] = SimilarityMetric.COSINE,
                     vectorizer: str = "tfidf",
                     ngram_range: Tuple[int, int] = (1, 1),
                     binary: bool = False,
                     preprocess: bool = True,
                     db_session=None,
                     store_in_db: bool = False) -> Set[Tuple[str, str, float]]:
        """
        Compare a list of texts using the specified similarity metric.
        
        Args:
            texts: List of text documents to compare
            names: Optional list of document names (defaults to indices if not provided)
            metric: Similarity metric to use
            vectorizer: Vectorization method ("tfidf" or "count")
            ngram_range: Range of n-grams to include in the vectorization
            binary: Whether to use binary feature values
            preprocess: Whether to preprocess texts before vectorization
            db_session: Optional SQLAlchemy database session for storing results
            store_in_db: Whether to store results in the database
            
        Returns:
            Set of (doc_id_1, doc_id_2, similarity_score) tuples
        """
        if names is None:
            names = [f"doc_{i}" for i in range(len(texts))]
            
        # Preprocess texts if needed and preprocessor is available
        processed_texts = texts
        if preprocess and self.preprocessor:
            processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Create vectorizer
        if vectorizer.lower() == "tfidf":
            vec = TfidfVectorizer(ngram_range=ngram_range, binary=binary)
        elif vectorizer.lower() == "count":
            vec = CountVectorizer(ngram_range=ngram_range, binary=binary)
        else:
            raise ValueError(f"Unsupported vectorizer: {vectorizer}. Use 'tfidf' or 'count'.")
        
        # Vectorize texts
        vectors = vec.fit_transform(processed_texts).toarray()
        
        # Create document-vector pairs
        document_vectors = list(zip(names, vectors))
        
        # Calculate similarities
        calculator = SimilarityCalculator(metric=metric)
        
        # For Levenshtein, we need the raw texts
        if metric == SimilarityMetric.LEVENSHTEIN or metric == "levenshtein":
            raw_texts = dict(zip(names, texts))
            similarity_dict = calculator.calculate_similarity_matrix(document_vectors, raw_texts)
        else:
            similarity_dict = calculator.calculate_similarity_matrix(document_vectors)
            
        # Convert to result set
        results = set()
        for (doc1, doc2), score in similarity_dict.items():
            results.add((doc1, doc2, score))
            
            # Store in database if requested and session is provided
            if store_in_db and db_session is not None and SimilarityResult is not None:
                try:
                    # Check if result already exists
                    existing = db_session.query(SimilarityResult).filter(
                        ((SimilarityResult.doc1 == doc1) & (SimilarityResult.doc2 == doc2)) |
                        ((SimilarityResult.doc1 == doc2) & (SimilarityResult.doc2 == doc1))
                    ).first()
                    
                    if existing:
                        # Update existing record
                        existing.score = score
                        existing.timestamp = datetime.datetime.now()
                        logger.info(f"Updated database record for {doc1} and {doc2}")
                    else:
                        # Create new record
                        new_result = SimilarityResult(
                            doc1=doc1,
                            doc2=doc2,
                            score=score,
                            timestamp=datetime.datetime.now()
                        )
                        db_session.add(new_result)
                        logger.info(f"Added new similarity result to database for {doc1} and {doc2}")
                        
                    # Note: We don't commit here - that should be handled by the caller
                except Exception as e:
                    logger.error(f"Error storing similarity result in database: {e}")
            
        return results
    
    def compare_two_texts(self, 
                         text1: str, 
                         text2: str,
                         metric: Union[str, SimilarityMetric] = SimilarityMetric.COSINE,
                         vectorizer: str = "tfidf",
                         ngram_range: Tuple[int, int] = (1, 1),
                         binary: bool = False,
                         preprocess: bool = True,
                         doc_id1: str = None,
                         doc_id2: str = None,
                         db_session=None,
                         store_in_db: bool = False) -> float:
        """
        Compare two texts using the specified similarity metric.
        
        Args:
            text1: First text
            text2: Second text
            metric: Similarity metric to use
            vectorizer: Vectorization method ("tfidf" or "count")
            ngram_range: Range of n-grams to include in the vectorization
            binary: Whether to use binary feature values
            preprocess: Whether to preprocess texts before vectorization
            doc_id1: Optional identifier for the first document
            doc_id2: Optional identifier for the second document
            db_session: Optional SQLAlchemy database session for storing results
            store_in_db: Whether to store results in the database
            
        Returns:
            Similarity score between 0 and 1
        """
        # For Levenshtein, we can use it directly on the texts
        if metric == SimilarityMetric.LEVENSHTEIN or metric == "levenshtein":
            calculator = SimilarityCalculator(metric=metric)
            calculator.set_raw_texts(text1, text2)
            # The vectors don't matter for Levenshtein, just pass empty arrays
            similarity_score = calculator.calculate_similarity(np.array([]), np.array([]))
            
            # Store in database if requested
            if store_in_db and db_session is not None and doc_id1 and doc_id2 and SimilarityResult is not None:
                self._store_similarity_in_db(db_session, doc_id1, doc_id2, similarity_score)
                
            return similarity_score
            
        # Preprocess texts if needed and preprocessor is available
        if preprocess and self.preprocessor:
            try:
                text1 = self.preprocessor.preprocess(text1)
                text2 = self.preprocessor.preprocess(text2)
            except Exception as e:
                logger.warning(f"Error during text preprocessing: {e}. Using raw texts.")
        
        # Create vectorizer
        if vectorizer.lower() == "tfidf":
            vec = TfidfVectorizer(ngram_range=ngram_range, binary=binary)
        elif vectorizer.lower() == "count":
            vec = CountVectorizer(ngram_range=ngram_range, binary=binary)
        else:
            raise ValueError(f"Unsupported vectorizer: {vectorizer}. Use 'tfidf' or 'count'.")
        
        # Vectorize texts
        try:
            vectors = vec.fit_transform([text1, text2]).toarray()
        except Exception as e:
            logger.error(f"Error vectorizing texts: {e}")
            raise
        
        # Calculate similarity
        calculator = SimilarityCalculator(metric=metric)
        similarity_score = calculator.calculate_similarity(vectors[0], vectors[1], text1, text2)
        
        # Store in database if requested
        if store_in_db and db_session is not None and doc_id1 and doc_id2 and SimilarityResult is not None:
            self._store_similarity_in_db(db_session, doc_id1, doc_id2, similarity_score)
            
        return similarity_score
    
    def _store_similarity_in_db(self, db_session, doc_id1: str, doc_id2: str, score: float) -> None:
        """
        Store a similarity result in the database.
        
        Args:
            db_session: SQLAlchemy database session
            doc_id1: First document ID
            doc_id2: Second document ID
            score: Similarity score
        """
        if SimilarityResult is None:
            logger.warning("SimilarityResult model not available. Skipping database storage.")
            return
            
        try:
            # Sort document IDs for consistent storage
            sorted_ids = sorted([doc_id1, doc_id2])
            doc1, doc2 = sorted_ids
            
            # Check if result already exists
            existing = db_session.query(SimilarityResult).filter(
                ((SimilarityResult.doc1 == doc1) & (SimilarityResult.doc2 == doc2)) |
                ((SimilarityResult.doc1 == doc2) & (SimilarityResult.doc2 == doc1))
            ).first()
            
            if existing:
                # Update existing record
                existing.score = score
                existing.timestamp = datetime.datetime.now()
                logger.info(f"Updated database record for {doc1} and {doc2}")
            else:
                # Create new record
                new_result = SimilarityResult(
                    doc1=doc1,
                    doc2=doc2,
                    score=score,
                    timestamp=datetime.datetime.now()
                )
                db_session.add(new_result)
                logger.info(f"Added new similarity result to database for {doc1} and {doc2}")
                
            # Note: We don't commit here - that should be handled by the caller
        except Exception as e:
            logger.error(f"Error storing similarity result in database: {e}")
    
    def compare_texts_as_dict(self, 
                             texts: List[str], 
                             names: List[str] = None,
                             metric: Union[str, SimilarityMetric] = SimilarityMetric.COSINE,
                             vectorizer: str = "tfidf",
                             ngram_range: Tuple[int, int] = (1, 1),
                             binary: bool = False,
                             preprocess: bool = True,
                             db_session=None,
                             store_in_db: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Compare a list of texts and return results as a nested dictionary.
        
        Args:
            texts: List of text documents to compare
            names: Optional list of document names (defaults to indices if not provided)
            metric: Similarity metric to use
            vectorizer: Vectorization method ("tfidf" or "count")
            ngram_range: Range of n-grams to include in the vectorization
            binary: Whether to use binary feature values
            preprocess: Whether to preprocess texts before vectorization
            db_session: Optional SQLAlchemy database session for storing results
            store_in_db: Whether to store results in the database
            
        Returns:
            Dictionary where keys are document names and values are dictionaries mapping 
            other document names to similarity scores
        """
        if names is None:
            names = [f"doc_{i}" for i in range(len(texts))]
            
        # Preprocess texts if needed and preprocessor is available
        processed_texts = texts
        if preprocess and self.preprocessor:
            try:
                processed_texts = [self.preprocessor.preprocess(text) for text in texts]
            except Exception as e:
                logger.warning(f"Error during text preprocessing: {e}. Using raw texts.")
                processed_texts = texts
        
        # Create vectorizer
        if vectorizer.lower() == "tfidf":
            vec = TfidfVectorizer(ngram_range=ngram_range, binary=binary)
        elif vectorizer.lower() == "count":
            vec = CountVectorizer(ngram_range=ngram_range, binary=binary)
        else:
            raise ValueError(f"Unsupported vectorizer: {vectorizer}. Use 'tfidf' or 'count'.")
        
        # Vectorize texts
        try:
            vectors = vec.fit_transform(processed_texts).toarray()
        except Exception as e:
            logger.error(f"Error vectorizing texts: {e}")
            raise
        
        # Create document-vector pairs
        document_vectors = list(zip(names, vectors))
        
        # Calculate similarities
        calculator = SimilarityCalculator(metric=metric)
        
        # For Levenshtein, we need the raw texts
        if metric == SimilarityMetric.LEVENSHTEIN or metric == "levenshtein":
            raw_texts = dict(zip(names, texts))
            result_dict = calculator.calculate_similarity_matrix_as_dict(document_vectors, raw_texts)
        else:
            result_dict = calculator.calculate_similarity_matrix_as_dict(document_vectors)
        
        # Store in database if requested
        if store_in_db and db_session is not None and SimilarityResult is not None:
            for doc1, similarities in result_dict.items():
                for doc2, score in similarities.items():
                    # Skip self-comparisons
                    if doc1 != doc2:
                        # Store only once per pair (we'll sort to ensure consistent order)
                        if doc1 < doc2:
                            self._store_similarity_in_db(db_session, doc1, doc2, score)
            
        return result_dict
    
    def compare_texts_as_array(self, 
                              texts: List[str], 
                              names: List[str] = None,
                              metric: Union[str, SimilarityMetric] = SimilarityMetric.COSINE,
                              vectorizer: str = "tfidf",
                              ngram_range: Tuple[int, int] = (1, 1),
                              binary: bool = False,
                              preprocess: bool = True,
                              db_session=None,
                              store_in_db: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Compare a list of texts and return results as a numpy array.
        
        Args:
            texts: List of text documents to compare
            names: Optional list of document names (defaults to indices if not provided)
            metric: Similarity metric to use
            vectorizer: Vectorization method ("tfidf" or "count")
            ngram_range: Range of n-grams to include in the vectorization
            binary: Whether to use binary feature values
            preprocess: Whether to preprocess texts before vectorization
            db_session: Optional SQLAlchemy database session for storing results
            store_in_db: Whether to store results in the database
            
        Returns:
            Tuple containing:
            - 2D numpy array of similarity scores
            - List of document names corresponding to rows/columns in the array
        """
        # Get results as dict first (will handle database storage if needed)
        result_dict = self.compare_texts_as_dict(
            texts, names, metric, vectorizer, ngram_range, binary, preprocess,
            db_session, store_in_db
        )
        
        # Get document names in a consistent order
        doc_names = list(result_dict.keys())
        
        # Convert to array
        n = len(doc_names)
        similarity_matrix = np.zeros((n, n))
        
        for i, doc1 in enumerate(doc_names):
            for j, doc2 in enumerate(doc_names):
                similarity_matrix[i, j] = result_dict[doc1][doc2]
        
        return similarity_matrix, doc_names