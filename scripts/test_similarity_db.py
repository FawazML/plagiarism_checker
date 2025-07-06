# test_similarity_db.py - Test the database integration in the similarity module
from plagcheck.db.db import init_db, get_db
from plagcheck.modules.similarity import DocumentSimilarity, SimilarityMetric
from plagcheck.db.models import SimilarityResult
import datetime

def test_similarity_db_integration():
    """Test the database integration in the similarity module."""
    print("Testing similarity module database integration...")
    
    # Initialize the database
    init_db()
    print("Database initialized.")
    
    # Create test documents
    doc1_id = "test_doc1"
    doc2_id = "test_doc2"
    doc1_text = "This is a sample document for testing similarity computation with database integration."
    doc2_text = "This document is similar to the sample and tests database integration in similarity computation."
    
    # Create similarity engine
    similarity_engine = DocumentSimilarity()
    print("Created similarity engine.")
    
    # Get database session
    session = get_db()
    try:
        # Compare texts with database integration
        print(f"Comparing documents: '{doc1_id}' and '{doc2_id}'...")
        similarity_score = similarity_engine.compare_two_texts(
            doc1_text, doc2_text,
            metric=SimilarityMetric.COSINE,
            preprocess=False,  # Disable preprocessing for simplicity
            doc_id1=doc1_id,
            doc_id2=doc2_id,
            db_session=session,
            store_in_db=True
        )
        
        # Commit the changes
        session.commit()
        print(f"Compared documents with cosine similarity: {similarity_score:.4f}")
        
        # Query the results
        results = session.query(SimilarityResult).all()
        print(f"Found {len(results)} results in the database:")
        for result in results:
            print(f"  {result.doc1} vs {result.doc2}: {result.score:.4f} ({result.timestamp})")
        
        # Try another metric
        print(f"\nComparing documents with a different metric...")
        jaccard_score = similarity_engine.compare_two_texts(
            text1=doc1_text,
            text2=doc2_text,
            metric=SimilarityMetric.JACCARD,
            preprocess=False,
            doc_id1=doc1_id,
            doc_id2=doc2_id,
            db_session=session,
            store_in_db=True
        )
        
        # Commit the changes
        session.commit()
        print(f"Compared documents with jaccard similarity: {jaccard_score:.4f}")
        
        # Query the results again
        results = session.query(SimilarityResult).all()
        print(f"Now found {len(results)} results in the database:")
        for result in results:
            print(f"  {result.doc1} vs {result.doc2}: {result.score:.4f} ({result.timestamp})")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        session.rollback()
    finally:
        session.close()
    
    print("\nTest completed.")

if __name__ == "__main__":
    test_similarity_db_integration()