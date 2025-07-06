from plagcheck.db.db import init_db, get_db
from plagcheck.modules.similarity import DocumentSimilarity
from plagcheck.db.models import SimilarityResult

# Initialize
init_db()
similarity_engine = DocumentSimilarity()

# Get texts to compare
text1 = "This is the first document for comparison."
text2 = "This is another document with some similarities."

# Get database session
session = get_db()
try:
    # Compare with database integration
    score = similarity_engine.compare_two_texts(
        text1, text2,
        doc_id1="custom_doc1",
        doc_id2="custom_doc2",
        db_session=session,
        store_in_db=True
    )
    
    # Commit the changes
    session.commit()
    print(f"Similarity score: {score}")
    
    # Query the results
    results = session.query(SimilarityResult).all()
    for result in results:
        print(f"{result.doc1} vs {result.doc2}: {result.score} ({result.timestamp})")
finally:
    session.close()