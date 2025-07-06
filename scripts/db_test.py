# db_test.py - A script to test database functionality
from db import init_db, get_db
from plagcheck.db.models import SimilarityResult
import datetime

def test_database():
    """Test database functionality by adding and retrieving a record."""
    print("Testing database functionality...")
    
    # Initialize the database
    init_db()
    print("Database initialized.")
    
    # Create a test record
    test_record = SimilarityResult(
        doc1="test_doc1",
        doc2="test_doc2",
        score=0.75,
        timestamp=datetime.datetime.now()
    )
    
    # Get a database session
    session = get_db()
    try:
        # Add the test record
        session.add(test_record)
        # Important: Explicitly commit the changes
        session.commit()
        print("Test record added to database with explicit commit.")
        
        # Verify the record was saved by querying with a new session
        new_session = get_db()
        try:
            records = new_session.query(SimilarityResult).all()
            print(f"Found {len(records)} records in database.")
            
            # Print each record
            for record in records:
                print(f"Record: {record.id}, {record.doc1} vs {record.doc2}, Score: {record.score}, Time: {record.timestamp}")
        finally:
            new_session.close()
    finally:
        session.close()
    
    print("Database test completed.")

if __name__ == "__main__":
    test_database()