# models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Create the SQLAlchemy Base
Base = declarative_base()

class SimilarityResult(Base):
    """
    SQLAlchemy model to store similarity comparison results between documents.
    
    Attributes:
        id (int): Primary key for the result
        doc1 (str): Identifier for the first document
        doc2 (str): Identifier for the second document
        score (float): Similarity score between the two documents
        timestamp (datetime): When the comparison was performed
    """
    __tablename__ = 'similarity_results'
    
    id = Column(Integer, primary_key=True)
    doc1 = Column(String(255), nullable=False)
    doc2 = Column(String(255), nullable=False)
    score = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<SimilarityResult(id={self.id}, doc1='{self.doc1}', doc2='{self.doc2}', score={self.score})>"


# Helper functions to initialize the database and create a session
def init_db(db_url='sqlite:///similarity_results.db'):
    """
    Initialize the database with the defined models.
    
    Args:
        db_url (str): Database URL to connect to
        
    Returns:
        engine: SQLAlchemy engine
    """
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return engine

def get_session(engine):
    """
    Create a new session using the provided engine.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        session: SQLAlchemy session
    """
    Session = sessionmaker(bind=engine)
    return Session()
