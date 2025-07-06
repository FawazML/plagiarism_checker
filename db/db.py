# db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from plagcheck.db.models import Base
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure SQLAlchemy database URL
DATABASE_URL = "sqlite:///results.db"

# Create the SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=False  # Set to True to see SQL queries for debugging
)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a thread-local session for handling multiple requests
db_session = scoped_session(SessionLocal)

def init_db():
    """
    Initialize the database by creating all tables defined in Base.
    This function should be called when the application starts.
    """
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully!")

def get_db():
    """
    Get a database session. Ensure you either commit or rollback and close the session when done.
    
    Returns:
        SQLAlchemy session for database operations
    
    Usage:
        db = get_db()
        try:
            # Use db for database operations
            ...
            db.commit()
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    """
    return db_session()

class DatabaseSession:
    """
    Context manager for handling database sessions.
    Automatically commits on successful execution and rolls back on exceptions.
    
    Usage:
        with DatabaseSession() as db:
            # Use db for database operations
            ...
    """
    def __enter__(self):
        self.db = db_session()
        return self.db
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                # No exception occurred, commit changes
                logger.info("Committing database changes")
                self.db.commit()
            else:
                # An exception occurred, rollback changes
                logger.error(f"Rolling back due to error: {exc_val}")
                self.db.rollback()
        except Exception as e:
            logger.error(f"Error in database session: {e}")
            self.db.rollback()
            raise
        finally:
            self.db.close()

# Function to dispose of engine when the application shuts down
def shutdown_db():
    """Call this function when shutting down the application."""
    db_session.remove()
    engine.dispose()