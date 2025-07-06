"""
Database components for the plagiarism checker package.

This module provides database connectivity and models for storing similarity results.
"""

from plagcheck.db.models import SimilarityResult
from plagcheck.db.db import init_db, get_db, shutdown_db
