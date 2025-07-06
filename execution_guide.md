# Plagiarism Checker Execution Guide with Database Integration

This guide provides detailed, step-by-step instructions for setting up and executing the enhanced Plagiarism Checker with SQLAlchemy database integration.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Database Setup](#database-setup)
4. [Executing the CLI Application](#executing-the-cli-application)
5. [Running the API Server](#running-the-api-server)
6. [Using the API](#using-the-api)
7. [Working with Stored Results](#working-with-stored-results)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

Before getting started, ensure you have the following:

- Python 3.6 or higher installed
- pip (Python package manager)
- Basic knowledge of command line interfaces
- Understanding of REST APIs (for API usage)

## Installation

### Step 1: Clone or Download the Project

Download the plagiarism checker project to your local machine, or clone it from a repository:

```bash
# Example if using git
git clone https://github.com/your-repository/plagiarism_checker_new.git
cd plagiarism_checker_new
```

### Step 2: Set Up a Virtual Environment (Recommended)

Create and activate a virtual environment to isolate dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

Install all required packages:

```bash
pip install sqlalchemy flask numpy scipy scikit-learn matplotlib tabulate colorama seaborn pandas
```

## Database Setup

The application uses SQLite through SQLAlchemy for persistent storage of similarity results.

### Step 1: Verify Database Files Are Present

Ensure the following files exist in your project:

- `models.py` - Contains SQLAlchemy model definitions
- `db.py` - Database connection and utility functions

### Step 2: Initialize the Database (Manual Method)

While the application initializes the database automatically when started, you can also do it manually:

```bash
python -c "from db import init_db; init_db()"
```

This will create a `results.db` SQLite file in the project root directory.

## Executing the CLI Application

The command-line interface (CLI) allows you to check plagiarism among text files in a directory.

### Basic Usage

To check plagiarism in text files from a specific directory:

```bash
python main.py --dir ./files --ext .txt
```

This will:
1. Initialize the database (if not already created)
2. Load all text files from the specified directory
3. Process the documents and calculate similarity
4. Store the results in the database
5. Display the similarity matrix in the console

### Advanced CLI Options

The CLI supports various options to customize the plagiarism checking:

```bash
python main.py --dir ./files --ext .txt \
  --normalize \
  --remove-stopwords \
  --ngram 2 \
  --similarity jaccard \
  --output results.json \
  --output-format json
```

Key parameters:

- `--normalize`: Apply text normalization (lowercase, etc.)
- `--remove-stopwords`: Remove common words (the, a, an, etc.)
- `--ngram N`: Use N-grams for comparison (N=1 for words, N>1 for character N-grams)
- `--similarity METRIC`: Choose similarity metric (cosine, jaccard, euclidean, etc.)
- `--compare-all`: Use all available similarity metrics
- `--output FILE`: Save results to specified file
- `--output-format FORMAT`: Format for output file (json, csv, html, markdown, latex)
- `--no-store-db`: Skip storing results in the database

### Generating Visualizations

To create visualizations of the similarity results:

```bash
python main.py --dir ./files --ext .txt \
  --image similarity.png \
  --image-format heatmap
```

Available visualization formats:
- `heatmap`: Color-coded matrix of similarity scores
- `network`: Network graph showing document relationships
- `bar`: Bar chart of similarity scores
- `cluster`: Hierarchical clustering dendrogram
- `mds`: Multidimensional scaling plot

### Generating Comprehensive Reports

For detailed reports with multiple visualizations:

```bash
python main.py --dir ./files --ext .txt \
  --output-dir ./report
```

This will create a report directory with multiple file formats and visualizations.

## Running the API Server

The application also provides a REST API for integrating with other systems.

### Starting the API Server

To run the API server:

```bash
python api.py
```

By default, the server runs on `localhost:5000`. You can modify these settings using environment variables:

```bash
# Change port
PORT=8000 python api.py

# Allow external connections (not recommended for production)
HOST=0.0.0.0 python api.py

# Disable debug mode for production
FLASK_ENV=production python api.py
```

The API server automatically initializes the database when started.

## Using the API

Interact with the API using HTTP requests. Here are some examples using curl:

### 1. Check API Status

```bash
curl http://localhost:5000/
```

Response:
```json
{
  "status": "success",
  "message": "Plagiarism Checker API is active",
  "version": "1.1.0"
}
```

### 2. Get Detailed API Status

```bash
curl http://localhost:5000/status
```

Response:
```json
{
  "status": "success",
  "document_count": 0,
  "cached_comparisons": 0,
  "document_ids": [],
  "api_version": "1.1.0",
  "available_metrics": ["cosine", "jaccard", "euclidean", "manhattan", "dice", "overlap", "levenshtein"]
}
```

### 3. Submit Documents

```bash
curl -X POST http://localhost:5000/submit \
     -H "Content-Type: application/json" \
     -d '{"doc_id": "doc1", "text": "This is a sample document for testing plagiarism detection."}'
```

```bash
curl -X POST http://localhost:5000/submit \
     -H "Content-Type: application/json" \
     -d '{"doc_id": "doc2", "text": "This is another document for testing the plagiarism system."}'
```

### 4. Compare Documents

```bash
curl -X POST http://localhost:5000/compare \
     -H "Content-Type: application/json" \
     -d '{"doc_id_1": "doc1", "doc_id_2": "doc2"}'
```

Response:
```json
{
  "status": "success",
  "doc_id_1": "doc1",
  "doc_id_2": "doc2",
  "metric": "cosine",
  "similarity_score": 0.75
}
```

To use a different similarity metric:

```bash
curl -X POST http://localhost:5000/compare \
     -H "Content-Type: application/json" \
     -d '{"doc_id_1": "doc1", "doc_id_2": "doc2", "metric": "jaccard"}'
```

### 5. Batch Compare Documents

To compare multiple document pairs at once:

```bash
curl -X POST http://localhost:5000/compare-batch \
     -H "Content-Type: application/json" \
     -d '{
       "pairs": [
         {"doc_id_1": "doc1", "doc_id_2": "doc2"},
         {"doc_id_1": "doc1", "doc_id_2": "doc3"},
         {"doc_id_1": "doc2", "doc_id_2": "doc3"}
       ],
       "metric": "cosine"
     }'
```

### 6. Get Results

To retrieve all comparison results:

```bash
curl http://localhost:5000/results
```

To get results in matrix format:

```bash
curl "http://localhost:5000/results?format=matrix"
```

To filter by a specific metric:

```bash
curl "http://localhost:5000/results?metric=cosine"
```

To filter by similarity threshold:

```bash
curl "http://localhost:5000/results?threshold=0.7"
```

## Working with Stored Results

The application stores all similarity results in a SQLite database (`results.db`).

### Accessing the Database Directly

You can query the database using the SQLite command-line tool:

```bash
sqlite3 results.db
```

Common SQLite commands:

```sql
-- List all tables
.tables

-- Show table schema
.schema similarity_results

-- Count all results
SELECT COUNT(*) FROM similarity_results;

-- Get all results
SELECT * FROM similarity_results;

-- Get results for a specific document
SELECT * FROM similarity_results 
WHERE doc1 = 'sample1' OR doc2 = 'sample1';

-- Get results above a certain threshold
SELECT * FROM similarity_results
WHERE score > 0.7
ORDER BY score DESC;
```

### Using Python to Query the Database

You can also create Python scripts to query the database using SQLAlchemy:

```python
from db import get_db
from models import SimilarityResult

# Get a database session
with get_db() as session:
    # Get all results
    results = session.query(SimilarityResult).all()
    
    # Print results
    for result in results:
        print(f"{result.doc1} vs {result.doc2}: {result.score:.2f} ({result.timestamp})")
        
    # Filter by threshold
    high_similarity = session.query(SimilarityResult).filter(
        SimilarityResult.score >= 0.8
    ).order_by(SimilarityResult.score.desc()).all()
    
    print(f"\nHigh similarity matches ({len(high_similarity)}):")
    for result in high_similarity:
        print(f"{result.doc1} vs {result.doc2}: {result.score:.2f}")
```

## Advanced Usage

### Creating Reports from Database Results

You can script generation of reports based on database results:

```python
import matplotlib.pyplot as plt
import pandas as pd
from db import get_db
from models import SimilarityResult

# Get database results
with get_db() as session:
    results = session.query(SimilarityResult).all()
    
    # Convert to DataFrame
    data = []
    for result in results:
        data.append({
            'doc1': result.doc1,
            'doc2': result.doc2,
            'score': result.score,
            'timestamp': result.timestamp
        })
    
    df = pd.DataFrame(data)
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    plt.hist(df['score'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.savefig('similarity_distribution.png')
    
    # Generate summary statistics
    print("Summary Statistics:")
    print(df['score'].describe())
    
    # Find potential plagiarism cases
    high_scores = df[df['score'] > 0.8]
    if not high_scores.empty:
        print("\nPotential Plagiarism Cases:")
        for _, row in high_scores.iterrows():
            print(f"{row['doc1']} vs {row['doc2']}: {row['score']:.2f}")
```

### Performing Trend Analysis

If you run plagiarism checks over time, you can analyze trends:

```python
import matplotlib.pyplot as plt
import pandas as pd
from db import get_db
from models import SimilarityResult
from sqlalchemy import func

# Get data by date
with get_db() as session:
    # Get average similarity score by day
    daily_scores = session.query(
        func.date(SimilarityResult.timestamp).label('date'),
        func.avg(SimilarityResult.score).label('avg_score'),
        func.count(SimilarityResult.id).label('comparison_count')
    ).group_by(func.date(SimilarityResult.timestamp)).all()
    
    # Convert to DataFrame
    df = pd.DataFrame(daily_scores, columns=['date', 'avg_score', 'comparison_count'])
    
    # Plot trends
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['avg_score'], marker='o', linestyle='-')
    plt.title('Average Similarity Score Trend')
    plt.xlabel('Date')
    plt.ylabel('Average Similarity Score')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('similarity_trend.png')
```

## Troubleshooting

### Common Issues and Solutions

#### Database Errors

1. **"No such table" error**:
   - Ensure the database is initialized by calling `init_db()`
   - Check that the database file `results.db` exists in the project root

2. **"Database is locked" error**:
   - Ensure you're not accessing the database from multiple processes simultaneously
   - Close any database connections or tools (like SQLite browser) that might be using the file

3. **Permission issues**:
   - Verify you have write permissions in the project directory
   - Run the application with appropriate permissions

#### API Server Issues

1. **Server won't start**:
   - Check if another process is already using the specified port
   - Ensure all required packages are installed
   - Check for syntax errors in the code

2. **"Connection refused" in client requests**:
   - Make sure the API server is running
   - Verify the host and port settings
   - Check for firewalls blocking the connection

#### CLI Application Issues

1. **No files found in directory**:
   - Check that the directory path is correct
   - Ensure files have the specified extension
   - Verify read permissions for the files

2. **Memory errors with large files**:
   - Process files in smaller batches
   - Increase available memory
   - Use file streaming approaches for very large files

### Getting Help

If you encounter issues not covered here:

1. Check the application logs for detailed error information
2. Refer to the documentation for specific components
3. Search for similar issues in the project's issue tracker
4. Contact the project maintainers for assistance

## Conclusion

This guide covers the basic and advanced usage of the Plagiarism Checker with database integration. The application provides a robust solution for detecting similarities between documents, with persistent storage of results for analysis and reporting.
