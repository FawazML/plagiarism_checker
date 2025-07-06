# Plagiarism Checker API: Comprehensive Execution Guide

This guide provides detailed, step-by-step instructions for setting up, running, and using the Plagiarism Checker API with all validation features.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Starting the API Server](#starting-the-api-server)
4. [API Validation Features](#api-validation-features)
5. [Using the API Endpoints](#using-the-api-endpoints)
6. [Data Visualization](#data-visualization)
7. [Client Applications](#client-applications)
8. [Error Handling](#error-handling)
9. [Performance and Security Considerations](#performance-and-security-considerations)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

Before starting, ensure you have:
- Python 3.6 or higher installed
- pip (Python package manager)
- Basic knowledge of RESTful APIs
- Terminal/command line access

## Installation

1. **Set up a virtual environment** (recommended):
   ```bash
   # Create a virtual environment
   python -m venv venv
   
   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install required dependencies**:
   ```bash
   pip install flask numpy scipy scikit-learn matplotlib tabulate seaborn
   ```

3. **Verify installation**:
   ```bash
   python -c "import flask, numpy, scipy, sklearn; print('All required packages installed successfully')"
   ```

## Starting the API Server

1. **Navigate to the project directory**:
   ```bash
   cd path/to/plagiarism_checker_new
   ```

2. **Start the API server**:
   ```bash
   python api.py
   ```

3. **Verify the server is running successfully**. You should see output similar to:
   ```
   * Serving Flask app "api" (lazy loading)
   * Environment: production
     WARNING: This is a development server. Do not use it in a production deployment.
   * Debug mode: on
   * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
   ```

4. **Test the API is running** by opening a new terminal window and running:
   ```bash
   curl http://localhost:5000/
   ```
   
   You should receive a response like:
   ```json
   {
     "status": "success",
     "message": "Plagiarism Checker API is active",
     "version": "1.0.0"
   }
   ```

## API Validation Features

The API implements comprehensive validation for all inputs:

### Document ID Validation:
- Must be a string
- Cannot be empty
- Can only contain letters, numbers, underscores, and hyphens
- Maximum length of 64 characters

### Text Content Validation:
- Must be a string
- Cannot be empty
- Maximum size of 1MB

### Metric Validation:
- Must be one of the supported metrics:
  - cosine
  - jaccard
  - euclidean
  - manhattan
  - dice
  - overlap
  - levenshtein

### Request Format Validation:
- Requests must have the correct content type
- JSON payload must be properly formatted
- Required fields must be present

## Using the API Endpoints

### 1. Check API Status

Command:
```bash
curl http://localhost:5000/status
```

Expected response:
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

### 2. Submit Documents

**First document:**
```bash
curl -X POST http://localhost:5000/submit \
     -H "Content-Type: application/json" \
     -d '{"doc_id": "sample1", "text": "This is a sample document for the plagiarism checker. It contains unique text that will be processed and analyzed by the system."}'
```

**Second document with some similarities:**
```bash
curl -X POST http://localhost:5000/submit \
     -H "Content-Type: application/json" \
     -d '{"doc_id": "sample2", "text": "This is another document for plagiarism checking. It also contains text that will be processed and analyzed, with some unique content."}'
```

**Third document with different content:**
```bash
curl -X POST http://localhost:5000/submit \
     -H "Content-Type: application/json" \
     -d '{"doc_id": "sample3", "text": "This document has completely different content about natural language processing, machine learning, and data science algorithms."}'
```

### 3. Compare Documents

**Compare with default cosine similarity:**
```bash
curl -X POST http://localhost:5000/compare \
     -H "Content-Type: application/json" \
     -d '{"doc_id_1": "sample1", "doc_id_2": "sample2"}'
```

**Compare using a different similarity metric:**
```bash
curl -X POST http://localhost:5000/compare \
     -H "Content-Type: application/json" \
     -d '{"doc_id_1": "sample1", "doc_id_2": "sample2", "metric": "jaccard"}'
```

**Compare another document pair:**
```bash
curl -X POST http://localhost:5000/compare \
     -H "Content-Type: application/json" \
     -d '{"doc_id_1": "sample1", "doc_id_2": "sample3"}'
```

### 4. View Results

**Get all results in pairs format:**
```bash
curl http://localhost:5000/results
```

**Get results in matrix format:**
```bash
curl "http://localhost:5000/results?format=matrix"
```

**Filter results by metric:**
```bash
curl "http://localhost:5000/results?metric=cosine"
```

**Get matrix results for a specific metric:**
```bash
curl "http://localhost:5000/results?format=matrix&metric=jaccard"
```

## Data Visualization

### Python Script for Generating a Heatmap

Create a file named `visualize_similarity.py`:

```python
import requests
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

def get_similarity_matrix(metric="cosine"):
    """Fetch similarity matrix from the API."""
    response = requests.get(f"http://localhost:5000/results?format=matrix&metric={metric}")
    
    if response.status_code != 200:
        print(f"Error: {response.json().get('message', 'Failed to fetch data')}")
        return None
    
    data = response.json()
    
    # Check if we have results
    if "metrics" not in data or metric not in data["metrics"]:
        print(f"No data available for {metric} metric")
        return None
    
    return data

def create_heatmap(data, metric="cosine", output_file=None):
    """Generate a heatmap visualization from similarity data."""
    # Extract document IDs and the similarity matrix
    doc_ids = data['document_ids']
    matrix = data['metrics'][metric]
    
    # Convert to numpy array for visualization
    matrix_values = np.zeros((len(doc_ids), len(doc_ids)))
    for i, doc1 in enumerate(doc_ids):
        for j, doc2 in enumerate(doc_ids):
            matrix_values[i, j] = matrix[doc1][doc2]
    
    # Set up the plot
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with annotations
    sns.heatmap(
        matrix_values, 
        annot=True, 
        fmt=".3f", 
        cmap="YlOrRd", 
        xticklabels=doc_ids, 
        yticklabels=doc_ids,
        vmin=0, 
        vmax=1
    )
    
    plt.title(f"Document Similarity ({metric.capitalize()})")
    plt.tight_layout()
    
    # Save to file if specified
    if output_file:
        plt.savefig(output_file)
        print(f"Heatmap saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    # Get metric from command line if provided, otherwise use cosine
    metric = sys.argv[1] if len(sys.argv) > 1 else "cosine"
    
    # Get output file name if provided
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Get data from API
    data = get_similarity_matrix(metric)
    
    if data:
        create_heatmap(data, metric, output_file)
```

Run the script:
```bash
python visualize_similarity.py cosine similarity_heatmap.png
```

### HTML Report Generator

Create a file named `generate_report.py`:

```python
import requests
import json
import datetime

def generate_html_report():
    """Generate an HTML report of similarity results."""
    # Get document stats
    status_response = requests.get("http://localhost:5000/status")
    status_data = status_response.json()
    
    # Get results in matrix format for all metrics
    results_response = requests.get("http://localhost:5000/results?format=matrix")
    results_data = results_response.json()
    
    if "metrics" not in results_data:
        print("No similarity data available. Please compare documents first.")
        return
    
    # Current date/time for the report
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Plagiarism Check Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; margin: 20px 0; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        .high {{ background-color: #ffcccc; }}
        .medium {{ background-color: #ffffcc; }}
        .low {{ background-color: #e6ffe6; }}
        .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .document-list {{ margin-bottom: 30px; }}
        .timestamp {{ color: #777; font-style: italic; }}
    </style>
</head>
<body>
    <h1>Plagiarism Check Results</h1>
    <div class="timestamp">Generated on: {current_time}</div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Documents analyzed: {status_data.get('document_count', 0)}</p>
        <p>Similarity comparisons: {status_data.get('cached_comparisons', 0)}</p>
    </div>
    
    <div class="document-list">
        <h2>Documents</h2>
        <ul>
"""

    # Add document list
    for doc_id in status_data.get('document_ids', []):
        html += f"            <li>{doc_id}</li>\n"
    
    html += """        </ul>
    </div>
    
    <h2>Similarity Matrices</h2>
"""

    # Add a table for each metric
    for metric, matrix in results_data.get('metrics', {}).items():
        html += f"    <h3>{metric.capitalize()} Similarity</h3>\n"
        html += "    <table>\n        <tr><th></th>\n"
        
        # Header row
        doc_ids = results_data.get('document_ids', [])
        for doc_id in doc_ids:
            html += f"            <th>{doc_id}</th>\n"
        html += "        </tr>\n"
        
        # Data rows
        for doc1 in doc_ids:
            html += f"        <tr><th>{doc1}</th>\n"
            for doc2 in doc_ids:
                value = matrix[doc1][doc2]
                css_class = ""
                if doc1 != doc2:  # Don't color the diagonal
                    if value >= 0.8:
                        css_class = "high"
                    elif value >= 0.5:
                        css_class = "medium"
                    else:
                        css_class = "low"
                html += f'            <td class="{css_class}">{value:.3f}</td>\n'
            html += "        </tr>\n"
        
        html += "    </table>\n\n"

    # Add interpretation guide
    html += """
    <h2>Interpretation Guide</h2>
    <ul>
        <li><span style="background-color: #ffcccc; padding: 2px 5px;">High similarity (0.8-1.0)</span>: Strong indication of potential plagiarism</li>
        <li><span style="background-color: #ffffcc; padding: 2px 5px;">Medium similarity (0.5-0.8)</span>: Possible partial reuse or common sources</li>
        <li><span style="background-color: #e6ffe6; padding: 2px 5px;">Low similarity (0.0-0.5)</span>: Likely different content</li>
    </ul>
    
    <h2>Metric Descriptions</h2>
    <ul>
        <li><strong>Cosine</strong>: Measures the cosine of the angle between document vectors, focusing on content rather than length</li>
        <li><strong>Jaccard</strong>: Measures similarity as intersection over union of terms</li>
        <li><strong>Euclidean</strong>: Based on Euclidean distance between document vectors</li>
        <li><strong>Levenshtein</strong>: Based on edit distance (character-level changes)</li>
    </ul>
</body>
</html>"""

    # Save to file
    with open("plagiarism_report.html", "w") as f:
        f.write(html)
    
    print("Report saved to plagiarism_report.html")

if __name__ == "__main__":
    generate_html_report()
```

Run the script:
```bash
python generate_report.py
```

## Client Applications

### Complete Python Client

Create a file named `plagiarism_client.py`:

```python
import requests
import json
import argparse
import sys

class PlagiarismClient:
    """Client for the Plagiarism Checker API."""
    
    def __init__(self, base_url="http://localhost:5000"):
        """Initialize the client with the API base URL."""
        self.base_url = base_url
    
    def check_api_status(self):
        """Check if the API is running."""
        try:
            response = requests.get(f"{self.base_url}/status")
            return response.json()
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to the API server")
            return None
    
    def submit_document(self, doc_id, text):
        """Submit a document to the API."""
        payload = {
            "doc_id": doc_id,
            "text": text
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/submit",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            return response.json()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Request failed: {str(e)}"
            }
    
    def submit_document_from_file(self, doc_id, file_path):
        """Submit a document from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return self.submit_document(doc_id, text)
        except Exception as e:
            return {
                "status": "error",
                "message": f"File error: {str(e)}"
            }
    
    def compare_documents(self, doc_id_1, doc_id_2, metric="cosine"):
        """Compare two documents."""
        payload = {
            "doc_id_1": doc_id_1,
            "doc_id_2": doc_id_2,
            "metric": metric
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/compare",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            return response.json()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Request failed: {str(e)}"
            }
    
    def get_results(self, format="pairs", metric=None):
        """Get all similarity results."""
        params = {}
        if format:
            params['format'] = format
        if metric:
            params['metric'] = metric
        
        try:
            response = requests.get(f"{self.base_url}/results", params=params)
            return response.json()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Request failed: {str(e)}"
            }

def print_json(data):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2))

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description='Plagiarism Checker Client')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Status command
    subparsers.add_parser('status', help='Check API status')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit a document')
    submit_parser.add_argument('doc_id', help='Document ID')
    submit_group = submit_parser.add_mutually_exclusive_group(required=True)
    submit_group.add_argument('--text', help='Document text')
    submit_group.add_argument('--file', help='Path to document file')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two documents')
    compare_parser.add_argument('doc_id_1', help='First document ID')
    compare_parser.add_argument('doc_id_2', help='Second document ID')
    compare_parser.add_argument('--metric', default='cosine', 
                               help='Similarity metric (default: cosine)')
    
    # Results command
    results_parser = subparsers.add_parser('results', help='Get similarity results')
    results_parser.add_argument('--format', default='pairs', choices=['pairs', 'matrix'],
                               help='Result format (default: pairs)')
    results_parser.add_argument('--metric', help='Filter by metric')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize client
    client = PlagiarismClient()
    
    # Execute commands
    if args.command == 'status' or args.command is None:
        result = client.check_api_status()
        if result:
            print_json(result)
        
    elif args.command == 'submit':
        if args.text:
            result = client.submit_document(args.doc_id, args.text)
        else:
            result = client.submit_document_from_file(args.doc_id, args.file)
        print_json(result)
        
    elif args.command == 'compare':
        result = client.compare_documents(args.doc_id_1, args.doc_id_2, args.metric)
        print_json(result)
        
    elif args.command == 'results':
        result = client.get_results(args.format, args.metric)
        print_json(result)

if __name__ == "__main__":
    main()
```

Use the client:
```bash
# Check API status
python plagiarism_client.py status

# Submit a document from text
python plagiarism_client.py submit doc1 --text "This is a test document"

# Submit a document from file
python plagiarism_client.py submit doc2 --file path/to/document.txt

# Compare documents
python plagiarism_client.py compare doc1 doc2 --metric jaccard

# Get results
python plagiarism_client.py results --format matrix --metric cosine
```

## Error Handling

The API includes comprehensive error handling:

### HTTP Status Codes:
- **200 OK**: Request succeeded
- **400 Bad Request**: Invalid input/parameters
- **404 Not Found**: Document or endpoint not found
- **405 Method Not Allowed**: Wrong HTTP method
- **500 Internal Server Error**: Server-side error

### Error Response Format:
All error responses follow this format:
```json
{
  "status": "error",
  "message": "Specific error message explaining the problem"
}
```

### Common Error Messages:
- Missing required fields
- Invalid document ID format
- Invalid similarity metric
- Document not found
- Request must be JSON

## Performance and Security Considerations

### Performance:
- **Document Size Limits**: Maximum text size is 1MB
- **Caching**: Results are cached to avoid redundant calculations
- **Document ID Validation**: Prevents potential issues with special characters

### Security:
- **Input Validation**: All inputs are validated to prevent injection attacks
- **Error Handling**: Does not expose internal details in error messages
- **Development Server Warning**: The built-in Flask server is not for production

### Production Recommendations:
- Use a production-grade WSGI server like Gunicorn or uWSGI
- Add authentication/authorization
- Implement rate limiting
- Set up HTTPS
- Use a database for persistent storage

## Troubleshooting

### Common Issues and Solutions:

1. **API server won't start:**
   - Check if the port is already in use:
     ```bash
     lsof -i :5000
     ```
   - Solution: Change the port in the api.py file or terminate the other process

2. **"Connection refused" errors:**
   - Ensure the server is running
   - Check if you're using the correct host/port
   - Verify firewall settings allow connections to the port

3. **Validation errors:**
   - Check the error message for specific validation failures
   - Ensure document IDs follow the allowed format
   - Verify text content isn't empty or too large

4. **No results from /results endpoint:**
   - You need to compare documents first using the /compare endpoint
   - Verify your documents were submitted successfully  

5. **Incorrect similarity scores:**
   - Try different metrics for comparison
   - Verify the documents contain the expected text
   - Consider the impact of text preprocessing (lowercasing, stopword removal, etc.)

### Debugging:
- The API runs in debug mode by default, showing detailed error messages
- Check the terminal where the API is running for error logs
- For client issues, add print statements to debug requests and responses

### Getting Help:
- Refer to this guide for detailed explanations
- Check the API status endpoint for version information
- Examine the source code comments for additional details