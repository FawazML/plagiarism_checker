import requests
import json

# Configuration
API_BASE = "http://localhost:5000"

def submit_document(doc_id, text):
    """Submit a document to the API."""
    response = requests.post(
        f"{API_BASE}/submit",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"doc_id": doc_id, "text": text})
    )
    return response.json()

def compare_documents(doc_id_1, doc_id_2, metric="cosine"):
    """Compare two documents using the specified metric."""
    response = requests.post(
        f"{API_BASE}/compare",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "doc_id_1": doc_id_1,
            "doc_id_2": doc_id_2,
            "metric": metric
        })
    )
    return response.json()

def get_api_status():
    """Get the current status of the API."""
    response = requests.get(f"{API_BASE}/status")
    return response.json()

def get_all_results(format="pairs", metric=None):
    """Get all similarity results."""
    params = {}
    if format:
        params['format'] = format
    if metric:
        params['metric'] = metric
    
    response = requests.get(f"{API_BASE}/results", params=params)
    return response.json()

# Example usage
if __name__ == "__main__":
    # Check initial status
    print("Initial API status:")
    print(json.dumps(get_api_status(), indent=2))
    print()
    
    # Submit sample documents
    doc1_text = "This is a test document for plagiarism detection. It contains some specific text."
    doc2_text = "This document is for testing plagiarism. It has some specific text inside."
    doc3_text = "This is a completely different document about machine learning and data science."
    
    print("Submitting document 1...")
    submit_result1 = submit_document("sample1", doc1_text)
    print(f"Result: {json.dumps(submit_result1, indent=2)}\n")
    
    print("Submitting document 2...")
    submit_result2 = submit_document("sample2", doc2_text)
    print(f"Result: {json.dumps(submit_result2, indent=2)}\n")
    
    print("Submitting document 3...")
    submit_result3 = submit_document("sample3", doc3_text)
    print(f"Result: {json.dumps(submit_result3, indent=2)}\n")
    
    # Compare with different metrics
    metrics = ["cosine", "jaccard", "euclidean"]
    
    for metric in metrics:
        print(f"Comparing documents using {metric} similarity...")
        result = compare_documents("sample1", "sample2", metric)
        print(f"Result: {json.dumps(result, indent=2)}\n")
    
    # Compare other document pairs
    print("Comparing sample1 and sample3...")
    result = compare_documents("sample1", "sample3")
    print(f"Result: {json.dumps(result, indent=2)}\n")
    
    print("Comparing sample2 and sample3...")
    result = compare_documents("sample2", "sample3")
    print(f"Result: {json.dumps(result, indent=2)}\n")
    
    # Get results in different formats
    print("Getting all results (pairs format):")
    all_results = get_all_results(format="pairs")
    print(f"Result: {json.dumps(all_results, indent=2)}\n")
    
    print("Getting all cosine results (matrix format):")
    cosine_results = get_all_results(format="matrix", metric="cosine")
    print(f"Result: {json.dumps(cosine_results, indent=2)}\n")