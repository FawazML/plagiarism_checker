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