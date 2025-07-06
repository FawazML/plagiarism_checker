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