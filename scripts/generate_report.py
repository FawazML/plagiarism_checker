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