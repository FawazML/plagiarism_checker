# api.py
from flask import Flask, jsonify, request
from plagcheck.modules.text_processor import TextProcessor
from plagcheck.modules.similarity import DocumentSimilarity, SimilarityMetric
import re
import os
import logging
# Import database modules
from plagcheck.db.db import init_db, get_db
from plagcheck.db.models import SimilarityResult

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize TextProcessor with default settings
text_processor = TextProcessor()

# Initialize DocumentSimilarity with the text processor
document_similarity = DocumentSimilarity(preprocessor=text_processor)

# Dictionary to store preprocessed documents
documents = {}

# Dictionary to store similarity scores between document pairs
# Format: {(doc_id_1, doc_id_2, metric): similarity_score}
similarity_scores = {}

# Validation functions
def validate_doc_id(doc_id):
    """Validate document ID format."""
    if not isinstance(doc_id, str):
        return False, "Document ID must be a string"
    
    if not doc_id.strip():
        return False, "Document ID cannot be empty"
        
    if not re.match(r'^[a-zA-Z0-9_-]+$', doc_id):
        return False, "Document ID can only contain letters, numbers, underscores, and hyphens"
        
    if len(doc_id) > 64:
        return False, "Document ID cannot be longer than 64 characters"
        
    return True, None

def validate_text(text):
    """Validate text content."""
    if not isinstance(text, str):
        return False, "Text must be a string"
    
    if not text.strip():
        return False, "Text cannot be empty"
        
    if len(text) > 1000000:  # Limit to 1MB of text
        return False, "Text is too large (max 1MB)"
        
    return True, None

def validate_metric(metric):
    """Validate similarity metric."""
    valid_metrics = [m.value for m in SimilarityMetric]
    
    if not isinstance(metric, str):
        return False, "Metric must be a string"
        
    if metric not in valid_metrics:
        return False, f"Invalid metric. Valid options: {', '.join(valid_metrics)}"
        
    return True, None

@app.route('/')
def index():
    """Root route that confirms the API is active."""
    return jsonify({
        'status': 'success',
        'message': 'Plagiarism Checker API is active',
        'version': '1.0.0'
    })

@app.route('/submit', methods=['POST'])
def submit_document():
    """
    Endpoint to submit a document for preprocessing and storage.
    
    Expects JSON payload with:
    - doc_id: Unique identifier for the document
    - text: Content of the document
    """
    # Validate request content type
    if not request.is_json:
        return jsonify({
            'status': 'error',
            'message': 'Request must be JSON'
        }), 400
    
    # Get JSON data from request
    data = request.get_json()
    
    # Validate data structure
    if not data or not isinstance(data, dict):
        return jsonify({
            'status': 'error',
            'message': 'Invalid JSON payload format'
        }), 400
    
    # Validate required fields are present
    if 'doc_id' not in data:
        return jsonify({
            'status': 'error',
            'message': 'Missing required field: doc_id must be provided'
        }), 400
        
    if 'text' not in data:
        return jsonify({
            'status': 'error',
            'message': 'Missing required field: text must be provided'
        }), 400
    
    doc_id = data['doc_id']
    text = data['text']
    
    # Validate doc_id format
    is_valid, error_message = validate_doc_id(doc_id)
    if not is_valid:
        return jsonify({
            'status': 'error',
            'message': error_message
        }), 400
    
    # Validate text content
    is_valid, error_message = validate_text(text)
    if not is_valid:
        return jsonify({
            'status': 'error',
            'message': error_message
        }), 400
    
    try:
        # Store the text directly - we'll preprocess during comparison
        documents[doc_id] = text
    except Exception as e:
        logger.error(f"Error storing document: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error storing document: {str(e)}'
        }), 500
    
    # When a document is updated, clear any cached similarity scores involving this document
    keys_to_remove = []
    for key in similarity_scores.keys():
        if doc_id in key[:2]:  # The first two elements of the key tuple are doc_ids
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del similarity_scores[key]
    
    logger.info(f"Document {doc_id} submitted successfully")
    return jsonify({
        'status': 'success',
        'message': f'Document {doc_id} has been stored',
        'doc_id': doc_id,
        'document_count': len(documents),
        'cached_comparisons_removed': len(keys_to_remove)
    })

@app.route('/compare', methods=['POST'])
def compare_documents():
    """
    Endpoint to compare two documents for similarity.
    
    Expects JSON payload with:
    - doc_id_1: ID of the first document to compare
    - doc_id_2: ID of the second document to compare
    - metric: (Optional) Similarity metric to use (default: cosine)
    """
    # Validate request content type
    if not request.is_json:
        return jsonify({
            'status': 'error',
            'message': 'Request must be JSON'
        }), 400
    
    # Get JSON data from request
    data = request.get_json()
    
    # Validate data structure
    if not data or not isinstance(data, dict):
        return jsonify({
            'status': 'error',
            'message': 'Invalid JSON payload format'
        }), 400
    
    # Validate required fields are present
    if 'doc_id_1' not in data:
        return jsonify({
            'status': 'error',
            'message': 'Missing required field: doc_id_1 must be provided'
        }), 400
        
    if 'doc_id_2' not in data:
        return jsonify({
            'status': 'error',
            'message': 'Missing required field: doc_id_2 must be provided'
        }), 400
    
    doc_id_1 = data['doc_id_1']
    doc_id_2 = data['doc_id_2']
    
    # Validate doc_id_1 format
    is_valid, error_message = validate_doc_id(doc_id_1)
    if not is_valid:
        return jsonify({
            'status': 'error',
            'message': f'Invalid doc_id_1: {error_message}'
        }), 400
    
    # Validate doc_id_2 format
    is_valid, error_message = validate_doc_id(doc_id_2)
    if not is_valid:
        return jsonify({
            'status': 'error',
            'message': f'Invalid doc_id_2: {error_message}'
        }), 400
    
    # Get the similarity metric (default to cosine if not provided)
    metric = data.get('metric', 'cosine')
    
    # Validate metric
    is_valid, error_message = validate_metric(metric)
    if not is_valid:
        return jsonify({
            'status': 'error',
            'message': error_message
        }), 400
    
    # Validate that documents exist
    if doc_id_1 not in documents:
        return jsonify({
            'status': 'error',
            'message': f'Document with ID {doc_id_1} not found'
        }), 404
    
    if doc_id_2 not in documents:
        return jsonify({
            'status': 'error',
            'message': f'Document with ID {doc_id_2} not found'
        }), 404
    
    # Don't allow comparing a document to itself (it's always 1.0)
    if doc_id_1 == doc_id_2:
        return jsonify({
            'status': 'success',
            'doc_id_1': doc_id_1,
            'doc_id_2': doc_id_2,
            'metric': metric,
            'similarity_score': 1.0,
            'from_cache': True,
            'note': 'Documents are identical (same ID)'
        })
    
    # Sort document IDs for caching purposes (so (doc1,doc2) and (doc2,doc1) are considered the same)
    sorted_ids = tuple(sorted([doc_id_1, doc_id_2]))
    doc_id_1, doc_id_2 = sorted_ids
    
    # Try to get from cache first
    cache_key = (doc_id_1, doc_id_2, metric)
    from_cache = cache_key in similarity_scores
    
    if from_cache:
        similarity_score = similarity_scores[cache_key]
        logger.info(f"Retrieved similarity score from cache for {doc_id_1} and {doc_id_2}")
    else:
        try:
            # Get raw texts
            text1 = documents[doc_id_1]
            text2 = documents[doc_id_2]
            
            # Get database session
            session = get_db()
            try:
                # Compare texts - use the updated method with database integration
                similarity_score = document_similarity.compare_two_texts(
                    text1, text2, 
                    SimilarityMetric(metric), 
                    preprocess=False,  # Disable preprocessing as we're having issues with it
                    doc_id1=doc_id_1,
                    doc_id2=doc_id_2,
                    db_session=session,
                    store_in_db=True
                )
                
                # Commit the session to save changes to the database
                session.commit()
                logger.info(f"Committed similarity result to database for {doc_id_1} and {doc_id_2}")
            except Exception as e:
                # Rollback on error
                session.rollback()
                logger.error(f"Error during database operation: {str(e)}")
                raise
            finally:
                session.close()
            
            # Cache the result
            similarity_scores[cache_key] = similarity_score
            logger.info(f"Calculated new similarity score for {doc_id_1} and {doc_id_2}")
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Error calculating similarity: {str(e)}'
            }), 500
    
    return jsonify({
        'status': 'success',
        'doc_id_1': doc_id_1,
        'doc_id_2': doc_id_2,
        'metric': metric,
        'similarity_score': similarity_score,
        'from_cache': from_cache
    })

@app.route('/compare-batch', methods=['POST'])
def compare_batch():
    """
    Endpoint to compare multiple document pairs at once.
    
    Expects JSON payload with:
    - pairs: Array of objects with doc_id_1 and doc_id_2
    - metric: (Optional) Similarity metric to use (default: cosine)
    """
    # Validate request content type
    if not request.is_json:
        return jsonify({
            'status': 'error',
            'message': 'Request must be JSON'
        }), 400
    
    # Get JSON data from request
    data = request.get_json()
    
    # Validate data structure
    if not data or not isinstance(data, dict):
        return jsonify({
            'status': 'error',
            'message': 'Invalid JSON payload format'
        }), 400
    
    # Validate required fields are present
    if 'pairs' not in data:
        return jsonify({
            'status': 'error',
            'message': 'Missing required field: pairs array must be provided'
        }), 400
    
    # Get pairs and validate array format
    pairs = data['pairs']
    if not isinstance(pairs, list):
        return jsonify({
            'status': 'error',
            'message': 'Pairs must be an array of objects with doc_id_1 and doc_id_2'
        }), 400
    
    # Get the similarity metric (default to cosine if not provided)
    metric = data.get('metric', 'cosine')
    
    # Validate metric
    is_valid, error_message = validate_metric(metric)
    if not is_valid:
        return jsonify({
            'status': 'error',
            'message': error_message
        }), 400
    
    # Process each pair
    results = {}
    errors = []
    
    # Get database session
    session = get_db()
    try:
        for i, pair in enumerate(pairs):
            # Validate pair format
            if not isinstance(pair, dict):
                errors.append({
                    'index': i,
                    'pair': pair,
                    'message': 'Pair must be an object with doc_id_1 and doc_id_2'
                })
                continue
                
            if 'doc_id_1' not in pair or 'doc_id_2' not in pair:
                errors.append({
                    'index': i,
                    'pair': pair,
                    'message': 'Pair must have doc_id_1 and doc_id_2 fields'
                })
                continue
                
            doc_id_1 = pair['doc_id_1']
            doc_id_2 = pair['doc_id_2']
            
            # Validate doc_id_1 format
            is_valid, error_message = validate_doc_id(doc_id_1)
            if not is_valid:
                errors.append({
                    'index': i,
                    'pair': pair,
                    'message': f'Invalid doc_id_1: {error_message}'
                })
                continue
            
            # Validate doc_id_2 format
            is_valid, error_message = validate_doc_id(doc_id_2)
            if not is_valid:
                errors.append({
                    'index': i,
                    'pair': pair,
                    'message': f'Invalid doc_id_2: {error_message}'
                })
                continue
                
            # Validate that documents exist
            if doc_id_1 not in documents:
                errors.append({
                    'index': i,
                    'pair': pair,
                    'message': f'Document with ID {doc_id_1} not found'
                })
                continue
                
            if doc_id_2 not in documents:
                errors.append({
                    'index': i,
                    'pair': pair,
                    'message': f'Document with ID {doc_id_2} not found'
                })
                continue
                
            # Handle identical documents (always 1.0)
            if doc_id_1 == doc_id_2:
                results[f'{doc_id_1}_{doc_id_2}'] = 1.0
                continue
            
            # Sort document IDs for caching purposes
            sorted_ids = tuple(sorted([doc_id_1, doc_id_2]))
            doc_id_1, doc_id_2 = sorted_ids
                
            # Try to get from cache first
            cache_key = (doc_id_1, doc_id_2, metric)
            if cache_key in similarity_scores:
                results[f'{doc_id_1}_{doc_id_2}'] = similarity_scores[cache_key]
                continue
                
            try:
                # Get raw texts
                text1 = documents[doc_id_1]
                text2 = documents[doc_id_2]
                
                # Compare texts - use the updated method with database integration
                similarity_score = document_similarity.compare_two_texts(
                    text1, text2, 
                    SimilarityMetric(metric), 
                    preprocess=False,  # Disable preprocessing as we're having issues with it
                    doc_id1=doc_id_1,
                    doc_id2=doc_id_2,
                    db_session=session,
                    store_in_db=True
                )
                
                # Cache the result
                similarity_scores[cache_key] = similarity_score
                logger.info(f"Calculated new similarity score for {doc_id_1} and {doc_id_2}")
                
                # Store the result
                results[f'{doc_id_1}_{doc_id_2}'] = similarity_score
                
            except Exception as e:
                logger.error(f"Error calculating similarity for pair {pair}: {str(e)}")
                errors.append({
                    'index': i,
                    'pair': pair,
                    'message': f'Error calculating similarity: {str(e)}'
                })
                
        # Commit all database changes at once
        session.commit()
        logger.info(f"Committed {len(results)} similarity results to database")
    except Exception as e:
        # Rollback on error
        session.rollback()
        logger.error(f"Error during database operations: {str(e)}")
    finally:
        session.close()
    
    # Prepare response
    response = {
        'status': 'success',
        'metric': metric,
        'total_pairs': len(pairs),
        'processed_pairs': len(results),
        'results': results
    }
    
    # Add errors if any
    if errors:
        response['errors'] = errors
        response['status'] = 'partial_success' if results else 'error'
    
    return jsonify(response)

@app.route('/status', methods=['GET'])
def get_status():
    """
    Endpoint to get the current status of the API.
    Returns the number of documents stored and cached comparisons.
    """
    # Get database results count
    db_results_count = 0
    try:
        session = get_db()
        try:
            db_results_count = session.query(SimilarityResult).count()
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Error querying database: {str(e)}")
    
    return jsonify({
        'status': 'success',
        'document_count': len(documents),
        'cached_comparisons': len(similarity_scores),
        'database_results': db_results_count,
        'document_ids': list(documents.keys()),
        'api_version': '1.1.0',
        'available_metrics': [m.value for m in SimilarityMetric]
    })

@app.route('/results', methods=['GET'])
def get_results():
    """
    Endpoint to get all stored similarity results.
    Returns all cached similarity scores formatted as a dictionary where
    each key is a document pair string and the value is their similarity score.
    
    Query parameters:
    - format: (Optional) Format of the results - 'pairs' (default) or 'matrix'
    - metric: (Optional) Filter results by metric (e.g., 'cosine', 'jaccard')
    - threshold: (Optional) Filter results to show only scores >= threshold
    """
    # Get query parameters
    result_format = request.args.get('format', 'pairs')
    metric_filter = request.args.get('metric', None)
    
    # Get threshold parameter (if provided)
    threshold_str = request.args.get('threshold', None)
    threshold = None
    if threshold_str:
        try:
            threshold = float(threshold_str)
            if not 0 <= threshold <= 1:
                return jsonify({
                    'status': 'error',
                    'message': f"Threshold must be between 0 and 1, got {threshold}"
                }), 400
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': f"Invalid threshold value: {threshold_str}. Must be a number between 0 and 1."
            }), 400
    
    # Validate query parameters
    if result_format not in ['pairs', 'matrix']:
        return jsonify({
            'status': 'error',
            'message': f"Invalid format: {result_format}. Supported formats: 'pairs', 'matrix'"
        }), 400
    
    # If metric filter is provided, validate it
    if metric_filter:
        is_valid, error_message = validate_metric(metric_filter)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': error_message
            }), 400
    
    # Check if there are any results
    if not similarity_scores:
        return jsonify({
            'status': 'success',
            'format': result_format,
            'metric_filter': metric_filter,
            'threshold': threshold,
            'message': 'No similarity scores are available. Compare documents first.',
            'result_count': 0,
            'results': {}
        })
    
    if result_format == 'pairs':
        # Format as document pairs
        results = {}
        for key, score in similarity_scores.items():
            doc_id_1, doc_id_2, metric = key
            
            # Apply metric filter if specified
            if metric_filter and metric != metric_filter:
                continue
                
            # Apply threshold filter if specified
            if threshold is not None and score < threshold:
                continue
                
            # Create a readable key for the pair
            pair_key = f"{doc_id_1}__{doc_id_2}__{metric}"
            results[pair_key] = score
            
        return jsonify({
            'status': 'success',
            'format': 'pairs',
            'metric_filter': metric_filter,
            'threshold': threshold,
            'result_count': len(results),
            'results': results
        })
        
    elif result_format == 'matrix':
        # Group results by metric
        metrics = {}
        document_ids = sorted(list(documents.keys()))
        
        # Find all unique metrics
        all_metrics = set()
        for _, _, metric in similarity_scores.keys():
            if not metric_filter or metric == metric_filter:
                all_metrics.add(metric)
        
        # Create a matrix for each metric
        for metric in all_metrics:
            # Initialize matrix with zeros
            matrix = {}
            for doc_id1 in document_ids:
                matrix[doc_id1] = {}
                for doc_id2 in document_ids:
                    if doc_id1 == doc_id2:
                        matrix[doc_id1][doc_id2] = 1.0  # Documents are identical to themselves
                    else:
                        matrix[doc_id1][doc_id2] = 0.0
            
            # Track if this metric has any results above the threshold
            has_threshold_matches = False if threshold else True
            
            # Fill in known values
            for key, score in similarity_scores.items():
                doc_id1, doc_id2, key_metric = key
                if key_metric == metric:
                    matrix[doc_id1][doc_id2] = score
                    matrix[doc_id2][doc_id1] = score  # Ensure symmetry
                    
                    # Check if score meets threshold
                    if threshold is not None and score >= threshold:
                        has_threshold_matches = True
            
            # If applying a threshold and no matches found, skip this metric
            if not has_threshold_matches:
                continue
                
            metrics[metric] = matrix
            
        return jsonify({
            'status': 'success',
            'format': 'matrix',
            'metric_filter': metric_filter,
            'threshold': threshold,
            'document_ids': document_ids,
            'metrics': metrics
        })

@app.route('/db-results', methods=['GET'])
def get_db_results():
    """
    Endpoint to get results stored in the database.
    Returns all similarity scores stored in the database.
    
    Query parameters:
    - threshold: (Optional) Filter results to show only scores >= threshold
    """
    # Get threshold parameter (if provided)
    threshold_str = request.args.get('threshold', None)
    threshold = None
    if threshold_str:
        try:
            threshold = float(threshold_str)
            if not 0 <= threshold <= 1:
                return jsonify({
                    'status': 'error',
                    'message': f"Threshold must be between 0 and 1, got {threshold}"
                }), 400
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': f"Invalid threshold value: {threshold_str}. Must be a number between 0 and 1."
            }), 400
            
    try:
        session = get_db()
        try:
            # Apply threshold filter if specified
            if threshold is not None:
                results = session.query(SimilarityResult).filter(SimilarityResult.score >= threshold).all()
            else:
                results = session.query(SimilarityResult).all()
            
            # Convert to dictionary format
            results_dict = {}
            for result in results:
                key = f"{result.doc1}__{result.doc2}"
                results_dict[key] = {
                    "score": result.score,
                    "timestamp": result.timestamp.isoformat() if result.timestamp else None
                }
            
            return jsonify({
                'status': 'success',
                'threshold': threshold,
                'result_count': len(results),
                'results': results_dict
            })
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Error retrieving results from database: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error retrieving results from database: {str(e)}"
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'status': 'error',
        'message': 'Method not allowed'
    }), 405

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

def create_app():
    """Create and configure the Flask application (for WSGI servers)."""
    # Initialize the database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        
    return app

def main():
    """Run the API server."""
    # Initialize the database
    init_db()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()