# main.py
import argparse
import os
from plagcheck.modules.file_handler import FileHandler
from plagcheck.modules.text_processor import TextProcessor, TextNormalizer, Tokenizer, StopwordRemover
from plagcheck.modules.similarity import SimilarityCalculator, SimilarityMetric, DocumentSimilarity
from plagcheck.modules.result_presenter import ResultPresenter, ColorTheme, OutputFormat
# Import the database modules
from plagcheck.db.db import init_db, get_db, shutdown_db
from plagcheck.db.models import SimilarityResult
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully!")

    parser = argparse.ArgumentParser(description='Check plagiarism among text documents')
    parser.add_argument('--dir', type=str, default=None, help='Directory containing text files')
    parser.add_argument('--ext', type=str, default='.txt', help='File extension to process')
    parser.add_argument('--output', type=str, default=None, help='Output file for results')
    parser.add_argument('--output-format', type=str, choices=['json', 'csv', 'html', 'markdown', 'latex'], 
                        default=None, help='Format for output file')
    parser.add_argument('--normalize', action='store_true', help='Apply text normalization')
    parser.add_argument('--remove-stopwords', action='store_true', help='Remove stopwords')
    parser.add_argument('--ngram', type=int, default=1, help='Use n-grams for comparison (n=1 for words, n>1 for char n-grams)')
    parser.add_argument('--min-df', type=float, default=1, help='Min document frequency for terms')
    parser.add_argument('--max-df', type=float, default=1.0, help='Max document frequency for terms')
    parser.add_argument('--similarity', type=str, default='cosine', 
                        choices=['cosine', 'jaccard', 'euclidean', 'manhattan', 'dice', 'overlap', 'levenshtein'],
                        help='Similarity metric to use')
    parser.add_argument('--binary', action='store_true', help='Use binary vectorization (0/1 instead of counts/weights)')
    parser.add_argument('--compare-all', action='store_true', help='Compare documents with all similarity metrics')
    parser.add_argument('--color-theme', type=str, default='heatmap', 
                        choices=['none', 'basic', 'heatmap', 'traffic_light'],
                        help='Color theme for console output')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Directory for saving comprehensive report with multiple formats')
    parser.add_argument('--image', type=str, default=None, help='Save visualization to image file')
    parser.add_argument('--image-format', type=str, default='heatmap', 
                        choices=['heatmap', 'network', 'bar', 'cluster', 'mds'],
                        help='Visualization format to use')
    parser.add_argument('--vmin', type=float, default=0.0, help='Minimum value for visualization colormap')
    parser.add_argument('--vmax', type=float, default=1.0, help='Maximum value for visualization colormap')
    parser.add_argument('--no-store-db', action='store_true', help='Skip storing results in the database')
    
    args = parser.parse_args()
    
    # Check if files are specified or print usage
    if not args.dir:
        parser.print_usage()
        print("\nError: No input directory specified.")
        return
    
    # Load text files
    file_handler = FileHandler()
    documents = file_handler.load_directory(args.dir, extension=args.ext)
    
    if not documents:
        print(f"No {args.ext} files found in {args.dir}")
        return
    
    print(f"Loaded {len(documents)} documents")
    
    # Configure text processor
    tokenizer = Tokenizer()
    normalizer = TextNormalizer() if args.normalize else None
    stopword_remover = StopwordRemover() if args.remove_stopwords else None
    
    text_processor = TextProcessor(
        tokenizer=tokenizer,
        normalizer=normalizer,
        stopword_remover=stopword_remover
    )
    
    # Configure similarity calculator
    similarity_calculator = SimilarityCalculator(
        vectorizer_type='tfidf',  # Use TF-IDF vectorizer
        ngram_range=(1, args.ngram),  # Use word n-grams
        min_df=args.min_df,
        max_df=args.max_df,
        binary=args.binary
    )
    
    # Configure document similarity
    document_similarity = DocumentSimilarity(
        preprocessor=text_processor,
        calculator=similarity_calculator
    )
    
    # Process texts
    processed_texts = {}
    print("Processing documents...")
    for doc_id, text in documents.items():
        processed_texts[doc_id] = text_processor.process_text(text)
    
    # Calculate similarity
    print("Calculating similarity...")
    
    # Determine which metrics to use
    if args.compare_all:
        metrics = [metric for metric in SimilarityMetric]
    else:
        metrics = [SimilarityMetric(args.similarity)]
    
    similarity_results = {}
    
    for metric in metrics:
        print(f"Using metric: {metric.value}")
        similarity_matrix = document_similarity.calculate_similarity_matrix(
            processed_texts, 
            metric=metric
        )
        similarity_results[metric] = similarity_matrix
        
        # Store results in database if not disabled
        if not args.no_store_db:
            try:
                with get_db() as db:
                    # Store each document pair comparison in the database
                    doc_ids = list(processed_texts.keys())
                    for i, doc1 in enumerate(doc_ids):
                        for j, doc2 in enumerate(doc_ids[i+1:], i+1):
                            # Get similarity score from the matrix
                            score = similarity_matrix[doc1][doc2]
                            
                            # Create and add the database entry
                            result = SimilarityResult(
                                doc1=doc1,
                                doc2=doc2,
                                score=score,
                                timestamp=datetime.now()
                            )
                            db.add(result)
                logger.info(f"Saved {len(doc_ids) * (len(doc_ids) - 1) // 2} comparison results to database for metric {metric.value}")
            except Exception as e:
                logger.error(f"Error saving to database: {str(e)}")
    
    # Configure result presenter
    result_presenter = ResultPresenter(
        color_theme=ColorTheme(args.color_theme),
        output_format=OutputFormat.CONSOLE
    )
    
    # Display results
    result_presenter.present_results(similarity_results)
    
    # Generate visualization if requested
    if args.image:
        print(f"Generating visualization: {args.image}")
        result_presenter.visualize(
            similarity_results[metrics[0]],  # Use first metric for visualization
            args.image,
            visualization_type=args.image_format,
            vmin=args.vmin,
            vmax=args.vmax,
            output_format=args.image_format
        )
    
    # Save to file if specified
    if args.output:
        result_presenter.save_results(
            similarity_results, 
            args.output,
            format=args.output_format
        )
    
    # Generate comprehensive report if output directory specified
    if args.output_dir:
        result_presenter.generate_full_report(
            similarity_results,
            args.output_dir,
            include_visualizations=True
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
    finally:
        # Clean up database resources when shutting down
        logger.info("Shutting down database connections...")
        shutdown_db()
