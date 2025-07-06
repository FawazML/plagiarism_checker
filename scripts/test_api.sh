#!/bin/bash
#!/bin/bash

# test_api.sh - Test script for the Plagiarism Checker API
# This script demonstrates how to use the API endpoints with curl commands
# It submits documents, compares them, and fetches result data

# Configuration
API_URL="http://localhost:5000"
VERBOSE=true  # Set to true for detailed output, false for minimal output

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print section headers
section() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Function to print success messages
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print info messages
info() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${YELLOW}ℹ $1${NC}"
    fi
}

# Function to print error messages
error() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

# Function to check if the API is running
check_api() {
    section "Checking if API is running"
    
    response=$(curl -s $API_URL)
    if [[ $response == *"Plagiarism Checker API is active"* ]]; then
        success "API is running"
    else
        error "API is not running. Please start the API first with 'python api.py'"
    fi
}

# Function to check API status
check_status() {
    section "Getting API status"
    
    curl -s $API_URL/status | json_pp
}

# Function to submit a document
submit_document() {
    local doc_id=$1
    local text="$2"
    
    info "Submitting document: $doc_id"
    
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"doc_id\":\"$doc_id\",\"text\":\"$text\"}" \
        $API_URL/submit)
    
    # Check if submission was successful
    if [[ $response == *"success"* ]]; then
        success "Document $doc_id submitted successfully"
        if [ "$VERBOSE" = true ]; then
            echo $response | json_pp
        fi
    else
        error "Failed to submit document $doc_id: $response"
    fi
}

# Function to compare two documents
compare_documents() {
    local doc_id_1=$1
    local doc_id_2=$2
    local metric=${3:-cosine}  # Default metric is cosine
    
    section "Comparing documents: $doc_id_1 vs $doc_id_2 using $metric metric"
    
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"doc_id_1\":\"$doc_id_1\",\"doc_id_2\":\"$doc_id_2\",\"metric\":\"$metric\"}" \
        $API_URL/compare | json_pp
}

# Function to compare multiple document pairs in batch
compare_batch() {
    local pairs_json=$1
    local metric=${2:-cosine}  # Default metric is cosine
    
    section "Comparing document pairs in batch using $metric metric"
    
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"pairs\":$pairs_json,\"metric\":\"$metric\"}" \
        $API_URL/compare-batch | json_pp
}

# Function to get all results
get_results() {
    local format=${1:-pairs}  # Default format is pairs
    local metric=$2  # Optional metric filter
    local threshold=$3  # Optional threshold filter
    
    section "Getting results in $format format"
    
    url="$API_URL/results?format=$format"
    if [ ! -z "$metric" ]; then
        url="$url&metric=$metric"
    fi
    if [ ! -z "$threshold" ]; then
        url="$url&threshold=$threshold"
    fi
    
    info "Request URL: $url"
    curl -s "$url" | json_pp
}

# Main script execution
echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}    Plagiarism Checker API Test Script       ${NC}"
echo -e "${BLUE}==============================================${NC}"

# Step 1: Check if API is running
check_api

# Step 2: Check initial API status
check_status

# Step 3: Submit test documents
section "Submitting test documents"

# Document 1 - Essay about artificial intelligence (very similar to doc2)
submit_document "essay1" "Artificial intelligence (AI) is intelligence demonstrated by machines, 
as opposed to the natural intelligence displayed by animals and humans. 
AI research has been defined as the field of study of intelligent agents, 
which refers to any system that perceives its environment and takes actions 
that maximize its chance of achieving its goals."

# Document 2 - Almost identical to doc1 with minor changes
submit_document "essay2" "Artificial intelligence (AI) is intelligence demonstrated by machines, 
unlike the natural intelligence shown by animals and humans. 
AI research has been defined as the field of study of intelligent agents, 
which refers to any system that perceives its surroundings and takes actions 
that maximize its chance of achieving its goals."

# Document 3 - Somewhat similar to doc1/doc2
submit_document "essay3" "Artificial intelligence involves creating computer systems 
that can perform tasks typically requiring human intelligence. 
The field focuses on developing agents that perceive their environment 
and make decisions to achieve specific objectives. Modern AI encompasses 
machine learning, natural language processing, and computer vision."

# Document 4 - Completely different topic
submit_document "essay4" "Climate change refers to long-term shifts in temperatures and weather patterns. 
These shifts may be natural, such as through variations in the solar cycle. 
But since the 1800s, human activities have been the main driver of climate change, 
primarily due to burning fossil fuels like coal, oil and gas, which produces 
heat-trapping gases."

# Step 4: Check API status after document submission
check_status

# Step 5: Compare document pairs to generate similarity scores
section "Comparing document pairs"

compare_documents "essay1" "essay2" "cosine"
compare_documents "essay1" "essay3" "cosine"
compare_documents "essay1" "essay4" "cosine"
compare_documents "essay2" "essay3" "cosine"
compare_documents "essay2" "essay4" "cosine"
compare_documents "essay3" "essay4" "cosine"

# Jaccard metric comparisons
compare_documents "essay1" "essay2" "jaccard"
compare_documents "essay1" "essay3" "jaccard"
compare_documents "essay1" "essay4" "jaccard"

# Step 6: Batch comparison
section "Testing batch comparison"
pairs_json='[["essay1","essay2"],["essay1","essay3"],["essay1","essay4"],["essay2","essay3"],["essay2","essay4"],["essay3","essay4"]]'
compare_batch "$pairs_json" "cosine"

# Step 7: Test threshold filtering in results endpoint
section "Testing threshold filtering in results endpoint"

# Get all results with no threshold (baseline)
get_results "pairs"

# Get results with high threshold (0.8) to find very similar documents
get_results "pairs" "" "0.8"

# Get results with medium threshold (0.5) to find somewhat similar documents
get_results "pairs" "" "0.5"

# Get results with low threshold (0.1) to find most documents
get_results "pairs" "" "0.1"

# Get matrix results with threshold filtering
get_results "matrix" "cosine" "0.7"

# Get results for a specific metric with threshold
get_results "pairs" "cosine" "0.7"

# Step 8: Show summary
section "Test Summary"
echo -e "The script has successfully:"
echo -e "  - Verified the API is running"
echo -e "  - Submitted 4 test documents with varying similarity"
echo -e "  - Compared document pairs with different metrics"
echo -e "  - Performed batch comparison of multiple document pairs"
echo -e "  - Demonstrated threshold filtering with different values"
echo -e "  - Tested threshold filtering with different output formats"