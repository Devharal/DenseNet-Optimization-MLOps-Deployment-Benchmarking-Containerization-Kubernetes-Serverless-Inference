#!/bin/bash

# DenseNet Optimization Benchmarking Suite
# Build and Run Script
# 
# This script builds the Docker container and runs the complete benchmarking suite
# with proper volume mounts and TensorBoard visualization.

set -e  # Exit on any error

# Default values
OUTPUT_DIR="./results"
GPU_ENABLED="true"
BUILD_NO_CACHE=false
CLEANUP=false
DETACH=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR     Output directory for results (default: ./results)"
    echo "  --gpu-enabled BOOL   Enable GPU benchmarking (default: true)"
    echo "  --no-cache          Build Docker image without cache"
    echo "  --cleanup           Clean up containers and images after run"
    echo "  --detach            Run in detached mode"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --output-dir ./my_results --gpu-enabled true"
    echo "  $0 --no-cache --cleanup"
    echo "  $0 --detach"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpu-enabled)
            GPU_ENABLED="$2"
            shift 2
            ;;
        --no-cache)
            BUILD_NO_CACHE=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --detach)
            DETACH=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker daemon."
        exit 1
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose."
        exit 1
    fi
    
    # Check GPU requirements if GPU is enabled
    if [ "$GPU_ENABLED" = "true" ]; then
        if ! command -v nvidia-smi &> /dev/null; then
            print_warning "nvidia-smi not found. GPU benchmarking may not work properly."
        else
            print_status "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
        fi
        
        # Check if nvidia-container-runtime is available
        if ! docker info 2>/dev/null | grep -q nvidia; then
            print_warning "NVIDIA Container Runtime not detected. GPU access may not work."
        fi
    fi
    
    print_success "Prerequisites check completed"
}

# Function to create output directories
setup_directories() {
    print_status "Setting up output directories..."
    
    # Create output directory structure
    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}/profiles"
    mkdir -p "${OUTPUT_DIR}/models" 
    mkdir -p "./logs/tensorboard"
    
    # Set proper permissions
    chmod -R 755 "${OUTPUT_DIR}"
    chmod -R 755 "./logs"
    
    print_success "Directories created: ${OUTPUT_DIR}, ./logs"
}

# Function to clean up old containers and images
cleanup_old() {
    print_status "Cleaning up old containers..."
    
    # Stop and remove containers if they exist
    docker-compose -f docker-compose.yml down --remove-orphans 2>/dev/null || true
    
    # Remove dangling images
    docker image prune -f 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image..."
    
    local build_args=""
    if [ "$BUILD_NO_CACHE" = true ]; then
        build_args="--no-cache"
    fi
    
    # Build the image
    if ! docker-compose -f docker-compose.yml build $build_args; then
        print_error "Failed to build Docker image"
        exit 1
    fi
    
    print_success "Docker image built successfully"
}

# Function to start TensorBoard service
start_tensorboard() {
    print_status "Starting TensorBoard service..."
    
    # Start TensorBoard in detached mode
    if ! docker-compose -f docker-compose.yml up -d tensorboard; then
        print_error "Failed to start TensorBoard service"
        exit 1
    fi
    
    # Wait for TensorBoard to be ready
    print_status "Waiting for TensorBoard to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:6006 > /dev/null 2>&1; then
            break
        fi
        print_status "Attempt $attempt/$max_attempts - waiting for TensorBoard..."
        sleep 2
        ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        print_warning "TensorBoard may not be fully ready, but continuing..."
    else
        print_success "TensorBoard is running at http://localhost:6006"
    fi
}

# Function to run benchmarking
run_benchmark() {
    print_status "Starting DenseNet benchmarking suite..."
    
    # Set environment variables for the container
    export OUTPUT_DIR_MOUNT=$(realpath "${OUTPUT_DIR}")
    export LOGS_DIR_MOUNT=$(realpath "./logs")
    
    # Run the benchmark container
    local run_args=""
    if [ "$DETACH" = true ]; then
        run_args="-d"
    fi
    
    # Update docker-compose.yml with current paths
    sed -i.bak "s|./results|${OUTPUT_DIR_MOUNT}|g" docker-compose.yml
    sed -i "s|./logs|${LOGS_DIR_MOUNT}|g" docker-compose.yml
    
    # Run the benchmarking container
    if [ "$DETACH" = true ]; then
        print_status "Running benchmark in detached mode..."
        if ! docker-compose -f docker-compose.yml up -d densenet-benchmark; then
            print_error "Failed to start benchmarking container"
            exit 1
        fi
        
        print_success "Benchmark started in background"
        print_status "You can monitor progress with: docker-compose -f docker-compose.yml logs -f densenet-benchmark"
        
    else
        print_status "Running benchmark (this may take 15-30 minutes)..."
        if ! docker-compose -f docker-compose.yml up densenet-benchmark; then
            print_error "Benchmark failed"
            exit 1
        fi
    fi
    
    # Restore original docker-compose.yml
    mv docker-compose.yml.bak docker-compose.yml
}

# Function to display results
show_results() {
    if [ "$DETACH" = true ]; then
        print_status "Benchmark is running in background. Results will be available in ${OUTPUT_DIR}"
        return
    fi
    
    print_status "Displaying benchmark results..."
    
    # Check if results file exists
    local results_file="${OUTPUT_DIR}/benchmark_results.csv"
    if [ -f "$results_file" ]; then
        print_success "Benchmark completed successfully!"
        print_status "Results saved to: $results_file"
        
        # Show quick summary
        echo ""
        echo "=== QUICK RESULTS SUMMARY ==="
        if command -v python3 &> /dev/null; then
            docker run --rm -v "${OUTPUT_DIR}:/results" python:3.10-slim bash -c "pip install pandas > /dev/null 2>&1 && python -c \"
import pandas as pd
try:
    df = pd.read_csv('/results/benchmark_results.csv')
    print(f'Total configurations tested: {len(df)}')
    print(f'Optimization techniques: {df[\"optimization_technique\"].nunique()}')
    print(f'Batch sizes tested: {sorted(df[\"batch_size\"].unique())}')
    
    # Best performance metrics
    batch_1 = df[df['batch_size'] == 1]
    if not batch_1.empty:
        best_latency = batch_1.loc[batch_1['latency_ms'].idxmin()]
        best_throughput = batch_1.loc[batch_1['throughput_samples_sec'].idxmax()]
        smallest_model = batch_1.loc[batch_1['model_size_mb'].idxmin()]
        
        print(f'\\nBest latency: {best_latency[\"latency_ms\"]:.2f} ms ({best_latency[\"optimization_technique\"]})')
        print(f'Best throughput: {best_throughput[\"throughput_samples_sec\"]:.2f} samples/sec ({best_throughput[\"optimization_technique\"]})')
        print(f'Smallest model: {smallest_model[\"model_size_mb\"]:.2f} MB ({smallest_model[\"optimization_technique\"]})')
except Exception as e:
    print(f'Could not analyze results: {e}')
\""
        else
            print_status "Install Python3 and pandas to see detailed summary"
        fi
        
        echo "==============================="
        echo ""
        print_status "View detailed results in TensorBoard: http://localhost:6006"
        print_status "Profile traces available in: ${OUTPUT_DIR}/profiles/"
        
    else
        print_error "Results file not found: $results_file"
        print_status "Check container logs: docker-compose -f docker-compose.yml logs densenet-benchmark"
    fi
}

# Function to cleanup resources
cleanup_resources() {
    if [ "$CLEANUP" = true ]; then
        print_status "Cleaning up resources..."
        
        # Stop all services
        docker-compose -f docker-compose.yml down --remove-orphans
        
        # Remove images
        docker rmi $(docker images -q "*densenet*" 2>/dev/null) 2>/dev/null || true
        
        # Clean up build cache
        docker builder prune -f 2>/dev/null || true
        
        print_success "Cleanup completed"
    else
        print_status "To stop services: docker-compose -f docker-compose.yml down"
        print_status "To clean up: docker-compose -f docker-compose.yml down && docker image prune"
    fi
}

# Function to handle script interruption
handle_interrupt() {
    print_warning "Script interrupted. Cleaning up..."
    docker-compose -f docker-compose.yml down 2>/dev/null || true
    exit 130
}

# Main execution
main() {
    # Set trap for interruption
    trap handle_interrupt SIGINT SIGTERM
    
    print_status "=== DenseNet Optimization Benchmarking Suite ==="
    print_status "Output Directory: $OUTPUT_DIR"
    print_status "GPU Enabled: $GPU_ENABLED"
    echo ""
    
    # Execute pipeline
    # check_prerequisites
    # setup_directories
    cleanup_old
    build_image
    start_tensorboard
    run_benchmark
    
    if [ "$DETACH" = false ]; then
        show_results
    fi
    
    cleanup_resources
    
    print_success "=== Benchmarking pipeline completed ==="
    
    if [ "$DETACH" = false ]; then
        echo ""
        print_status "Next steps:"
        echo "  1. Review results in ${OUTPUT_DIR}/benchmark_results.csv"
        echo "  2. Explore TensorBoard visualizations at http://localhost:6006"
        echo "  3. Check detailed profiling traces in ${OUTPUT_DIR}/profiles/"
        echo "  4. Stop TensorBoard: docker-compose -f docker-compose.yml down"
    fi
}

# Run main function
main "$@"