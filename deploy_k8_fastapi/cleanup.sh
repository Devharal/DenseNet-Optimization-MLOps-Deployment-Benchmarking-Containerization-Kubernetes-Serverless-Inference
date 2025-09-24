#!/bin/bash
# cleanup.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

CLUSTER_NAME="densenet-cluster"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster-name)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        --all)
            DELETE_IMAGES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --cluster-name NAME  Kind cluster name (default: densenet-cluster)"
            echo "  --all               Also delete Docker images"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🧹 Cleaning up DenseNet Kubernetes deployment..."
echo "Cluster: $CLUSTER_NAME"

# Delete Kind cluster
print_status "Deleting Kind cluster: $CLUSTER_NAME"
if kind get clusters | grep -q "$CLUSTER_NAME"; then
    kind delete cluster --name "$CLUSTER_NAME"
    print_success "Cluster deleted successfully"
else
    print_warning "Cluster '$CLUSTER_NAME' not found"
fi

# Clean up Docker images if requested
if [ "$DELETE_IMAGES" = "true" ]; then
    print_status "Cleaning up Docker images..."
    
    # Remove application image
    if docker images | grep -q "densenet-inference"; then
        docker rmi densenet-inference:latest || print_warning "Could not remove densenet-inference:latest"
        print_success "Removed densenet-inference image"
    fi
    
    # Clean up dangling images
    docker image prune -f
    print_success "Cleaned up dangling images"
fi

# Clean up local files (optional)
read -p "Do you want to clean up local result files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Cleaning up local files..."
    
    rm -rf ./results/ || true
    rm -rf ./logs/ || true
    rm -f api_test_results.json || true
    rm -f benchmark_results.json || true
    
    print_success "Local files cleaned up"
fi

print_success "Cleanup completed!"

echo ""
echo "Summary:"
echo "========"
echo "✅ Kind cluster '$CLUSTER_NAME' removed"

echo ""
echo "To verify cleanup:"
echo "• Check clusters: kind get clusters"
echo "• Check images: docker images | grep densenet"