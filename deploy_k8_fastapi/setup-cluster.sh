#!/bin/bash
# setup-cluster.sh

set -e

echo "🚀 Setting up Kind cluster for DenseNet serving..."

# Colors for output
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

# Check if required tools are installed
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v kind &> /dev/null; then
        print_error "Kind is not installed. Please install Kind first."
        echo "Install with: go install sigs.k8s.io/kind@v0.20.0"
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    print_success "All prerequisites are installed"
}

# Create Kind cluster
create_cluster() {
    print_status "Creating Kind cluster..."
    
    # Delete existing cluster if it exists
    if kind get clusters | grep -q densenet-cluster; then
        print_warning "Existing cluster found. Deleting..."
        kind delete cluster --name densenet-cluster
    fi
    
    # Create new cluster
    kind create cluster --config kind-config.yaml --wait 300s
    
    if [ $? -eq 0 ]; then
        print_success "Kind cluster created successfully"
    else
        print_error "Failed to create Kind cluster"
        exit 1
    fi
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    
    docker build -t densenet-inference:latest .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Load image into Kind cluster
load_image() {
    print_status "Loading Docker image into Kind cluster..."
    
    kind load docker-image densenet-inference:latest --name densenet-cluster
    
    if [ $? -eq 0 ]; then
        print_success "Image loaded into Kind cluster"
    else
        print_error "Failed to load image into Kind cluster"
        exit 1
    fi
}

# Deploy application
deploy_app() {
    print_status "Deploying DenseNet API to Kubernetes..."
    
    # Apply all manifests
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/deployment.yaml
    # kubectl wait --for=condition=available --timeout=300s deployment/densenet-api -n densenet-serving
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/hpa.yaml
    
    # Wait for deployment to be ready
    print_status "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/densenet-api -n densenet-serving
    
    if [ $? -eq 0 ]; then
        print_success "Application deployed successfully"
    else
        print_error "Failed to deploy application"
        exit 1
    fi
}

# Verify deployment
verify_deployment() {
    print_status "Verifying deployment..."
    
    # Check pod status
    kubectl get pods -n densenet-serving
    
    # Check service
    kubectl get svc -n densenet-serving
    
    # Test API endpoint
    print_status "Testing API endpoint..."
    sleep 10
    
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:30080/health)
    
    if [ "$response" = "200" ]; then
        print_success "API endpoint is healthy"
    else
        print_warning "API endpoint returned status: $response"
        print_status "Checking pod logs..."
        kubectl logs -n densenet-serving -l app=densenet-api --tail=20
    fi
}

# Display access information
show_access_info() {
    echo ""
    echo "🎉 Deployment complete!"
    echo ""
    echo "Access Information:"
    echo "===================="
    echo "• API Base URL: http://localhost:30080"
    echo "• Health Check: http://localhost:30080/health"
    echo "• API Docs: http://localhost:30080/docs"
    echo "• Model Info: http://localhost:30080/model-info"
    echo ""
    echo "Useful Commands:"
    echo "================="
    echo "• Check pods: kubectl get pods -n densenet-serving"
    echo "• Check logs: kubectl logs -n densenet-serving -l app=densenet-api"
    echo "• Port forward: kubectl port-forward -n densenet-serving svc/densenet-api-service 8080:80"
    echo "• Delete cluster: kind delete cluster --name densenet-cluster"
    echo ""
}

# Main execution
main() {
    echo "🔧 DenseNet Kubernetes Deployment Setup"
    echo "========================================"
    
    check_prerequisites
    create_cluster
    build_image
    load_image
    deploy_app
    verify_deployment
    show_access_info
}

# Run main function
main "$@"