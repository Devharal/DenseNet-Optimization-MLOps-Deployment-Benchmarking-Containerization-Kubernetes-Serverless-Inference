#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🚀 DenseNet KNative Complete Deployment Script${NC}"
echo -e "${PURPLE}==============================================${NC}"

# Function to check prerequisites
check_prerequisites() {
    echo -e "${BLUE}🔍 Checking prerequisites...${NC}"
    
    local missing_tools=()
    
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    if ! command -v kind &> /dev/null; then
        missing_tools+=("kind")
    fi
    
    if ! command -v kubectl &> /dev/null; then
        missing_tools+=("kubectl")
    fi
    
    if ! command -v python3 &> /dev/null; then
        missing_tools+=("python3")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        echo -e "${RED}❌ Missing tools: ${missing_tools[*]}${NC}"
        echo "Please install the missing tools and try again."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All prerequisites are satisfied${NC}"
}

# Function to setup cluster
setup_cluster() {
    echo -e "\n${BLUE}🏗️  Setting up Kind cluster...${NC}"
    
    if kind get clusters 2>/dev/null | grep -q "densenet-knative-cluster"; then
        echo -e "${YELLOW}⚠️  Cluster already exists. Deleting and recreating...${NC}"
        kind delete cluster --name densenet-knative-cluster
    fi
    
    ./setup-cluster.sh
}

# Function to build and deploy
build_and_deploy() {
    echo -e "\n${BLUE}🔨 Building and deploying service...${NC}"
    ./deploy-knative.sh
}

# Function to test deployment
test_deployment() {
    echo -e "\n${BLUE}🧪 Testing deployment...${NC}"
    
    # Install required Python packages for testing
    if ! python3 -c "import requests" &> /dev/null; then
        echo -e "${YELLOW}📦 Installing required packages for testing...${NC}"
        pip3 install requests pillow numpy --user --quiet
    fi
    
    python3 test_client.py
}

# Function to show final status
show_status() {
    echo -e "\n${BLUE}📊 Final Status${NC}"
    echo "===================="
    
    SERVICE_URL=$(kubectl get ksvc densenet-inference -o jsonpath='{.status.url}' 2>/dev/null || echo "Not available")
    
    echo -e "${GREEN}🔗 Service URL: $SERVICE_URL${NC}"
    echo -e "${GREEN}🏥 Health Check: $SERVICE_URL/health${NC}"
    echo -e "${GREEN}🧠 Prediction: $SERVICE_URL/predict${NC}"
    
    echo -e "\n${BLUE}📋 Service Status:${NC}"
    kubectl get ksvc densenet-inference
    
    echo -e "\n${BLUE}🔍 Pod Status:${NC}"
    kubectl get pods -l serving.knative.dev/service=densenet-inference
}

# Function to cleanup on failure
cleanup_on_failure() {
    echo -e "\n${RED}💥 Deployment failed. Cleaning up...${NC}"
    ./cleanup.sh 2>/dev/null || true
}

# Main execution
main() {
    # Set trap for cleanup on failure
    trap cleanup_on_failure ERR
    
    echo -e "${YELLOW}Starting complete deployment process...${NC}"
    
    # Step 1: Check prerequisites
    check_prerequisites
    
    # Step 2: Setup cluster
    setup_cluster
    
    # Step 3: Build and deploy
    build_and_deploy
    
    # Step 4: Test deployment
    test_deployment
    
    # Step 5: Show final status
    show_status
    
    echo -e "\n${GREEN}🎉 Complete deployment successful!${NC}"
    echo -e "${PURPLE}===========================================${NC}"
    echo -e "${GREEN}✅ DenseNet is now running serverless on KNative${NC}"
    echo -e "${GREEN}✅ Auto-scaling is enabled (scale-to-zero)${NC}"
    echo -e "${GREEN}✅ Health checks are configured${NC}"
    echo -e "${GREEN}✅ API is ready for inference requests${NC}"
    
    echo -e "\n${BLUE}💡 Next Steps:${NC}"
    echo "1. Make inference requests to: $(kubectl get ksvc densenet-inference -o jsonpath='{.status.url}' 2>/dev/null)"
    echo "2. Monitor with: kubectl logs -l serving.knative.dev/service=densenet-inference -f"
    echo "3. Scale behavior: kubectl get pods -l serving.knative.dev/service=densenet-inference -w"
    echo "4. Cleanup when done: ./cleanup.sh"
}

# Handle command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Deploy DenseNet inference service on KNative"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --no-test      Skip testing after deployment"
    echo ""
    echo "Prerequisites:"
    echo "  - Docker (running)"
    echo "  - Kind"
    echo "  - kubectl"
    echo "  - Python 3"
    echo ""
    exit 0
fi

# Check for no-test flag
if [ "$1" = "--no-test" ]; then
    test_deployment() {
        echo -e "${YELLOW}⏭️  Skipping tests as requested${NC}"
    }
fi

# Run main function
main