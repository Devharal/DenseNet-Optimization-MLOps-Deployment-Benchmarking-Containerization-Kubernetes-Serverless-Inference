#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚢 Deploying DenseNet to KNative...${NC}"

# Check if cluster exists
if ! kubectl cluster-info --context kind-densenet-knative-cluster &> /dev/null; then
    echo -e "${RED}❌ Kind cluster not found. Please run setup-cluster.sh first.${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# echo -e "${YELLOW}📦 Building Docker image...${NC}"
# # Build the Docker image
# docker build -f Dockerfile.knative -t densenet-inference:latest .

echo -e "${YELLOW}📥 Loading image into Kind cluster...${NC}"
# Load the image into Kind cluster (this solves the registry issue!)
kind load docker-image devharal/densenet-inference:latest --name densenet-knative-cluster

echo -e "${YELLOW}🚀 Deploying KNative service...${NC}"
# Deploy the KNative service
kubectl apply -f knative-service.yaml

echo -e "${YELLOW}⏳ Waiting for service to be ready...${NC}"
# Wait for the service to be ready
kubectl wait --for=condition=Ready ksvc/densenet-inference --timeout=3000s

echo -e "${GREEN}✅ Deployment successful!${NC}"

# Get the service URL
echo -e "${BLUE}📡 Service Information:${NC}"
SERVICE_URL=$(kubectl get ksvc densenet-inference -o jsonpath='{.status.url}')
echo -e "${GREEN}Service URL: ${SERVICE_URL}${NC}"

# Show service status
echo -e "\n${BLUE}📊 Service Status:${NC}"
kubectl get ksvc densenet-inference

echo -e "\n${BLUE}🔍 Pod Status:${NC}"
kubectl get pods -l serving.knative.dev/service=densenet-inference

echo -e "\n${BLUE}📋 Logs (last 10 lines):${NC}"
POD_NAME=$(kubectl get pods -l serving.knative.dev/service=densenet-inference -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [ ! -z "$POD_NAME" ]; then
    kubectl logs $POD_NAME --tail=10
else
    echo "No pods running yet (service may be scaled to zero)"
fi

echo -e "\n${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "${BLUE}💡 Tips:${NC}"
echo "1. Test the service: curl -X POST $SERVICE_URL/predict -H 'Content-Type: application/json' -d '{\"image\": \"base64_image_here\"}'"
echo "2. Health check: curl $SERVICE_URL/health"
echo "3. View logs: kubectl logs -l serving.knative.dev/service=densenet-inference -f"
echo "4. Scale to zero after inactivity (auto-scaling enabled)"