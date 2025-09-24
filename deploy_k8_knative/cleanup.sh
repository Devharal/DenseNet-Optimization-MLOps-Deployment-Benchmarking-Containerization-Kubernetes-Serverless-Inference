#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🧹 Cleaning up DenseNet KNative deployment...${NC}"

# Delete KNative service
echo -e "${YELLOW}🗑️  Deleting KNative service...${NC}"
kubectl delete -f knative-service.yaml --ignore-not-found=true

# Wait for cleanup
echo -e "${YELLOW}⏳ Waiting for resources to be cleaned up...${NC}"
sleep 10

# Delete the Kind cluster
echo -e "${YELLOW}💥 Deleting Kind cluster...${NC}"
kind delete cluster --name densenet-knative-cluster

# Clean up Docker images (optional)
read -p "Do you want to remove the Docker image? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🗑️  Removing Docker image...${NC}"
    docker rmi densenet-inference:latest --force || true
fi

echo -e "${GREEN}✅ Cleanup completed!${NC}"
echo "All resources have been removed."