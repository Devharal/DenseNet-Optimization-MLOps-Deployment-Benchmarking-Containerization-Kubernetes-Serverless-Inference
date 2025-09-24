#!/bin/bash
set -e

echo "🚀 Setting up Kind cluster for DenseNet KNative deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if kind is installed
if ! command -v kind &> /dev/null; then
    echo -e "${RED}Kind is not installed. Please install kind first.${NC}"
    echo "Visit: https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}kubectl is not installed. Please install kubectl first.${NC}"
    echo "Visit: https://kubernetes.io/docs/tasks/tools/install-kubectl/"
    exit 1
fi

# Check if docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

echo -e "${YELLOW}Creating Kind cluster...${NC}"
kind create cluster --config=kind-config.yaml --wait=5m

echo -e "${YELLOW}Setting kubectl context...${NC}"
kubectl cluster-info --context kind-densenet-knative-cluster

echo -e "${YELLOW}Installing KNative Serving...${NC}"
# Install KNative Serving CRDs
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.0/serving-crds.yaml

# Wait for CRDs to be established
kubectl wait --for=condition=Established crd/services.serving.knative.dev --timeout=60s

# Install KNative Serving core
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.0/serving-core.yaml

echo -e "${YELLOW}Waiting for KNative Serving to be ready...${NC}"
kubectl wait --for=condition=Available deployment --all -n knative-serving --timeout=3000s

echo -e "${YELLOW}Installing Kourier networking layer...${NC}"
kubectl apply -f https://github.com/knative/net-kourier/releases/download/knative-v1.11.0/kourier.yaml

# Configure KNative to use Kourier
kubectl patch configmap/config-network \
  --namespace knative-serving \
  --type merge \
  --patch '{"data":{"ingress-class":"kourier.ingress.networking.knative.dev"}}'

echo -e "${YELLOW}Waiting for Kourier to be ready...${NC}"
kubectl wait --for=condition=Available deployment 3scale-kourier-gateway -n kourier-system --timeout=300s

echo -e "${YELLOW}Setting up Magic DNS (nip.io)...${NC}"
kubectl patch configmap/config-domain \
  --namespace knative-serving \
  --type merge \
  --patch '{"data":{"127.0.0.1.nip.io":""}}'

echo -e "${GREEN}✅ Kind cluster setup complete!${NC}"
echo -e "${GREEN}Cluster name: densenet-knative-cluster${NC}"
echo -e "${GREEN}KNative Serving is ready for deployment${NC}"

echo -e "${YELLOW}Verifying installation...${NC}"
kubectl get pods -n knative-serving
echo ""
kubectl get pods -n kourier-system
echo ""

echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
echo "You can now proceed with the model deployment."