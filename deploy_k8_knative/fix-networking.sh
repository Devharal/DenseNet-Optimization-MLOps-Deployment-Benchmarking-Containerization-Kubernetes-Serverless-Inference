#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Fixing KNative networking issues...${NC}"

# Check if cluster exists
if ! kubectl cluster-info --context kind-densenet-knative-cluster &> /dev/null; then
    echo -e "${RED}❌ Kind cluster not found.${NC}"
    exit 1
fi

echo -e "${YELLOW}🔍 Checking current network setup...${NC}"
kubectl get svc -n kourier-system

# Get the NodePort for Kourier
KOURIER_PORT=$(kubectl get svc kourier -n kourier-system -o jsonpath='{.spec.ports[?(@.port==80)].nodePort}')
echo -e "${BLUE}📡 Kourier is running on NodePort: ${KOURIER_PORT}${NC}"

# Test direct pod access first
echo -e "\n${YELLOW}🧪 Testing direct pod access...${NC}"
POD_NAME=$(kubectl get pods -l serving.knative.dev/service=densenet-inference -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [ ! -z "$POD_NAME" ]; then
    echo "Testing pod directly..."
    kubectl port-forward $POD_NAME 8080:8080 &
    PORT_FORWARD_PID=$!
    sleep 3
    
    echo "Testing health check..."
    if curl -s http://localhost:8080/health > /dev/null; then
        echo -e "${GREEN}✅ Pod is working correctly${NC}"
    else
        echo -e "${RED}❌ Pod health check failed${NC}"
    fi
    
    # Kill port-forward
    kill $PORT_FORWARD_PID 2>/dev/null || true
    wait $PORT_FORWARD_PID 2>/dev/null || true
else
    echo -e "${RED}❌ No pods found for densenet-inference${NC}"
fi

echo -e "\n${YELLOW}🔧 Setting up proper networking...${NC}"

# Create a simple port-forward solution
cat << 'EOF' > start-proxy.sh
#!/bin/bash
echo "🚀 Starting KNative service proxy..."
echo "This will make your service available at http://localhost:8080"

# Get the service pod
POD_NAME=$(kubectl get pods -l serving.knative.dev/service=densenet-inference -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$POD_NAME" ]; then
    echo "❌ No pods found. The service might have scaled to zero."
    echo "Making a request to wake it up..."
    
    # Try to wake up the service
    kubectl get ksvc densenet-inference
    
    # Wait and try again
    sleep 10
    POD_NAME=$(kubectl get pods -l serving.knative.dev/service=densenet-inference -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
fi

if [ ! -z "$POD_NAME" ]; then
    echo "✅ Found pod: $POD_NAME"
    echo "🔗 Service will be available at: http://localhost:8080"
    echo "📋 Endpoints:"
    echo "   - Health: http://localhost:8080/health"
    echo "   - Predict: http://localhost:8080/predict"
    echo "   - Info: http://localhost:8080/"
    echo ""
    echo "Press Ctrl+C to stop the proxy"
    kubectl port-forward $POD_NAME 8080:8080
else
    echo "❌ Still no pods found. Check service status:"
    kubectl get ksvc densenet-inference
    kubectl describe ksvc densenet-inference
fi
EOF

chmod +x start-proxy.sh

echo -e "${GREEN}✅ Networking fix applied!${NC}"
echo -e "${BLUE}📋 How to access your service:${NC}"
echo "1. Run: ./start-proxy.sh"
echo "2. In another terminal, test:"
echo "   curl http://localhost:8080/health"
echo "   curl http://localhost:8080/"
echo ""
echo -e "${YELLOW}💡 The issue was that Kind doesn't expose LoadBalancer services by default.${NC}"
echo -e "${YELLOW}   Port-forwarding is the simplest solution for local testing.${NC}"