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
