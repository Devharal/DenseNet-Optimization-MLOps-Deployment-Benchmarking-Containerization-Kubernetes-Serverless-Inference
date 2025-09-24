#!/bin/bash
# deploy-kubernetes.sh

set -e

# Default values
OUTPUT_DIR="./results"
GPU_ENABLED="false"
CLUSTER_NAME="densenet-cluster"

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
        --cluster-name)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --output-dir DIR     Output directory for results (default: ./results)"
            echo "  --gpu-enabled BOOL   Enable GPU support (default: false)"
            echo "  --cluster-name NAME  Kind cluster name (default: densenet-cluster)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

print_status "Starting Kubernetes deployment..."
print_status "Output directory: $OUTPUT_DIR"
print_status "GPU enabled: $GPU_ENABLED"
print_status "Cluster name: $CLUSTER_NAME"

# Setup cluster
./setup-cluster.sh

# Run benchmarking inside the cluster
print_status "Running benchmarking tests..."

# Create a job to run benchmarks
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: densenet-benchmark
  namespace: densenet-serving
spec:
  template:
    spec:
      containers:
      - name: benchmark
        image: densenet-inference:latest
        imagePullPolicy: Never
        command: ["python"]
        args: ["-c", "
import requests
import json
import time
import base64
from PIL import Image
import io

def create_test_image():
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def benchmark_api():
    base_url = 'http://densenet-api-service.densenet-serving.svc.cluster.local'
    results = []
    
    test_image = create_test_image()
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        print(f'Testing batch size: {batch_size}')
        
        for i in range(5):  # 5 runs per batch size
            try:
                start_time = time.time()
                response = requests.post(
                    f'{base_url}/predict',
                    json={'image': test_image, 'batch_size': batch_size},
                    timeout=60
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    results.append({
                        'batch_size': batch_size,
                        'run': i + 1,
                        'latency_ms': data['latency_ms'],
                        'total_time_ms': (end_time - start_time) * 1000,
                        'model_variant': data['model_variant'],
                        'status': 'success'
                    })
                    print(f'  Run {i+1}: {data[\"latency_ms\"]:.2f}ms')
                else:
                    print(f'  Run {i+1}: Failed with status {response.status_code}')
            except Exception as e:
                print(f'  Run {i+1}: Error - {str(e)}')
    
    # Save results
    with open('/tmp/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('Benchmark completed!')

if __name__ == '__main__':
    benchmark_api()
        "]
        volumeMounts:
        - name: results-volume
          mountPath: /tmp
      volumes:
      - name: results-volume
        hostPath:
          path: $(pwd)/$OUTPUT_DIR
          type: DirectoryOrCreate
      restartPolicy: Never
  backoffLimit: 3
EOF

# Wait for job completion
print_status "Waiting for benchmark job to complete..."
kubectl wait --for=condition=complete --timeout=600s job/densenet-benchmark -n densenet-serving

# Copy results
print_status "Copying benchmark results..."
POD_NAME=$(kubectl get pods -n densenet-serving -l job-name=densenet-benchmark -o jsonpath='{.items[0].metadata.name}')

if [ ! -z "$POD_NAME" ]; then
    kubectl cp "densenet-serving/$POD_NAME:/tmp/benchmark_results.json" "$OUTPUT_DIR/benchmark_results.json" || true
    kubectl logs -n densenet-serving "$POD_NAME" > "$OUTPUT_DIR/benchmark_logs.txt"
fi

# Generate summary
print_status "Generating deployment summary..."
cat > "$OUTPUT_DIR/deployment_summary.txt" << EOF
DenseNet Kubernetes Deployment Summary
=====================================

Cluster Information:
- Cluster Name: $CLUSTER_NAME
- Namespace: densenet-serving
- GPU Enabled: $GPU_ENABLED

Deployment Status:
$(kubectl get all -n densenet-serving)

Service Endpoints:
- API URL: http://localhost:30080
- Health Check: http://localhost:30080/health
- API Documentation: http://localhost:30080/docs

HPA Status:
$(kubectl get hpa -n densenet-serving)

Recent Pod Events:
$(kubectl get events -n densenet-serving --sort-by='.lastTimestamp' | tail -10)
EOF

# Test API availability
print_status "Testing API availability..."
sleep 5

if curl -s http://localhost:30080/health > /dev/null; then
    print_success "API is accessible at http://localhost:30080"
    
    # Get model info
    MODEL_INFO=$(curl -s http://localhost:30080/model-info)
    echo "Model Information:" >> "$OUTPUT_DIR/deployment_summary.txt"
    echo "$MODEL_INFO" >> "$OUTPUT_DIR/deployment_summary.txt"
    
    print_success "Deployment completed successfully!"
else
    print_error "API is not accessible. Check the logs:"
    kubectl logs -n densenet-serving -l app=densenet-api --tail=20
fi

# Cleanup job
kubectl delete job densenet-benchmark -n densenet-serving || true

echo ""
echo "📊 Results saved to: $OUTPUT_DIR"
echo "🔗 Access the API at: http://localhost:30080"
echo "📚 API Documentation: http://localhost:30080/docs"
echo ""
echo "To cleanup: kind delete cluster --name $CLUSTER_NAME"