# DenseNet KNative Serverless Deployment

This solution provides a complete serverless deployment of DenseNet-121 inference using KNative on a local Kind Kubernetes cluster. It solves the common Docker registry issues by using local image loading.

## 🎯 Features

- **Serverless Deployment**: Auto-scaling with scale-to-zero capability
- **Local Development**: No need for Docker registry push/pull
- **Production Ready**: Health checks, resource limits, proper logging
- **Easy Testing**: Automated test client included
- **Complete Automation**: One-command setup and deployment

## 📁 Project Structure

```
knative-deployment/
├── kind-config.yaml          # Kind cluster configuration
├── setup-cluster.sh          # Cluster setup script
├── Dockerfile.knative        # Lightweight Docker image
├── requirements.txt          # Python dependencies
├── app.py                   # Flask API server
├── model_utils.py           # Model optimization utilities
├── knative-service.yaml     # KNative service manifest
├── deploy-knative.sh        # Deployment script
├── test_client.py           # Test client
├── cleanup.sh               # Cleanup script
└── README-KNative.md        # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Docker** - Make sure Docker is running
2. **Kind** - Install from https://kind.sigs.k8s.io/docs/user/quick-start/#installation
3. **kubectl** - Install from https://kubernetes.io/docs/tasks/tools/install-kubectl/

### Step-by-Step Setup

1. **Setup the Kind cluster with KNative**:
```bash
chmod +x setup-cluster.sh
./setup-cluster.sh
```

2. **Deploy the DenseNet service**:
```bash
chmod +x deploy-knative.sh
./deploy-knative.sh
```

3. **Test the deployment**:
```bash
python3 test_client.py
```

4. **Cleanup when done**:
```bash
chmod +x cleanup.sh
./cleanup.sh
```

## 🔧 How It Works

### The Registry Issue Solution

The error you were getting happened because Kubernetes was trying to pull `densenet-inference:latest` from Docker Hub, but the image only existed locally. This solution fixes it by:

1. **Building the image locally**: `docker build -t densenet-inference:latest .`
2. **Loading into Kind cluster**: `kind load docker-image densenet-inference:latest --name cluster-name`
3. **Using `imagePullPolicy: Never`** in the KNative service manifest

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Test Client   │───▶│  KNative Service │───▶│  DenseNet Pod   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   Auto Scaler    │
                       │  (Scale to Zero) │
                       └──────────────────┘
```

### Key Components

- **Flask API**: Lightweight REST API for inference
- **Model Optimization**: TorchScript tracing for better performance
- **Health Checks**: Readiness and liveness probes
- **Auto-scaling**: Scale from 0 to 3 replicas based on demand
- **Resource Management**: CPU/Memory limits and requests

## 🌐 API Endpoints

### Health Check
```bash
curl http://localhost:8080/health
```

### Root Information
```bash
curl http://localhost:8080/
```

### Prediction
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_here",
    "batch_size": 1
  }'
```

## 📊 Response Format

```json
{
  "predictions": [
    {
      "class_id": 281,
      "class_name": "tabby_cat",
      "confidence": 0.8934
    }
  ],
  "latency_ms": 150.25,
  "model_variant": "densenet121_optimized",
  "batch_size": 1
}
```

## 🔍 Monitoring & Debugging

### View Service Status
```bash
kubectl get ksvc densenet-inference
```

### Check Pod Status
```bash
kubectl get pods -l serving.knative.dev/service=densenet-inference
```

### View Logs
```bash
kubectl logs -l serving.knative.dev/service=densenet-inference -f
```

### Describe Service (for debugging)
```bash
kubectl describe ksvc densenet-inference
```

## ⚙️ Configuration

### Auto-scaling Configuration
The service is configured with:
- **Min Scale**: 0 (scale to zero when no requests)
- **Max Scale**: 3 replicas
- **Scale Down Delay**: 30 seconds
- **Target Concurrency**: 10 requests per replica

### Resource Limits
- **CPU**: 100m request, 1000m limit
- **Memory**: 512Mi request, 2Gi limit

### Modify Scaling Behavior
Edit `knative-service.yaml`:
```yaml
annotations:
  autoscaling.knative.dev/minScale: "0"
  autoscaling.knative.dev/maxScale: "5"
  autoscaling.knative.dev/target: "20"
```

## 🐛 Troubleshooting

### Common Issues

1. **Service not starting**:
   ```bash
   kubectl describe ksvc densenet-inference
   kubectl logs -l serving.knative.dev/service=densenet-inference
   ```

2. **Image pull errors**:
   - Make sure you ran `kind load docker-image densenet-inference:latest --name densenet-knative-cluster`
   - Check that `imagePullPolicy: Never` is set in the service manifest

3. **Health check failures**:
   - Check if the Flask app is starting correctly
   - Verify port 8080 is exposed and the health endpoint responds

4. **Cluster not found**:
   ```bash
   kind get clusters
   kubectl cluster-info --context kind-densenet-knative-cluster
   ```

### Debug Commands

```bash
# Check KNative serving status
kubectl get pods -n knative-serving

# Check if Kourier is running
kubectl get pods -n kourier-system

# Get service URL
kubectl get ksvc densenet-inference -o jsonpath='{.status.url}'

# Check service events
kubectl get events --sort-by=.metadata.creationTimestamp
```

## 🎛️ Advanced Usage

### Custom Model Loading
Modify `model_utils.py` to load your own optimized model:

```python
def load_custom_model(device):
    model = torch.jit.load('path/to/optimized/model.pt')
    return model.to(device)
```

### Multiple Model Versions
Deploy multiple versions using KNative traffic splitting:

```yaml
spec:
  traffic:
  - percent: 80
    revisionName: densenet-inference-00001
  - percent: 20
    revisionName: densenet-inference-00002
```

### GPU Support
To enable GPU support, modify the Dockerfile and service manifest:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
```

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
```

## 📈 Performance Characteristics

- **Cold Start**: ~5-10 seconds (first request after scale-to-zero)
- **Warm Latency**: ~100-200ms per inference
- **Memory Usage**: ~512MB base, up to 2GB under load
- **Throughput**: ~10-50 requests/second per replica

## 🔒 Security Considerations

- The service runs without authentication (demo purposes)
- For production, add authentication middleware
- Use secrets for sensitive configuration
- Enable network policies for traffic isolation

## 📚 Learning Resources

- [KNative Documentation](https://knative.dev/docs/)
- [Kind Documentation](https://kind.sigs.k8s.io/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

## 🤝 Contributing

This is a demonstration project. For production use:
1. Add comprehensive error handling
2. Implement authentication and authorization
3. Add monitoring and observability
4. Use production-grade image registries
5. Implement proper CI/CD pipelines

## 📄 License

This project is for educational purposes. Use at your own risk in production environments.

---

**Happy serverless ML deployment! 🚀**