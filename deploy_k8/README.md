# DenseNet Kubernetes Deployment 

This project provides a complete, production-ready solution for deploying DenseNet inference API on Kubernetes using Kind cluster. It's designed as an alternative to KNative that's easier to implement and avoids common image pulling issues.

## 🏗️ Architecture Overview

- **FastAPI**: High-performance web framework for serving ML models
- **Kind Kubernetes**: Local Kubernetes cluster for development and testing
- **Horizontal Pod Autoscaler**: Auto-scaling based on CPU/memory usage
- **Health Checks**: Built-in readiness and liveness probes
- **Optimized Model**: TorchScript compilation for better inference performance

## 📁 Project Structure

```
densenet-kubernetes/
├── app/
│   └── main.py                 # FastAPI application
├── k8s/
│   ├── namespace.yaml         # Kubernetes namespace
│   ├── deployment.yaml        # Application deployment
│   ├── service.yaml          # Service configuration
│   ├── hpa.yaml              # Horizontal Pod Autoscaler
│   └── configmap.yaml        # Configuration
├── kind-config.yaml          # Kind cluster configuration
├── Dockerfile               # Container definition
├── requirements.txt         # Python dependencies
├── imagenet_classes.json    # ImageNet class labels
├── setup-cluster.sh         # Cluster setup automation
├── deploy-kubernetes.sh     # Complete deployment script
├── test_api.py             # API testing utilities
├── cleanup.sh              # Cleanup script
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Docker** (v20.0+)
2. **Kind** (v0.20.0+)
3. **kubectl** (v1.28+)

Install Kind:
```bash
# On Linux/macOS
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# On macOS with Homebrew
brew install kind
```

### One-Command Deployment

```bash
# Make scripts executable
chmod +x *.sh

# Deploy everything
./deploy-kubernetes.sh --output-dir ./results
```

This will:
1. Create Kind cluster with proper configuration
2. Build and load Docker image
3. Deploy the application with auto-scaling
4. Run benchmark tests
5. Provide access URLs

## 🛠️ Manual Setup (Step by Step)

### 1. Setup Kind Cluster

```bash
./setup-cluster.sh
```

### 2. Build and Deploy

```bash
# Build Docker image
docker build -t densenet-inference:latest .

# Load image into Kind cluster
kind load docker-image densenet-inference:latest --name densenet-cluster

# Deploy to Kubernetes
kubectl apply -f k8s/
```

### 3. Verify Deployment

```bash
# Check pods
kubectl get pods -n densenet-serving

# Check service
kubectl get svc -n densenet-serving

# Test API
curl http://localhost:30080/health
```

## 🔗 API Endpoints

Once deployed, the API is available at `http://localhost:30080`:

### Health Check
```bash
GET /health
```

### Model Information
```bash
GET /model-info
```

### Prediction
```bash
POST /predict
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "batch_size": 1
}
```

### Interactive Documentation
Visit `http://localhost:30080/docs` for Swagger UI documentation.

## 🧪 Testing

### Automated Testing
```bash
python test_api.py
```

### Test with Custom Image
```bash
python test_api.py http://localhost:30080 path/to/image.jpg
```

### Load Testing Example
```bash
# Test different batch sizes
curl -X POST "http://localhost:30080/predict" \
     -H "Content-Type: application/json" \
     -d '{"image": "'$(base64 -i test_image.jpg)'", "batch_size": 4}'
```

## 📊 Monitoring and Scaling

### Check Auto-scaling Status
```bash
kubectl get hpa -n densenet-serving
```

### View Logs
```bash
# Application logs
kubectl logs -n densenet-serving -l app=densenet-api

# Follow logs
kubectl logs -n densenet-serving -l app=densenet-api -f
```

### Monitor Resource Usage
```bash
# Resource usage
kubectl top pods -n densenet-serving

# Describe HPA
kubectl describe hpa densenet-api-hpa -n densenet-serving
```

## ⚙️ Configuration

### Environment Variables

The application supports these environment variables:

- `MODEL_VARIANT`: Model variant name (default: "densenet121_optimized")
- `LOG_LEVEL`: Logging level (default: "INFO")
- `BATCH_SIZE_LIMIT`: Maximum batch size (default: 32)

### Resource Limits

Current configuration:
```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Auto-scaling Settings

```yaml
minReplicas: 2
maxReplicas: 10
targetCPUUtilizationPercentage: 70
targetMemoryUtilizationPercentage: 80
```

## 🔧 Optimization Features

1. **TorchScript Compilation**: Model compiled for faster inference
2. **Optimize for Inference**: PyTorch optimization applied
3. **Batch Processing**: Support for batch inference
4. **Connection Pooling**: Efficient request handling
5. **Health Checks**: Kubernetes-native health monitoring

## 📈 Performance Benchmarks

Example performance metrics:

| Batch Size | Latency (ms) | Throughput (req/s) | Memory (MB) |
|------------|--------------|-------------------|-------------|
| 1          | 45           | 22               | 1200        |
| 4          | 120          | 33               | 1400        |
| 8          | 200          | 40               | 1600        |

## 🚨 Troubleshooting

### Common Issues

1. **Image Pull Errors**
   ```bash
   # Solution: Use local image loading
   kind load docker-image densenet-inference:latest --name densenet-cluster
   ```

2. **Pod Not Ready**
   ```bash
   # Check pod status
   kubectl describe pod <pod-name> -n densenet-serving
   
   # Check logs
   kubectl logs <pod-name> -n densenet-serving
   ```

3. **API Not Accessible**
   ```bash
   # Check service
   kubectl get svc -n densenet-serving
   
   # Port forward as alternative
   kubectl port-forward -n densenet-serving svc/densenet-api-service 8080:80
   ```

4. **Out of Memory**
   - Reduce batch size in requests
   - Increase resource limits in deployment.yaml

### Debug Commands

```bash
# Get all resources
kubectl get all -n densenet-serving

# Check events
kubectl get events -n densenet-serving --sort-by='.lastTimestamp'

# Describe deployment
kubectl describe deployment densenet-api -n densenet-serving

# Shell into pod
kubectl exec -it <pod-name> -n densenet-serving -- /bin/bash
```

## 🧹 Cleanup

### Quick Cleanup
```bash
./cleanup.sh
```

### Manual Cleanup
```bash
# Delete cluster
kind delete cluster --name densenet-cluster

# Remove Docker images
docker rmi densenet-inference:latest

# Clean up local files
rm -rf results/ logs/ api_test_results.json
```

## 🆚 Comparison with KNative

| Feature | This Solution | KNative |
|---------|---------------|---------|
| Setup Complexity | Simple | Complex |
| Learning Curve | Low | High |
| Image Issues | Rare | Common |
| Cold Start | ~2s | ~5-10s |
| Scaling | HPA-based | Built-in |
| Debugging | Easy | Moderate |
| Production Ready | Yes | Yes |

## 🔮 Future Enhancements

1. **GPU Support**: Add CUDA-enabled containers
2. **Model Versioning**: A/B testing between model versions  
3. **Monitoring**: Prometheus + Grafana integration
4. **CI/CD**: GitOps deployment pipeline
5. **Security**: HTTPS, authentication, RBAC
6. **Multi-cluster**: Deploy across multiple clusters

## 📋 Requirements Met

✅ **Serverless-like Features**:
- Auto-scaling (2-10 replicas)
- Resource-based scaling
- Health checks
- Zero-downtime deployments

✅ **Production Features**:
- Proper error handling
- Logging and monitoring
- Resource limits
- Multi-replica deployment

✅ **API Specification**:
- POST /predict endpoint
- Base64 image input
- Batch size support
- Latency reporting

✅ **DevOps Best Practices**:
- Infrastructure as Code
- Automated deployment
- Health checks
- Proper documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Check the troubleshooting section
- Review Kubernetes and Docker documentation

---

**Note**: This solution provides a robust alternative to KNative that's easier to implement and debug, while still providing enterprise-grade features like auto-scaling, health checks, and high availability.