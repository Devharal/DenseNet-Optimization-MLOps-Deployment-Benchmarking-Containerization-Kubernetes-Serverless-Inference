# DenseNet Optimization & Benchmarking Suite

A comprehensive MLOps solution for benchmarking and optimizing DenseNet-121 architecture for production deployment, featuring containerized workflows, automated benchmarking, and serverless deployment capabilities.
![Architecture](02-k8s-architecture-1.gif)

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Usage Guide](#usage-guide)
- [Optimization Approaches](#optimization-approaches)
- [Results Analysis](#results-analysis)
- [Serverless Deployment](#serverless-deployment)
- [Performance Benchmarks](#performance-benchmarks)
- [Trade-offs Discussion](#trade-offs-discussion)
- [Known Limitations](#known-limitations)
- [Deployment with KNative](#deployment-with-knative)

## Project Overview

This project implements a complete MLOps pipeline for DenseNet-121 optimization, featuring:

- **Comprehensive Benchmarking**: Automated profiling using PyTorch Profiler and TensorBoard
- **Multiple Optimization Techniques**: Quantization, pruning, knowledge distillation, and TensorRT optimization
- **Containerized Deployment**: Docker and Docker Compose for reproducible environments
- **Serverless Architecture**: KNative deployment on Kubernetes for auto-scaling inference
- **Production-Ready Monitoring**: Prometheus metrics, health checks, and observability

## Features

### Core Capabilities
- **Multi-Technique Optimization**: Compare baseline vs. optimized models
- **Automated Benchmarking**: Single-command execution of complete benchmark suite
- **Comprehensive Profiling**: Memory usage, latency, throughput, and accuracy metrics
- **Scalable Infrastructure**: Kubernetes-based serverless deployment
- **Production Monitoring**: Health checks, metrics, and logging

### Optimization Techniques
1. **Dynamic Quantization**: Reduce model size and improve inference speed
2. **Structured Pruning**: Remove redundant parameters while maintaining accuracy
3. **Knowledge Distillation**: Create compact student models
4. **TensorRT Optimization**: GPU acceleration with optimized kernels

## Setup Instructions

### Project Structure
```
├── main.py                 # Core benchmarking logic
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-service orchestration
├── requirements.txt       # Python dependencies
├── build_and_run.sh      # Main execution script
└── README.md
├── deploy_k8_fastapi/
└── deploy_k8_knative/
```
### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Docker Runtime (for GPU support)
- 8GB+ available RAM
- 20GB+ available disk space

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd densenet-optimization
   ```

2. **Make scripts executable**
   ```bash
   chmod +x build_and_run.sh
   ```

3. **Run the complete benchmark suite**
   ```bash
   ./build_and_run.sh --output-dir ./results --gpu-enabled true
   ```


### Advanced Setup

For custom configurations:

```bash
# CPU-only benchmarking
./build_and_run.sh --gpu-enabled false --output-dir ./cpu_results

# Build without cache
./build_and_run.sh --no-cache --cleanup

# Run in detached mode
./build_and_run.sh --detach

# The easiest way to run the full pipeline:
make benchmark

```


## Usage Guide

### Basic Benchmarking

The primary benchmarking script automatically:
1. Builds optimized Docker images
2. Starts TensorBoard for visualization
3. Runs comprehensive benchmarks across all optimization techniques
4. Generates detailed results and profiling data

### Monitoring Progress

- **TensorBoard**: Navigate to `http://localhost:6006` for real-time metrics
- **Container Logs**: `docker-compose logs -f densenet-benchmark`
- **Results**: Check `./results/benchmark_results.csv` for detailed metrics

### Manual Model Testing

```python
from main import DenseNetBenchmark

# Initialize benchmark suite
benchmark = DenseNetBenchmark(output_dir="./test_results")

# Run specific optimization
model = benchmark.load_base_model()
results = benchmark.benchmark_model(model, "custom_test", "baseline")
```

## Optimization Approaches

### 1. Dynamic Quantization
**Technique**: Convert floating-point weights to 8-bit integers
- **Memory Reduction**: ~75% smaller model size
- **Speed Improvement**: 2-3x faster inference on CPU
- **Accuracy Impact**: Minimal (<1% degradation)
- **Use Case**: CPU inference, mobile deployment

### 2. Structured Pruning
**Technique**: Remove entire channels/filters based on importance
- **Memory Reduction**: ~30-50% smaller model
- **Speed Improvement**: 1.5-2x faster inference
- **Accuracy Impact**: Moderate (2-5% degradation)
- **Use Case**: Resource-constrained environments

### 3. Knowledge Distillation
**Technique**: Train smaller "student" model to mimic larger "teacher"
- **Memory Reduction**: ~60% smaller architecture
- **Speed Improvement**: 3-4x faster inference
- **Accuracy Impact**: Significant (5-10% degradation)
- **Use Case**: Edge deployment, real-time applications


## Results Analysis

### Results

```csv
model_variant,batch_size,device,ram_usage_mb,vram_usage_mb,cpu_utilization_pct,gpu_utilization_pct,latency_ms,throughput_samples_sec,accuracy_top1,accuracy_top5,model_size_mb,optimization_technique
densenet121_baseline,1,cuda,1572.97,821.0,0.0,5.0,46.07,21.71,0.0,0.0,30.76,none
densenet121_baseline,4,cuda,1713.9,867.0,0.0,23.0,66.91,59.78,0.0,0.0,30.76,none
densenet121_baseline,8,cuda,1859.14,925.0,0.0,9.0,57.98,137.97,0.0,0.0,30.76,none
densenet121_quantized,1,cpu,1981.08,937.0,0.8,1.0,110.31,9.07,0.0,0.0,26.85,dynamic_quantization
densenet121_quantized,4,cpu,2133.48,937.0,0.0,7.0,355.38,11.26,0.0,0.0,26.85,dynamic_quantization
densenet121_quantized,8,cpu,2311.54,929.0,0.0,0.0,675.51,11.84,0.0,0.0,26.85,dynamic_quantization
densenet121_pruned,1,cuda,2268.35,943.0,0.0,37.0,44.76,22.34,0.0,0.0,30.76,structured_pruning
densenet121_pruned,4,cuda,2268.03,943.0,0.0,34.0,64.79,61.74,0.0,0.0,30.76,structured_pruning
densenet121_pruned,8,cuda,2277.53,951.0,6.7,13.0,51.18,156.3,0.0,0.0,30.76,structured_pruning
densenet121_distilled,1,cuda,2552.83,951.0,16.3,1.0,4.43,225.64,0.0,0.0,0.81,knowledge_distillation
densenet121_distilled,4,cuda,2655.0,951.0,13.7,8.0,10.22,391.29,0.0,0.0,0.81,knowledge_distillation
densenet121_distilled,8,cuda,2680.19,951.0,14.0,8.0,15.11,529.62,0.0,0.0,0.81,knowledge_distillation


```
### Visualization

TensorBoard provides interactive visualizations:
- **Performance Trends**: Latency vs. batch size
- **Resource Utilization**: CPU/GPU usage patterns
- **Memory Profiles**: Allocation patterns over time
- **Accuracy Comparison**: Trade-off analysis

## Performance Benchmarks

### Baseline Performance (DenseNet-121, CUDA)

* **Model Size**: 30.76 MB  
* **RAM Usage**: 1572–1859 MB  
* **VRAM Usage**: 821–925 MB  
* **Latency**:
  * Batch=1 → 46.07 ms  
  * Batch=4 → 66.91 ms  
  * Batch=8 → 57.98 ms  
* **Throughput**:
  * Batch=1 → 21.71 samples/sec  
  * Batch=4 → 59.78 samples/sec  
  * Batch=8 → 137.97 samples/sec  
* **Accuracy (Top-1 / Top-5)**: Not measured (synthetic dataset)  
![Architecture](images/baseline.png)  

---

### Quantized Model (Dynamic Quantization, CPU)

* **Model Size**: 26.85 MB (~12.7% smaller)  
* **RAM Usage**: 1981–2311 MB  
* **VRAM Usage**: ~937 MB (constant)  
* **Latency**:
  * Batch=1 → 110.31 ms  
  * Batch=4 → 355.38 ms  
  * Batch=8 → 675.51 ms  
* **Throughput**:
  * Batch=1 → 9.07 samples/sec  
  * Batch=4 → 11.26 samples/sec  
  * Batch=8 → 11.84 samples/sec  
* **Observation**: Quantization reduced size but led to **significant latency increase** compared to GPU baseline.  
![Architecture](images/quanized.png)  

---

### Pruned Model (Structured Pruning, CUDA)

* **Model Size**: 30.76 MB (no reduction)  
* **RAM Usage**: ~2268–2277 MB  
* **VRAM Usage**: 943–951 MB  
* **Latency**:
  * Batch=1 → 44.76 ms  
  * Batch=4 → 64.79 ms  
  * Batch=8 → 51.18 ms  
* **Throughput**:
  * Batch=1 → 22.34 samples/sec  
  * Batch=4 → 61.74 samples/sec  
  * Batch=8 → 156.30 samples/sec  
* **Accuracy**: Not measured (synthetic dataset)  
![Architecture](images/pruned.png)  

---

### Distilled Model (Knowledge Distillation, CUDA)

* **Model Size**: 0.81 MB (~97% smaller)  
* **RAM Usage**: 2552–2680 MB  
* **VRAM Usage**: 951 MB (constant)  
* **Latency**:
  * Batch=1 → 4.43 ms  
  * Batch=4 → 10.22 ms  
  * Batch=8 → 15.11 ms  
* **Throughput**:
  * Batch=1 → 225.64 samples/sec  
  * Batch=4 → 391.29 samples/sec  
  * Batch=8 → 529.62 samples/sec  
* **Accuracy**: Not measured (synthetic dataset)  
![Architecture](images/distilled.png)  

---

## Trade-offs Discussion

### Performance vs. Accuracy

* **Baseline (CUDA)**: Balanced performance with good latency and throughput; however, model size is relatively large (30.76 MB).  
* **Quantization (CPU)**: Provides memory savings but shows **much worse latency and throughput**, making it unsuitable for real-time tasks unless GPU is unavailable.  
* **Pruning (CUDA)**: Achieved slightly faster inference than baseline with higher throughput at batch=8, but model size didn’t reduce (due to current pruning config). Needs accuracy revalidation.  
* **Distillation (CUDA)**: Extremely compact model with **drastically reduced latency** and **highest throughput**. Excellent candidate for deployment if accuracy holds up on real datasets.  

### Resource vs. Deployment Complexity

* **CPU Quantized Models**: Deployable where GPUs aren’t available, but the trade-off is very poor latency.  
* **GPU Pruned Models**: Efficient in larger batch sizes, though memory usage increases. Needs tuning for real-world accuracy.  
* **Distilled Models**: Ideal for **edge devices or large-scale inference** because of minimal size and superior throughput.  
* **Baseline**: Safe and reliable, but resource-heavy.  

👉 The final choice depends on whether the **deployment target prioritizes GPU/CPU availability, throughput, or memory efficiency**.

## Known Limitations

### Current Limitations
1. **GPU Support**: Requires NVIDIA Docker runtime for GPU benchmarking
2. **Model Variants**: Currently supports DenseNet-121 only
3. **Dataset**: Uses synthetic data for benchmarking (not real ImageNet)
4. **Platform**: Optimized for Linux environments
5. **Cold Start**: Serverless deployment has initial latency overhead

### Accuracy Considerations
- Synthetic dataset may not reflect real-world performance
- Optimization techniques may behave differently on actual ImageNet data
- Model accuracy should be validated on production datasets

### Scalability Limits
- Single-node benchmarking (no distributed testing)
- Limited batch size testing (max 32)
- Memory constraints on large models




## Deployment with KNative
### Bonus Challenge

As an advanced exercise, you can deploy this project using **Knative**.  
We’ve prepared a detailed step-by-step guide for you:

👉 [Deployment with Knative Guide](/deploy_k8_knative/README.md)

This bonus challenge will help you explore:
- Knative Serving for serverless deployment
- Autoscaling and traffic splitting
- Easy deployment on Kubernetes

Feel free to check out the guide and try deploying it yourself!

### Deployment without KNative

👉 [Deployment without Knative Guide](/deploy_k8_fastapi/README.md)
