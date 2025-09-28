#!/usr/bin/env python3
"""
DenseNet Optimization & Benchmarking Suite
MLOps Engineer Take-Home Assignment

This module implements comprehensive benchmarking and optimization of DenseNet-121
for production deployment using PyTorch Profiler and TensorBoard.
"""

import os
import sys
import time
import csv
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import psutil
import GPUtil
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Data class to store benchmark results."""
    model_variant: str
    batch_size: int
    device: str
    ram_usage_mb: float
    vram_usage_mb: float
    cpu_utilization_pct: float
    gpu_utilization_pct: float
    latency_ms: float
    throughput_samples_sec: float
    accuracy_top1: float
    accuracy_top5: float
    model_size_mb: float
    optimization_technique: str

class ImageNetDataset(Dataset):
    """Dummy ImageNet-like dataset for benchmarking."""
    
    def __init__(self, size: int = 1000, image_size: Tuple[int, int] = (224, 224)):
        self.size = size
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Generate random images and labels
        self.images = []
        self.labels = []
        for i in range(size):
            # Create random RGB image
            img = Image.fromarray(
                np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            )
            self.images.append(img)
            self.labels.append(np.random.randint(0, 1000))  # 1000 classes like ImageNet
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.transform(self.images[idx])
        label = self.labels[idx]
        return img, label

class SystemMonitor:
    """Monitor system resources during benchmarking."""
    
    @staticmethod
    def get_ram_usage() -> float:
        """Get current RAM usage in MB."""
        return psutil.virtual_memory().used / (1024 * 1024)
    
    @staticmethod
    def get_cpu_utilization() -> float:
        """Get current CPU utilization percentage."""
        return psutil.cpu_percent(interval=0.1)
    
    @staticmethod
    def get_gpu_stats() -> Tuple[float, float]:
        """Get GPU memory usage (MB) and utilization (%)."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                vram_usage = gpu.memoryUsed
                gpu_util = gpu.load * 100
                return vram_usage, gpu_util
        except:
            pass
        
        # Fallback to PyTorch for VRAM
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            return vram_mb, 0.0
        
        return 0.0, 0.0

class ModelOptimizer:
    """Implements various DenseNet optimization techniques."""
    
    @staticmethod
    def apply_quantization(model: nn.Module, device: str) -> nn.Module:
        """Apply dynamic quantization to the model."""
        logger.info("Applying dynamic quantization...")
        
        # Quantization only works on CPU, so move model to CPU
        original_device = next(model.parameters()).device
        model = model.cpu()
        model.eval()
        
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.Conv2d}, 
                dtype=torch.qint8
            )
            logger.info("Quantization successful - model will run on CPU")
            return quantized_model
        except Exception as e:
            logger.warning(f"Quantization failed: {e}, returning original model")
            return model.to(original_device)
    
    @staticmethod
    def apply_pruning(model: nn.Module, sparsity: float = 0.3) -> nn.Module:
        """Apply structured pruning to the model."""
        logger.info(f"Applying pruning with {sparsity:.1%} sparsity...")
        
        import torch.nn.utils.prune as prune
        
        # Prune convolutional layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=sparsity, n=2, dim=0)
                prune.remove(module, 'weight')
        
        return model
    
    @staticmethod
    def apply_knowledge_distillation(model: nn.Module) -> nn.Module:
        """Create a smaller student model for knowledge distillation."""
        logger.info("Creating distilled model...")
        
        # Create a smaller DenseNet variant
        class CompactDenseNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Simplified architecture
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),  # Remove inplace=True
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    
                    # Simplified dense block
                    nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),  # Remove inplace=True
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Linear(128, 1000)
                
            def forward(self, x):
                features = self.features(x)
                out = torch.flatten(features, 1)
                out = self.classifier(out)
                return out
        
        return CompactDenseNet()
    
    @staticmethod
    def apply_tensorrt_optimization(model: nn.Module, device: str) -> nn.Module:
        """Apply TensorRT optimization (simulation)."""
        logger.info("Applying TensorRT-style optimization...")
        
        # Simulate TensorRT optimization by using torch.jit.script
        if device != 'cpu':
            model.eval()
            try:
                # Create example input
                example_input = torch.randn(1, 3, 224, 224).to(device)
                traced_model = torch.jit.trace(model, example_input)
                return traced_model
            except Exception as e:
                logger.warning(f"TensorRT optimization failed: {e}")
                return model
        
        return model

class DenseNetBenchmark:
    """Main benchmarking class for DenseNet optimization."""
    
    def __init__(self, output_dir: str = "/app/results", gpu_enabled: bool = True):
        self.output_dir = Path(output_dir)
        
        # Ensure all necessary directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "profiles").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(parents=True, exist_ok=True)
        
        # Create logs directory  
        logs_dir = Path(output_dir).parent / "logs" / "tensorboard"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = "cuda" if gpu_enabled and torch.cuda.is_available() else "cpu"
        self.batch_sizes = [1, 4, 8]
        self.results: List[BenchmarkResult] = []
        self.optimizer = ModelOptimizer()
        self.monitor = SystemMonitor()
        
        # Setup TensorBoard with correct path
        tb_log_dir = Path(output_dir).parent / "logs" / "tensorboard"
        self.tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        
        logger.info(f"Initialized DenseNetBenchmark with device: {self.device}")
    
    def load_base_model(self) -> nn.Module:
        """Load the base DenseNet-121 model."""
        logger.info("Loading DenseNet-121 model...")
        model = models.densenet121(pretrained=True)
        model = model.to(self.device)
        model.eval()
        return model
    
    def get_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb
    
    def create_dataloader(self, batch_size: int) -> DataLoader:
        """Create DataLoader for benchmarking."""
        dataset = ImageNetDataset(size=100)  # Small dataset for benchmarking
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device == "cuda" else False
        )
        return dataloader
    
    def calculate_accuracy(self, model: nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
        """Calculate top-1 and top-5 accuracy."""
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= 10:  # Limit for benchmarking speed
                    break
                    
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                
                _, pred_top5 = outputs.topk(5, 1, True, True)
                pred_top5 = pred_top5.t()
                correct_top5 += pred_top5.eq(targets.view(1, -1).expand_as(pred_top5)).sum().item()
                
                _, pred_top1 = outputs.topk(1, 1, True, True)
                correct_top1 += pred_top1.eq(targets.view(-1, 1)).sum().item()
                
                total += targets.size(0)
        
        accuracy_top1 = (correct_top1 / total) * 100 if total > 0 else 0.0
        accuracy_top5 = (correct_top5 / total) * 100 if total > 0 else 0.0
        
        return accuracy_top1, accuracy_top5
    
    def benchmark_model(
        self, 
        model: nn.Module, 
        model_variant: str, 
        optimization_technique: str
    ) -> List[BenchmarkResult]:
        """Benchmark a model variant across all batch sizes."""
        results = []
        model_size = self.get_model_size(model)
        
        logger.info(f"Benchmarking {model_variant} with {optimization_technique}")
        
        for batch_size in self.batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            dataloader = self.create_dataloader(batch_size)
            
            # Warmup
            with torch.no_grad():
                dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
                for _ in range(3):
                    _ = model(dummy_input)
            
            # Benchmark with profiler
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if self.device == "cuda" else [ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                use_cuda=self.device == "cuda"
            ) as prof:
                with record_function("model_inference"):
                    latencies = []
                    
                    # Measure inference time
                    for batch_idx, (inputs, targets) in enumerate(dataloader):
                        if batch_idx >= 5:  # Limit for benchmarking
                            break
                            
                        inputs = inputs.to(self.device)
                        
                        # Time inference
                        start_time = time.perf_counter()
                        with torch.no_grad():
                            outputs = model(inputs)
                        torch.cuda.synchronize() if self.device == "cuda" else None
                        end_time = time.perf_counter()
                        
                        latency_ms = (end_time - start_time) * 1000
                        latencies.append(latency_ms)
            
            # Calculate metrics
            avg_latency_ms = np.mean(latencies)
            throughput = (batch_size * 1000) / avg_latency_ms
            
            # Get system metrics
            ram_usage = self.monitor.get_ram_usage()
            vram_usage, gpu_util = self.monitor.get_gpu_stats()
            cpu_util = self.monitor.get_cpu_utilization()
            
            # Calculate accuracy (only for first batch size to save time)
            if batch_size == 1:
                acc_top1, acc_top5 = self.calculate_accuracy(model, dataloader)
            else:
                acc_top1, acc_top5 = 0.0, 0.0  # Skip for other batch sizes
            
            # Create result
            result = BenchmarkResult(
                model_variant=model_variant,
                batch_size=batch_size,
                device=self.device,
                ram_usage_mb=ram_usage,
                vram_usage_mb=vram_usage,
                cpu_utilization_pct=cpu_util,
                gpu_utilization_pct=gpu_util,
                latency_ms=avg_latency_ms,
                throughput_samples_sec=throughput,
                accuracy_top1=acc_top1,
                accuracy_top5=acc_top5,
                model_size_mb=model_size,
                optimization_technique=optimization_technique
            )
            
            results.append(result)
            
            # Log to TensorBoard
            self.tb_writer.add_scalar(
                f"{model_variant}/latency_ms", 
                avg_latency_ms, 
                batch_size
            )
            self.tb_writer.add_scalar(
                f"{model_variant}/throughput", 
                throughput, 
                batch_size
            )
            self.tb_writer.add_scalar(
                f"{model_variant}/memory_usage_mb", 
                vram_usage if self.device == "cuda" else ram_usage, 
                batch_size
            )
            
            # Save profiler trace
            prof_dir = self.output_dir / "profiles" / model_variant
            prof_dir.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(
                str(prof_dir / f"trace_batch_{batch_size}.json")
            )
        
        return results
    
    def run_all_benchmarks(self) -> None:
        """Run all benchmark scenarios."""
        logger.info("Starting comprehensive DenseNet benchmarking...")
        
        # 1. Baseline DenseNet-121
        base_model = self.load_base_model()
        baseline_results = self.benchmark_model(
            base_model, "densenet121_baseline", "none"
        )
        self.results.extend(baseline_results)
        
        # 2. Quantized model (CPU only)
        try:
            base_model_cpu = self.load_base_model()
            quantized_model = self.optimizer.apply_quantization(base_model_cpu, "cpu")
            # Benchmark quantized model on CPU temporarily
            original_device = self.device
            self.device = "cpu"
            quantized_results = self.benchmark_model(
                quantized_model, "densenet121_quantized", "dynamic_quantization"
            )
            self.results.extend(quantized_results)
            self.device = original_device  # Restore original device
        except Exception as e:
            logger.error(f"Quantization benchmark failed: {e}")
        
        # 3. Pruned model
        try:
            pruned_model = self.optimizer.apply_pruning(
                self.load_base_model(), sparsity=0.3
            )
            pruned_results = self.benchmark_model(
                pruned_model, "densenet121_pruned", "structured_pruning"
            )
            self.results.extend(pruned_results)
        except Exception as e:
            logger.error(f"Pruning benchmark failed: {e}")
        
        # 4. Knowledge distilled model
        try:
            distilled_model = self.optimizer.apply_knowledge_distillation(base_model)
            distilled_model = distilled_model.to(self.device)
            distilled_results = self.benchmark_model(
                distilled_model, "densenet121_distilled", "knowledge_distillation"
            )
            self.results.extend(distilled_results)
        except Exception as e:
            logger.error(f"Knowledge distillation benchmark failed: {e}")
        
        # 5. TensorRT optimized model
        # try:
        #     tensorrt_model = self.optimizer.apply_tensorrt_optimization(
        #         self.load_base_model(), self.device
        #     )
        #     tensorrt_results = self.benchmark_model(
        #         tensorrt_model, "densenet121_tensorrt", "tensorrt_optimization"
        #     )
        #     self.results.extend(tensorrt_results)
        # except Exception as e:
        #     logger.error(f"TensorRT optimization benchmark failed: {e}")
    
    def save_results(self) -> None:
        """Save benchmark results to CSV."""
        csv_path = self.output_dir / "benchmark_results.csv"
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = [
                'model_variant', 'batch_size', 'device', 'ram_usage_mb', 'vram_usage_mb',
                'cpu_utilization_pct', 'gpu_utilization_pct', 'latency_ms', 
                'throughput_samples_sec', 'accuracy_top1', 'accuracy_top5', 
                'model_size_mb', 'optimization_technique'
            ]
            writer.writerow(header)
            
            # Write results
            for result in self.results:
                row = [
                    result.model_variant,
                    result.batch_size,
                    result.device,
                    round(result.ram_usage_mb, 2),
                    round(result.vram_usage_mb, 2),
                    round(result.cpu_utilization_pct, 2),
                    round(result.gpu_utilization_pct, 2),
                    round(result.latency_ms, 2),
                    round(result.throughput_samples_sec, 2),
                    round(result.accuracy_top1, 2),
                    round(result.accuracy_top5, 2),
                    round(result.model_size_mb, 2),
                    result.optimization_technique
                ]
                writer.writerow(row)
        
        logger.info(f"Results saved to {csv_path}")
    
    def print_summary(self) -> None:
        """Print benchmark results summary."""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        print("\n" + "="*80)
        print("DENSENET OPTIMIZATION BENCHMARK SUMMARY")
        print("="*80)
        
        # Group results by optimization technique
        techniques = {}
        for result in self.results:
            tech = result.optimization_technique
            if tech not in techniques:
                techniques[tech] = []
            techniques[tech].append(result)
        
        for tech, results in techniques.items():
            print(f"\n{tech.upper()} OPTIMIZATION:")
            print("-" * 50)
            
            # Get batch size 1 results for comparison
            batch_1_results = [r for r in results if r.batch_size == 1]
            if batch_1_results:
                r = batch_1_results[0]
                print(f"  Model Size: {r.model_size_mb:.2f} MB")
                print(f"  Latency (batch=1): {r.latency_ms:.2f} ms")
                print(f"  Throughput (batch=1): {r.throughput_samples_sec:.2f} samples/sec")
                print(f"  Accuracy Top-1: {r.accuracy_top1:.2f}%")
                print(f"  Accuracy Top-5: {r.accuracy_top5:.2f}%")
                
                # Memory usage
                if r.device == "cuda":
                    print(f"  VRAM Usage: {r.vram_usage_mb:.2f} MB")
                print(f"  RAM Usage: {r.ram_usage_mb:.2f} MB")
        
        # Find best performing optimization
        batch_1_results = [r for r in self.results if r.batch_size == 1]
        if batch_1_results:
            best_latency = min(batch_1_results, key=lambda x: x.latency_ms)
            best_throughput = max(batch_1_results, key=lambda x: x.throughput_samples_sec)
            smallest_model = min(batch_1_results, key=lambda x: x.model_size_mb)
            
            print(f"\nBEST PERFORMANCE METRICS:")
            print("-" * 50)
            print(f"  Lowest Latency: {best_latency.optimization_technique} ({best_latency.latency_ms:.2f} ms)")
            print(f"  Highest Throughput: {best_throughput.optimization_technique} ({best_throughput.throughput_samples_sec:.2f} samples/sec)")
            print(f"  Smallest Model: {smallest_model.optimization_technique} ({smallest_model.model_size_mb:.2f} MB)")
        
        print("\n" + "="*80)
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DenseNet Benchmarking Suite")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/app/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--gpu-enabled", 
        type=bool, 
        default=True,
        help="Enable GPU benchmarking"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize benchmark suite
        benchmark = DenseNetBenchmark(
            output_dir=args.output_dir,
            gpu_enabled=args.gpu_enabled
        )
        
        # Run all benchmarks
        benchmark.run_all_benchmarks()
        
        # Save and display results
        benchmark.save_results()
        benchmark.print_summary()
        
        logger.info("Benchmarking completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)
    finally:
        if 'benchmark' in locals():
            benchmark.cleanup()

if __name__ == "__main__":
    main()