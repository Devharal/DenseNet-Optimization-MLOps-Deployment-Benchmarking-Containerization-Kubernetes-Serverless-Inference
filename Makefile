# DenseNet Optimization & Benchmarking Makefile
# Comprehensive build, test, and deployment automation

# Configuration
PROJECT_NAME := densenet-optimization
DOCKER_IMAGE := densenet-inference
DOCKER_TAG := latest
CLUSTER_NAME := densenet-knative
NAMESPACE := densenet-inference

# Directories
RESULTS_DIR := ./results
LOGS_DIR := ./logs
MODELS_DIR := ./models
CACHE_DIR := ./cache

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

.PHONY: help setup clean build test benchmark deploy-local deploy-knative monitor logs status teardown

help: ## Show this help message
	@echo "$(BLUE)DenseNet Optimization & Benchmarking$(NC)"
	@echo "======================================"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make setup              # Initial project setup"
	@echo "  make benchmark          # Run complete benchmark suite"
	@echo "  make deploy-knative     # Deploy to KNative serverless"
	@echo "  make test               # Run all tests"
	@echo "  make clean              # Clean up resources"

setup: ## Initial project setup and dependency installation
	@echo "$(BLUE)Setting up project...$(NC)"
	@mkdir -p $(RESULTS_DIR) $(LOGS_DIR) $(MODELS_DIR) $(CACHE_DIR)
	@chmod +x build_and_run.sh setup-cluster.sh deploy-knative.sh
	@if command -v docker >/dev/null 2>&1; then \
		echo "$(GREEN)✓ Docker is available$(NC)"; \
	else \
		echo "$(RED)✗ Docker is required but not installed$(NC)"; \
		exit 1; \
	fi
	@if command -v docker-compose >/dev/null 2>&1 || docker compose version >/dev/null 2>&1; then \
		echo "$(GREEN)✓ Docker Compose is available$(NC)"; \
	else \
		echo "$(RED)✗ Docker Compose is required$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Project setup completed$(NC)"

clean: ## Clean up all resources and temporary files
	@echo "$(YELLOW)Cleaning up resources...$(NC)"
	@docker-compose down --remove-orphans --volumes 2>/dev/null || true
	@docker rmi $(shell docker images -q "*densenet*" 2>/dev/null) 2>/dev/null || true
	@kind delete cluster --name $(CLUSTER_NAME) 2>/dev/null || true
	@docker system prune -f 2>/dev/null || true
	@rm -rf $(RESULTS_DIR)/* $(LOGS_DIR)/* $(CACHE_DIR)/* 2>/dev/null || true
	@rm -f *.log *.tmp 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup completed$(NC)"

build: ## Build Docker images for benchmarking
	@echo "$(BLUE)Building Docker images...$(NC)"
	@docker-compose build --no-cache
	@echo "$(GREEN)✓ Docker images built successfully$(NC)"

test: ## Run comprehensive test suite
	@echo "$(BLUE)Running test suite...$(NC)"
	@python3 test_suite.py --verbose
	@echo "$(GREEN)✓ Tests completed$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	@python3 test_suite.py --unit-only --verbose

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	@python3 test_suite.py --integration-only --verbose

benchmark: ## Run complete benchmarking suite
	@echo "$(BLUE)Starting comprehensive benchmarking...$(NC)"
	@./build_and_run.sh --output-dir $(RESULTS_DIR) --gpu-enabled true 
	@echo "$(GREEN)✓ Benchmarking completed$(NC)"
	@echo "$(BLUE)Results available at:$(NC) $(RESULTS_DIR)/benchmark_results.csv"
	@echo "$(BLUE)TensorBoard:$(NC) http://localhost:6006"

benchmark-cpu: ## Run CPU-only benchmarking
	@echo "$(BLUE)Starting CPU benchmarking...$(NC)"
	@./build_and_run.sh --output-dir $(RESULTS_DIR) --gpu-enabled false

benchmark-quick: ## Run quick benchmarking (reduced test set)
	@echo "$(BLUE)Starting quick benchmarking...$(NC)"
	@docker-compose up --build densenet-benchmark
	@echo "$(GREEN)✓ Quick benchmarking completed$(NC)"


benchmark-compare: ## Compare benchmark results
	@echo "$(BLUE)Benchmark Comparison:$(NC)"
	@echo "====================="
	@if [ -f $(RESULTS_DIR)/benchmark_results.csv ]; then \
		python3 -c " \
import pandas as pd; \
df = pd.read_csv('$(RESULTS_DIR)/benchmark_results.csv'); \
print('Models tested:', df['model_variant'].nunique()); \
print('Optimization techniques:', list(df['optimization_technique'].unique())); \
batch_1 = df[df['batch_size'] == 1]; \
if not batch_1.empty: \
    best_latency = batch_1.loc[batch_1['latency_ms'].idxmin()]; \
    best_throughput = batch_1.loc[batch_1['throughput_samples_sec'].idxmax()]; \
    print(f'Best latency: {best_latency[\"latency_ms\"]:.2f}ms ({best_latency[\"optimization_technique\"]})'); \
    print(f'Best throughput: {best_throughput[\"throughput_samples_sec\"]:.2f} samples/sec ({best_throughput[\"optimization_technique\"]})'); \
else: \
    print('No batch_size=1 results found') \
		"; \
	else \
		echo "$(YELLOW)No benchmark results found. Run 'make benchmark' first.$(NC)"; \
	fi


# Quick shortcuts
quick: benchmark ## Quick alias for benchmark
all: setup build test benchmark deploy-local ## Run complete pipeline
ci: install-deps lint test ## Continuous integration pipeline
cd: build deploy-knative test-api ## Continuous deployment pipeline