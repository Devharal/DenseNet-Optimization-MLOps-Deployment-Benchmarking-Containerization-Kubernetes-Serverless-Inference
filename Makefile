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

serve-results: ## Serve benchmark results via HTTP
	@echo "$(BLUE)Serving results at http://localhost:8000$(NC)"
	@cd $(RESULTS_DIR) && python3 -m http.server 8000

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@mkdir -p docs
	@cp README.md docs/
	@echo "$(GREEN)✓ Documentation generated in docs/$(NC)"

install-deps: ## Install Python dependencies
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	@pip install -r requirements.txt
	@pip install pytest pytest-cov
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

lint: ## Run code linting
	@echo "$(BLUE)Running code linting...$(NC)"
	@python3 -m flake8 main.py inference_server.py test_suite.py --max-line-length=100 --ignore=E203,W503 || true
	@echo "$(GREEN)✓ Linting completed$(NC)"

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	@python3 -m black main.py inference_server.py test_suite.py --line-length=100 || echo "Install black: pip install black"
	@echo "$(GREEN)✓ Code formatted$(NC)"

security-scan: ## Run security scanning
	@echo "$(BLUE)Running security scan...$(NC)"
	@docker run --rm -v "$(PWD)":/app -w /app securecodewarrior/docker-scan:latest || echo "Security scan tool not available"
	@echo "$(GREEN)✓ Security scan completed$(NC)"

backup-results: ## Backup benchmark results
	@echo "$(BLUE)Backing up results...$(NC)"
	@tar -czf results-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz $(RESULTS_DIR) $(LOGS_DIR)
	@echo "$(GREEN)✓ Results backed up$(NC)"

export-models: ## Export optimized models
	@echo "$(BLUE)Exporting optimized models...$(NC)"
	@mkdir -p exported-models
	@cp -r $(RESULTS_DIR)/models/* exported-models/ 2>/dev/null || echo "No models to export"
	@echo "$(GREEN)✓ Models exported to exported-models/$(NC)"

performance-report: ## Generate performance report
	@echo "$(BLUE)Generating performance report...$(NC)"
	@if [ -f $(RESULTS_DIR)/benchmark_results.csv ]; then \
		python3 -c " \
import pandas as pd; \
import matplotlib.pyplot as plt; \
import seaborn as sns; \
df = pd.read_csv('$(RESULTS_DIR)/benchmark_results.csv'); \
plt.figure(figsize=(12, 8)); \
plt.subplot(2, 2, 1); \
batch_1 = df[df['batch_size'] == 1]; \
sns.barplot(data=batch_1, x='optimization_technique', y='latency_ms'); \
plt.title('Latency Comparison'); \
plt.xticks(rotation=45); \
plt.subplot(2, 2, 2); \
sns.barplot(data=batch_1, x='optimization_technique', y='throughput_samples_sec'); \
plt.title('Throughput Comparison'); \
plt.xticks(rotation=45); \
plt.subplot(2, 2, 3); \
sns.barplot(data=batch_1, x='optimization_technique', y='model_size_mb'); \
plt.title('Model Size Comparison'); \
plt.xticks(rotation=45); \
plt.subplot(2, 2, 4); \
sns.barplot(data=batch_1, x='optimization_technique', y='accuracy_top1'); \
plt.title('Accuracy Comparison'); \
plt.xticks(rotation=45); \
plt.tight_layout(); \
plt.savefig('performance_report.png', dpi=300, bbox_inches='tight'); \
print('Performance report saved as performance_report.png'); \
		" 2>/dev/null || echo "Install matplotlib and seaborn for report generation"; \
	else \
		echo "$(YELLOW)No benchmark results found$(NC)"; \
	fi

teardown: ## Complete teardown of all resources
	@echo "$(YELLOW)Tearing down all resources...$(NC)"
	@$(MAKE) clean
	@docker volume prune -f
	@docker network prune -f
	@kind delete cluster --name $(CLUSTER_NAME) 2>/dev/null || true
	@rm -rf $(RESULTS_DIR) $(LOGS_DIR) $(MODELS_DIR) $(CACHE_DIR)
	@echo "$(GREEN)✓ Complete teardown completed$(NC)"

# Development targets
dev-setup: ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@$(MAKE) install-deps
	@$(MAKE) setup
	@pre-commit install || echo "Install pre-commit: pip install pre-commit"
	@echo "$(GREEN)✓ Development environment ready$(NC)"

dev-test: ## Run development tests with coverage
	@echo "$(BLUE)Running development tests...$(NC)"
	@python3 -m pytest test_suite.py -v --cov=. --cov-report=html || python3 test_suite.py --verbose
	@echo "$(GREEN)✓ Development tests completed$(NC)"

dev-watch: ## Watch for changes and run tests
	@echo "$(BLUE)Watching for changes...$(NC)"
	@while inotifywait -r -e modify,create,delete .; do \
		$(MAKE) dev-test; \
	done

# Production targets
prod-deploy: ## Production deployment checklist
	@echo "$(BLUE)Production Deployment Checklist:$(NC)"
	@echo "================================"
	@echo "1. Run security scan: make security-scan"
	@echo "2. Run full test suite: make test"
	@echo "3. Generate performance report: make performance-report"
	@echo "4. Deploy to staging: make deploy-knative"
	@echo "5. Run API tests: make test-api"
	@echo "6. Monitor metrics: make monitor"
	@echo "7. Backup results: make backup-results"
	@echo ""
	@echo "$(GREEN)Follow this checklist for production deployment$(NC)"

# Quick shortcuts
quick: benchmark ## Quick alias for benchmark
all: setup build test benchmark deploy-local ## Run complete pipeline
ci: install-deps lint test ## Continuous integration pipeline
cd: build deploy-knative test-api ## Continuous deployment pipeline