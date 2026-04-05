# ============================================================================
# Allen-Cahn-CUDA — Project Makefile
# ----------------------------------------------------------------------------
# Orchestrates build, test, Docker, and Kubernetes workflows.
#
# Usage:
#   make help          Show all targets
#   make build         CMake configure + build (Release)
#   make test          Run all tests
#   make docker-build-prod   Build production Docker image
#   make k8s-deploy-dev      Deploy to dev Kubernetes environment
# ============================================================================

# ── Configurable Variables ──────────────────────────────────────────────────

# Build
BUILD_DIR          ?= build
CMAKE_GENERATOR    ?= Ninja
CMAKE_PRESET       ?= release
CMAKE_PRESET_DEBUG ?= debug
PARALLEL_JOBS      ?= $(shell nproc 2>/dev/null || echo 4)

# Docker
DOCKER_REGISTRY    ?= ghcr.io/allen-cahn-cuda
IMAGE_NAME         ?= allencahn-cuda
IMAGE_TAG          ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo latest)
DOCKER_BUILD_ARGS  ?=

# Kubernetes
K8S_NAMESPACE_DEV  ?= allen-cahn-dev
K8S_NAMESPACE_PROD ?= allen-cahn-prod
KUSTOMIZE          ?= kubectl kustomize
KUBECTL            ?= kubectl

# Linting / Formatting
CLANG_TIDY         ?= clang-tidy
CLANG_FORMAT       ?= clang-format
COMPILE_COMMANDS   ?= $(BUILD_DIR)/$(CMAKE_PRESET_DEBUG)/compile_commands.json

# ── Phony Targets ───────────────────────────────────────────────────────────

.PHONY: help \
        build build-debug clean \
        test test-unit test-integration test-coverage \
        docker-build-dev docker-build-test docker-build-prod docker-build-all \
        docker-test docker-run docker-shell \
        docker-compose-up docker-compose-down \
        k8s-deploy-dev k8s-deploy-prod \
        k8s-undeploy-dev k8s-undeploy-prod \
        k8s-status k8s-logs \
        lint format

.DEFAULT_GOAL := help

# ============================================================================
# Build Targets
# ============================================================================

build: ## CMake configure + build (Release)
	@echo "==> Configuring (preset: $(CMAKE_PRESET))..."
	cmake --preset $(CMAKE_PRESET) -G $(CMAKE_GENERATOR)
	@echo "==> Building..."
	cmake --build $(BUILD_DIR)/$(CMAKE_PRESET) --parallel $(PARALLEL_JOBS)

build-debug: ## CMake configure + build (Debug, tests enabled)
	@echo "==> Configuring (preset: $(CMAKE_PRESET_DEBUG))..."
	cmake --preset $(CMAKE_PRESET_DEBUG) -G $(CMAKE_GENERATOR)
	@echo "==> Building..."
	cmake --build $(BUILD_DIR)/$(CMAKE_PRESET_DEBUG) --parallel $(PARALLEL_JOBS)

clean: ## Remove all build artifacts
	@echo "==> Cleaning build directory..."
	rm -rf $(BUILD_DIR)
	@echo "==> Clean complete."

# ============================================================================
# Test Targets
# ============================================================================

test: build-debug ## Run all tests
	@echo "==> Running all tests..."
	ctest --test-dir $(BUILD_DIR)/$(CMAKE_PRESET_DEBUG) --output-on-failure --parallel $(PARALLEL_JOBS)

test-unit: build-debug ## Run unit tests only
	@echo "==> Running unit tests..."
	ctest --test-dir $(BUILD_DIR)/$(CMAKE_PRESET_DEBUG) --output-on-failure --parallel $(PARALLEL_JOBS) \
		-R "unit_tests"

test-integration: build-debug ## Run integration tests only
	@echo "==> Running integration tests..."
	ctest --test-dir $(BUILD_DIR)/$(CMAKE_PRESET_DEBUG) --output-on-failure --parallel $(PARALLEL_JOBS) \
		-R "integration_tests"

test-coverage: ## Build with coverage instrumentation and generate report
	@echo "==> Configuring with coverage flags..."
	cmake --preset $(CMAKE_PRESET_DEBUG) -G $(CMAKE_GENERATOR) \
		-DCMAKE_CXX_FLAGS="--coverage -fprofile-arcs -ftest-coverage" \
		-DCMAKE_CUDA_FLAGS="--coverage -fprofile-arcs -ftest-coverage" \
		-DCMAKE_EXE_LINKER_FLAGS="--coverage"
	@echo "==> Building..."
	cmake --build $(BUILD_DIR)/$(CMAKE_PRESET_DEBUG) --parallel $(PARALLEL_JOBS)
	@echo "==> Running tests..."
	ctest --test-dir $(BUILD_DIR)/$(CMAKE_PRESET_DEBUG) --output-on-failure --parallel $(PARALLEL_JOBS)
	@echo "==> Generating coverage report..."
	@if command -v gcovr >/dev/null 2>&1; then \
		gcovr --root . --filter src/ --print-summary --html-details $(BUILD_DIR)/coverage/index.html; \
		echo "==> Coverage report: $(BUILD_DIR)/coverage/index.html"; \
	elif command -v lcov >/dev/null 2>&1; then \
		lcov --capture --directory $(BUILD_DIR) --output-file $(BUILD_DIR)/coverage.info --ignore-errors mismatch; \
		lcov --remove $(BUILD_DIR)/coverage.info '/usr/*' '*/tests/*' '*/build/*' --output-file $(BUILD_DIR)/coverage.info; \
		genhtml $(BUILD_DIR)/coverage.info --output-directory $(BUILD_DIR)/coverage; \
		echo "==> Coverage report: $(BUILD_DIR)/coverage/index.html"; \
	else \
		echo "WARNING: Neither gcovr nor lcov found. Install one for HTML coverage reports."; \
	fi

# ============================================================================
# Docker Targets
# ============================================================================

docker-build-dev: ## Build development Docker image
	@echo "==> Building dev image..."
	docker build -f docker/Dockerfile.dev \
		-t $(IMAGE_NAME):dev \
		-t $(DOCKER_REGISTRY)/$(IMAGE_NAME):dev \
		$(DOCKER_BUILD_ARGS) .

docker-build-test: ## Build test Docker image
	@echo "==> Building test image..."
	docker build -f docker/Dockerfile.test \
		-t $(IMAGE_NAME):test \
		-t $(DOCKER_REGISTRY)/$(IMAGE_NAME):test \
		$(DOCKER_BUILD_ARGS) .

docker-build-prod: ## Build production Docker image
	@echo "==> Building prod image (tag: $(IMAGE_TAG))..."
	docker build -f docker/Dockerfile.prod \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		-t $(IMAGE_NAME):latest \
		-t $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG) \
		-t $(DOCKER_REGISTRY)/$(IMAGE_NAME):latest \
		$(DOCKER_BUILD_ARGS) .

docker-build-all: docker-build-dev docker-build-test docker-build-prod ## Build all Docker images

docker-test: ## Run tests inside Docker container
	@echo "==> Running tests in Docker..."
	docker compose --profile test run --rm test

docker-run: ## Run production simulation in Docker container
	@echo "==> Running simulation in Docker..."
	docker compose --profile prod run --rm prod

docker-shell: ## Open interactive shell in dev container
	@echo "==> Launching dev shell..."
	docker compose --profile dev run --rm dev /bin/bash

docker-compose-up: ## Start all services via docker-compose
	@echo "==> Starting services..."
	docker compose --profile dev up -d

docker-compose-down: ## Stop all services and remove containers
	@echo "==> Stopping services..."
	docker compose --profile dev --profile test --profile prod down

# ============================================================================
# Kubernetes Targets
# ============================================================================

k8s-deploy-dev: ## Deploy to dev Kubernetes environment
	@echo "==> Deploying to dev (namespace: $(K8S_NAMESPACE_DEV))..."
	$(KUSTOMIZE) k8s/overlays/dev | $(KUBECTL) apply -f -

k8s-deploy-prod: ## Deploy to prod Kubernetes environment
	@echo "==> Deploying to prod (namespace: $(K8S_NAMESPACE_PROD))..."
	$(KUSTOMIZE) k8s/overlays/prod | $(KUBECTL) apply -f -

k8s-undeploy-dev: ## Remove dev Kubernetes deployment
	@echo "==> Removing dev deployment..."
	$(KUSTOMIZE) k8s/overlays/dev | $(KUBECTL) delete -f - --ignore-not-found

k8s-undeploy-prod: ## Remove prod Kubernetes deployment
	@echo "==> Removing prod deployment..."
	$(KUSTOMIZE) k8s/overlays/prod | $(KUBECTL) delete -f - --ignore-not-found

k8s-status: ## Show status of all Allen-Cahn k8s resources
	@echo "==> Dev environment ($(K8S_NAMESPACE_DEV)):"
	-$(KUBECTL) get all -n $(K8S_NAMESPACE_DEV) 2>/dev/null || echo "    Namespace not found or not accessible."
	@echo ""
	@echo "==> Prod environment ($(K8S_NAMESPACE_PROD)):"
	-$(KUBECTL) get all -n $(K8S_NAMESPACE_PROD) 2>/dev/null || echo "    Namespace not found or not accessible."

k8s-logs: ## Show logs from running pods
	@echo "==> Fetching logs from dev pods..."
	-$(KUBECTL) logs -n $(K8S_NAMESPACE_DEV) -l app.kubernetes.io/managed-by=kustomize --tail=100 --all-containers 2>/dev/null || true
	@echo ""
	@echo "==> Fetching logs from prod pods..."
	-$(KUBECTL) logs -n $(K8S_NAMESPACE_PROD) -l app.kubernetes.io/managed-by=kustomize --tail=100 --all-containers 2>/dev/null || true

# ============================================================================
# Utility Targets
# ============================================================================

lint: ## Run clang-tidy on the codebase
	@if ! command -v $(CLANG_TIDY) >/dev/null 2>&1; then \
		echo "ERROR: $(CLANG_TIDY) not found. Install clang-tidy to use this target."; \
		exit 1; \
	fi
	@if [ ! -f "$(COMPILE_COMMANDS)" ]; then \
		echo "==> compile_commands.json not found; building debug first..."; \
		$(MAKE) build-debug; \
	fi
	@echo "==> Running clang-tidy..."
	find src -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.hpp' | \
		xargs $(CLANG_TIDY) -p $(COMPILE_COMMANDS)

format: ## Run clang-format on all source files
	@if ! command -v $(CLANG_FORMAT) >/dev/null 2>&1; then \
		echo "ERROR: $(CLANG_FORMAT) not found. Install clang-format to use this target."; \
		exit 1; \
	fi
	@echo "==> Formatting source files..."
	find src tests -type f \( -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.hpp' -o -name '*.h' \) | \
		xargs $(CLANG_FORMAT) -i
	@echo "==> Done."

# ============================================================================
# Help
# ============================================================================

help: ## Show all available targets with descriptions
	@echo ""
	@echo "Allen-Cahn-CUDA Project Makefile"
	@echo "================================"
	@echo ""
	@echo "Usage: make [TARGET] [VARIABLE=value ...]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables (override with VAR=value):"
	@echo "  BUILD_DIR            Build output directory      [$(BUILD_DIR)]"
	@echo "  CMAKE_GENERATOR      CMake generator             [$(CMAKE_GENERATOR)]"
	@echo "  PARALLEL_JOBS        Parallel build jobs         [$(PARALLEL_JOBS)]"
	@echo "  DOCKER_REGISTRY      Docker registry prefix      [$(DOCKER_REGISTRY)]"
	@echo "  IMAGE_NAME           Docker image name           [$(IMAGE_NAME)]"
	@echo "  IMAGE_TAG            Docker image tag            [$(IMAGE_TAG)]"
	@echo "  K8S_NAMESPACE_DEV    K8s dev namespace           [$(K8S_NAMESPACE_DEV)]"
	@echo "  K8S_NAMESPACE_PROD   K8s prod namespace          [$(K8S_NAMESPACE_PROD)]"
	@echo ""
