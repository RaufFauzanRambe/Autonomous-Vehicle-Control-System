# =============================================================================
# Autonomous Vehicle Control System - Makefile
# =============================================================================
# Build automation for C++ core, Python modules, ROS2 workspace,
# Docker images, documentation, and testing.
# =============================================================================

# ---- Configuration ----
PROJECT_NAME    := avcs
BUILD_DIR       := build
INSTALL_DIR     := install
SRC_DIR         := src
TEST_DIR        := test
DOC_DIR         := docs
SCRIPT_DIR      := scripts
CONFIG_DIR      := config

CMAKE_BUILD_TYPE ?= Release
CMAKE_FLAGS     := -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
                   -DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR) \
                   -DBUILD_TESTING=ON

NUM_JOBS        := $(shell nproc 2>/dev/null || echo 4)
PYTHON          := python3
PIP             := pip3
CONDA_ENV       := avcs
DOCKER_IMAGE    := avcs:latest
DOCKER_REGISTRY := ghcr.io/avcs

# ---- Colors ----
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
NC     := \033[0m

# =============================================================================
# Primary Targets
# =============================================================================

.PHONY: all
all: build-cpp build-python build-ros
	@echo "$(GREEN)[$(PROJECT_NAME)] Full build complete.$(NC)"

# =============================================================================
# C++ Build
# =============================================================================

.PHONY: build-cpp
build-cpp: $(BUILD_DIR)
	@echo "$(YELLOW)[$(PROJECT_NAME)] Building C++ core...$(NC)"
	cmake --build $(BUILD_DIR) --config $(CMAKE_BUILD_TYPE) -j$(NUM_JOBS)
	@echo "$(GREEN)[$(PROJECT_NAME)] C++ build complete.$(NC)"

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. $(CMAKE_FLAGS)

.PHONY: cmake-configure
cmake-configure:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. $(CMAKE_FLAGS)

.PHONY: install-cpp
install-cpp: build-cpp
	cmake --install $(BUILD_DIR)

# =============================================================================
# Python Build
# =============================================================================

.PHONY: build-python
build-python:
	@echo "$(YELLOW)[$(PROJECT_NAME)] Installing Python package...$(NC)"
	$(PIP) install -e .[dev]
	@echo "$(GREEN)[$(PROJECT_NAME)] Python package installed.$(NC)"

.PHONY: install-requirements
install-requirements:
	@echo "$(YELLOW)[$(PROJECT_NAME)] Installing Python dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)[$(PROJECT_NAME)] Dependencies installed.$(NC)"

.PHONY: create-conda-env
create-conda-env:
	@echo "$(YELLOW)[$(PROJECT_NAME)] Creating conda environment...$(NC)"
	conda env create -f environment.yml
	@echo "$(GREEN)[$(PROJECT_NAME)] Conda environment '$(CONDA_ENV)' created.$(NC)"

# =============================================================================
# ROS2 Build
# =============================================================================

.PHONY: build-ros
build-ros:
	@echo "$(YELLOW)[$(PROJECT_NAME)] Building ROS2 workspace...$(NC)"
	colcon build --symlink-install --cmake-args $(CMAKE_FLAGS)
	@echo "$(GREEN)[$(PROJECT_NAME)] ROS2 workspace built.$(NC)"

.PHONY: source-ros
source-ros:
	@echo "source $(CURDIR)/install/setup.bash"

# =============================================================================
# Testing
# =============================================================================

.PHONY: test
test: test-cpp test-python
	@echo "$(GREEN)[$(PROJECT_NAME)] All tests complete.$(NC)"

.PHONY: test-cpp
test-cpp: build-cpp
	@echo "$(YELLOW)[$(PROJECT_NAME)] Running C++ tests...$(NC)"
	cd $(BUILD_DIR) && ctest --output-on-failure -j$(NUM_JOBS)

.PHONY: test-python
test-python: build-python
	@echo "$(YELLOW)[$(PROJECT_NAME)] Running Python tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term

.PHONY: test-integration
test-integration:
	@echo "$(YELLOW)[$(PROJECT_NAME)] Running integration tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR)/integration -v --timeout=120

.PHONY: test-ros
test-ros:
	@echo "$(YELLOW)[$(PROJECT_NAME)] Running ROS2 tests...$(NC)"
	colcon test --packages-select $(PROJECT_NAME)
	colcon test-result --verbose

# =============================================================================
# Docker
# =============================================================================

.PHONY: docker-build
docker-build:
	@echo "$(YELLOW)[$(PROJECT_NAME)] Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)[$(PROJECT_NAME)] Docker image built: $(DOCKER_IMAGE)$(NC)"

.PHONY: docker-run
docker-run:
	docker run -it --rm --gpus all \
		--network host \
		-v $(CURDIR)/config:/avcs/config \
		-v $(CURDIR)/data:/avcs/data \
		$(DOCKER_IMAGE)

.PHONY: docker-push
docker-push:
	docker tag $(DOCKER_IMAGE) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)

.PHONY: docker-compose-up
docker-compose-up:
	docker-compose up --build -d

.PHONY: docker-compose-down
docker-compose-down:
	docker-compose down -v

# =============================================================================
# Documentation
# =============================================================================

.PHONY: docs
docs: docs-cpp docs-python
	@echo "$(GREEN)[$(PROJECT_NAME)] Documentation built.$(NC)"

.PHONY: docs-cpp
docs-cpp:
	@echo "$(YELLOW)[$(PROJECT_NAME)] Generating C++ docs (Doxygen)...$(NC)"
	doxygen Doxyfile

.PHONY: docs-python
docs-python:
	@echo "$(YELLOW)[$(PROJECT_NAME)] Generating Python docs (Sphinx)...$(NC)"
	cd $(DOC_DIR) && $(PYTHON) -m sphinx -b html . _build/html

# =============================================================================
# Code Quality
# =============================================================================

.PHONY: lint
lint: lint-cpp lint-python
	@echo "$(GREEN)[$(PROJECT_NAME)] Linting complete.$(NC)"

.PHONY: lint-cpp
lint-cpp:
	@echo "$(YELLOW)[$(PROJECT_NAME)] Linting C++ (clang-tidy)...$(NC)"
	find $(SRC_DIR) -name '*.cpp' -o -name '*.hpp' | xargs clang-tidy -p $(BUILD_DIR)

.PHONY: lint-python
lint-python:
	@echo "$(YELLOW)[$(PROJECT_NAME)] Linting Python...$(NC)"
	$(PYTHON) -m flake8 $(SRC_DIR) $(TEST_DIR) --max-line-length=120
	$(PYTHON) -m mypy $(SRC_DIR) --ignore-missing-imports

.PHONY: format
format: format-cpp format-python
	@echo "$(GREEN)[$(PROJECT_NAME)] Formatting complete.$(NC)"

.PHONY: format-cpp
format-cpp:
	find $(SRC_DIR) -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i -style=file

.PHONY: format-python
format-python:
	$(PYTHON) -m black $(SRC_DIR) $(TEST_DIR) --line-length=120
	$(PYTHON) -m isort $(SRC_DIR) $(TEST_DIR)

# =============================================================================
# Simulation
# =============================================================================

.PHONY: sim-carla
sim-carla:
	@echo "$(YELLOW)[$(PROJECT_NAME)] Launching CARLA simulation...$(NC)"
	$(PYTHON) $(SCRIPT_DIR)/launch_carla.py --town Town03 --sync

.PHONY: sim-sumo
sim-sumo:
	@echo "$(YELLOW)[$(PROJECT_NAME)] Launching SUMO simulation...$(NC)"
	$(PYTHON) $(SCRIPT_DIR)/launch_sumo.py --scenario highway_merge

.PHONY: run-scenarios
run-scenarios:
	$(PYTHON) $(SCRIPT_DIR)/run_scenarios.py --all

# =============================================================================
# Cleanup
# =============================================================================

.PHONY: clean
clean: clean-cpp clean-python clean-ros
	@echo "$(GREEN)[$(PROJECT_NAME)] Clean complete.$(NC)"

.PHONY: clean-cpp
clean-cpp:
	rm -rf $(BUILD_DIR)
	rm -rf $(INSTALL_DIR)

.PHONY: clean-python
clean-python:
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf *.egg-info dist .pytest_cache .coverage htmlcov

.PHONY: clean-ros
clean-ros:
	rm -rf log install

.PHONY: clean-docker
clean-docker:
	docker rmi $(DOCKER_IMAGE) 2>/dev/null || true

# =============================================================================
# Help
# =============================================================================

.PHONY: help
help:
	@echo "Autonomous Vehicle Control System - Build Targets"
	@echo "=================================================="
	@echo ""
	@echo "  Primary:"
	@echo "    all                Build everything (C++ + Python + ROS2)"
	@echo "    build-cpp          Build C++ core only"
	@echo "    build-python       Install Python package"
	@echo "    build-ros          Build ROS2 workspace"
	@echo ""
	@echo "  Testing:"
	@echo "    test               Run all tests"
	@echo "    test-cpp           Run C++ unit tests"
	@echo "    test-python        Run Python unit tests"
	@echo "    test-integration   Run integration tests"
	@echo ""
	@echo "  Docker:"
	@echo "    docker-build       Build Docker image"
	@echo "    docker-run         Run in Docker container"
	@echo "    docker-compose-up  Start all services"
	@echo ""
	@echo "  Quality:"
	@echo "    lint               Lint all code"
	@echo "    format             Format all code"
	@echo ""
	@echo "  Docs:"
	@echo "    docs               Build all documentation"
	@echo ""
	@echo "  Cleanup:"
	@echo "    clean              Remove all build artifacts"
