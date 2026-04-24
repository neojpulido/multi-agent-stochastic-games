# Makefile for FIT5226 MARL Project
# AI Architect / Professor Standard

.PHONY: all test stage1 stage2-sarah stage2-robert run-all clean

# Environment Variables
export PYTHONPATH := .
PYTHON := python
PYTEST := pytest

# Default target
all: test run-all

# --- Testing ---
test:
	@echo "🧪 Running unit, integration, and functional tests..."
	$(PYTEST) tests

# --- Experiments ---
stage1:
	@echo "🚀 Running Assignment 1: Single-Agent Baseline..."
	$(PYTHON) main.py --config configs/stage1_baseline.json

stage2-sarah:
	@echo "🚀 Running Stage 2 Phase 1: Sarah's Safety Coordination..."
	$(PYTHON) main.py --config configs/stage2_sarah_safe.json

stage2-robert:
	@echo "🚀 Running Stage 2 Phase 2: Robert's Stress Test..."
	$(PYTHON) main.py --config configs/stage2_robert_efficient.json

stress-test:
	@echo "🚀 Running High Stochasticity Stress Test..."
	$(PYTHON) main.py --config configs/stage2_stress_test.json

default-run:
	@echo "🚀 Running Default Balanced Run..."
	$(PYTHON) main.py --config configs/default_experiment.json

# Run all experiments sequentially
run-all: stage1 stage2-sarah stage2-robert

# --- Cleanup ---
clean:
	@echo "🧹 Cleaning up pycache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
