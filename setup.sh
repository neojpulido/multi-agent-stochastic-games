#!/bin/bash

# =================================================================
# Project: FIT5226 Assignment 1 - Q-Learning Agent
# Architecture: Zero-Leak / Decoupled Environment
# =================================================================

# Color codes for professional output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ENV_NAME="fit5226_ass1"
PYTHON_VERSION="3.14"

echo -e "${BLUE}=== Initializing Multi-Agent Systems Environment ===${NC}"

# 1. Verify Conda Installation
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Error: Conda not found. Please install Miniconda or Anaconda.${NC}"
    exit 1
fi

# 2. Source Conda (Shell-agnostic initialization)
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo -e "${RED}❌ Error: Could not find conda.sh at $CONDA_BASE${NC}"
    exit 1
fi

# 3. Environment Provisioning
if conda info --envs | grep -q "$ENV_NAME"; then
    echo -e "${GREEN}✅ Environment '$ENV_NAME' already exists.${NC}"
else
    echo -e "${BLUE}🚀 Provisioning isolated environment: Python $PYTHON_VERSION...${NC}"
    # Seeding with pip/ipykernel prevents PEP 668 'externally-managed-environment' errors
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" pip ipykernel -y
fi

# 4. Dependency Injection
echo -e "${BLUE}🔄 Activating $ENV_NAME and syncing dependencies...${NC}"
conda activate "$ENV_NAME"

# Verify we are using the environment's pip, not the system pip
PIP_PATH=$(which pip)
if [[ $PIP_PATH != *"/envs/$ENV_NAME/"* ]]; then
    echo -e "${RED}❌ Critical Safety Error: System pip detected instead of Conda pip.${NC}"
    exit 1
fi

if [ -f requirements.txt ]; then
    echo -e "${GREEN}📦 Installing from requirements.txt...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo -e "${RED}⚠️ requirements.txt missing! Falling back to manual install...${NC}"
    conda install numpy matplotlib -y
fi

# 5. Final Status Report
echo -e "\n${GREEN}========================================================${NC}"
echo -e "${GREEN}✅ SETUP COMPLETE - ENVIRONMENT HERMETICALLY SEALED${NC}"
echo -e "${GREEN}========================================================${NC}"