#!/bin/bash

# Check if 'uv' is installed, and install it if it's missing
if ! command -v uv &> /dev/null; then
    echo "uv is not installed, installing it..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create a virtual environment using 'uv' inside the '.venv' folder
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install the dependencies from pyproject.toml
uv sync

# Confirm the environment setup is complete
echo "Virtual environment setup complete, dependencies installed."