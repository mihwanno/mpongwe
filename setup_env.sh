#!/bin/bash
set -e

# Check if 'uv' is installed, and install it if it's missing.
# If 'uv' exists, self-update it: old versions mis-parse pyproject.toml
# (e.g. "invalid type: string ..., expected a sequence").
if ! command -v uv &> /dev/null; then
    echo "uv is not installed, installing it..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "Updating uv to the latest version..."
    uv self-update || true
fi

# Remove a broken venv from a previous failed run (stale lock/parse errors)
if [ -d ".venv" ]; then
    echo "Removing existing .venv for a clean setup..."
    rm -rf .venv
fi

# Create a virtual environment using 'uv' inside the '.venv' folder.
# Pin Python 3.12: Studio default (e.g. 3.14) has no wheels for
# faiss-cpu / bitsandbytes / torch, and `requires-python = ">=3.12"`
# makes uv pick the newest version (3.14) unless pinned.
# `uv venv --python 3.12` auto-downloads 3.12 if missing.
# `--seed` installs pip too: without it `python -m pip` fails with
# "No module named pip" (uv venvs are pip-less by default), which breaks
# `src/finetune/run_pipeline.py install_dependencies()`.
if uv venv --seed --python 3.12 2>/dev/null; then
    echo "Created venv with Python 3.12."
else
    echo "Python 3.12 not available, using default Python..."
    uv venv --seed
fi

# Pin for future uv commands so it doesn't default back to 3.14
uv python pin 3.12 2>/dev/null || true

# Activate the virtual environment
source .venv/bin/activate

# Install training dependencies from requirements.txt (not `uv sync`).
# `uv sync` resolves the optional `comet` extra too, which is currently
# unsatisfiable (unbabel-comet needs numpy<2, langchain-community==0.4.1
# needs numpy>=2.1). requirements.txt skips COMET by default.
uv pip install -r requirements.txt

# For COMET evaluation (optional, incompatible with default numpy>=2):
# uv pip install "unbabel-comet>=2.0.2" "numpy<2" --resolution=lowest-direct
# Confirm the environment setup is complete
echo "Virtual environment setup complete, dependencies installed."