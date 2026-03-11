#!/bin/bash

VENV_DIR=".venv"

# Create a virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    /opt/homebrew/bin/python3.13 -m venv "$VENV_DIR"
fi

# Activate the venv
source "$VENV_DIR/bin/activate"

# Install required packages only if they are missing
if ! python -c "import torch, numpy" &> /dev/null; then
    echo "Installing missing dependencies..."
    pip install torch numpy
fi

# Run the app
python main.py
