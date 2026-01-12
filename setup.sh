#!/bin/bash
# Setup script for microCT inference pipeline
# This script installs dependencies and sets up the environment

set -e

echo "=========================================="
echo "microCT Inference Pipeline Setup"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python: $python_version"

# Check if Python 3.10 or higher
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "ERROR: Python 3.10 or higher is required. Found: $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install nnUNet in development mode
echo ""
echo "Installing nnUNet..."
pip install -e .

# Install additional requirements
if [ -f "requirements.txt" ]; then
    echo ""
    echo "Installing additional requirements..."
    pip install -r requirements.txt
fi

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Obtain model weights (see MODEL_WEIGHTS.md for details):"
echo "   - Model weights are NOT included in this repository"
echo "   - You need to copy them from the training location"
echo "   - See MODEL_WEIGHTS.md for instructions"
echo ""
echo "3. Set up nnUNet environment variables:"
echo "   export nnUNet_results=\"/path/to/your/nnUNet_results\""
echo ""
echo "4. Verify model weights are set up correctly:"
echo "   python3 verify_weights.py"
echo ""
echo "5. Prepare your data:"
echo "   python3 prepare_data_for_inference.py --input_dir /path/to/data --output_dir /path/to/output"
echo ""
echo "6. Run inference:"
echo "   python3 run_inference.py --input_dir /path/to/prepared/data --output_dir /path/to/predictions"
echo ""
