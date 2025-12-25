#!/bin/bash
# Simple script to run the robot arm demo

cd "$(dirname "$0")"

echo "Running robot arm demo..."
echo ""

# Check if rerun is installed
if ! python -c "import rerun" 2>/dev/null; then
    echo "❌ rerun-sdk is not installed."
    echo ""
    echo "Install it with:"
    echo "  pip install rerun-sdk"
    echo ""
    echo "Or if you're in a conda environment:"
    echo "  conda install -c conda-forge rerun-sdk"
    echo ""
    exit 1
fi

# Check if numpy is installed
if ! python -c "import numpy" 2>/dev/null; then
    echo "❌ numpy is not installed."
    echo "Install it with: pip install numpy"
    exit 1
fi

echo "✓ Dependencies found"
echo ""

# Run with default steps or use argument
STEPS=${1:-100}
echo "Running with $STEPS steps..."
python main.py --steps "$STEPS"


