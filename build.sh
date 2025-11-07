#!/bin/bash
# Build script for BitNet.cpp binary compilation
# This automates the compilation of Microsoft's BitNet 1.58-bit quantization kernels

set -e  # Exit on error

echo "======================================"
echo "BitNet.cpp Build Script"
echo "======================================"
echo ""

# Check if binary already exists
BINARY_PATH="bin/BitNet/build/bin/llama-cli"
if [ -f "$BINARY_PATH" ]; then
    echo "✓ BitNet binary already exists at: $BINARY_PATH"
    
    # Non-interactive mode: skip rebuild unless FORCE_REBUILD=1
    if [ -z "$FORCE_REBUILD" ]; then
        echo "Skipping build. Use existing binary."
        echo "To force rebuild: FORCE_REBUILD=1 ./build.sh"
        exit 0
    fi
    
    echo "Force rebuild requested..."
fi

echo "Step 1: Setting up BitNet repository..."
if [ ! -d "bin/BitNet" ]; then
    echo "  Cloning BitNet repository..."
    mkdir -p bin
    git clone --recursive https://github.com/microsoft/BitNet.git bin/BitNet
else
    echo "  ✓ BitNet repository exists"
fi

echo ""
echo "Step 2: Configuring build environment..."
cd bin/BitNet

# Create build directory
mkdir -p build
cd build

echo ""
echo "Step 3: Running CMake configuration..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo ""
echo "Step 4: Compiling BitNet binary (this may take a few minutes)..."
cmake --build . --config Release -j $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "======================================"
if [ -f "bin/llama-cli" ]; then
    echo "✓ Build successful!"
    echo "  Binary location: $(pwd)/bin/llama-cli"
    echo "  Relative path: bin/BitNet/build/bin/llama-cli"
    echo ""
    echo "You can now use BitNet GGUF models with the application."
else
    echo "✗ Build failed - binary not found"
    exit 1
fi
echo "======================================"
