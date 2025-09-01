#!/bin/bash

echo "=== Shape As Points - Complete Setup Script ==="
echo "This script will download and setup the ShapeNet dataset for training"
echo "Supports resuming interrupted downloads automatically"
echo ""

# Set working directory
cd /workspace/github/shape_as_points

# Function to cleanup partial downloads on script interruption
cleanup() {
    echo ""
    echo "⚠️  Script interrupted!"
    echo "Partial download preserved and can be resumed by re-running this script"
    exit 1
}

# Trap interruption signals
trap cleanup INT TERM

# Create data directory
mkdir -p data

echo "📦 Downloading ShapeNet dataset (~85GB compressed, ~220GB uncompressed)..."
echo "This will take some time depending on your internet connection"
echo ""

cd data

# Download with resume capability and progress monitoring
echo "Checking for existing partial download..."
if [ -f "shapenet_psr.zip" ]; then
    echo "Found partial download, resuming..."
fi

wget --progress=bar:force:noscroll \
     --continue \
     --timeout=30 \
     --tries=3 \
     https://s3.eu-central-1.amazonaws.com/avg-projects/shape_as_points/data/shapenet_psr.zip

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "❌ Download failed!"
    echo "Please check your internet connection and try again"
    echo "Partial download has been preserved and can be resumed"
    exit 1
fi

echo ""
echo "✅ Download completed!"
echo "🔍 Verifying download integrity..."

# Verify the download is complete (basic size check)
if [ ! -f "shapenet_psr.zip" ]; then
    echo "❌ Download file not found!"
    exit 1
fi

# Check if file is not empty
if [ ! -s "shapenet_psr.zip" ]; then
    echo "❌ Downloaded file is empty!"
    echo "Removing corrupted file..."
    rm -f shapenet_psr.zip
    exit 1
fi

echo "📦 Extracting dataset..."

# Extract the archive
unzip -q shapenet_psr.zip

# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo "❌ Extraction failed!"
    echo "The zip file might be corrupted. Please delete it and re-run this script"
    echo "To delete: rm -f data/shapenet_psr.zip"
    exit 1
fi

# Clean up zip file
rm shapenet_psr.zip

cd ..

echo ""
echo "✅ Dataset setup completed!"
echo ""
echo "📊 Dataset statistics:"
echo "Directory: $(du -sh data/shapenet_psr 2>/dev/null || echo 'Calculating...')"
echo ""

# Verify dataset structure
if [ -d "data/shapenet_psr" ]; then
    echo "🎯 Dataset verification:"
    echo "   Categories found: $(ls data/shapenet_psr | wc -l)"
    echo "   Sample categories: $(ls data/shapenet_psr | head -3 | tr '\n' ' ')"
    echo ""
    echo "✅ Training environment is ready!"
    echo ""
    echo "🚀 To start training, run:"
    echo "   ./start_training.sh"
    echo ""
    echo "📝 Training configuration: configs/learning_based/noise_large/ours.yaml"
    echo "📁 Output directory: out/shapenet/noise_025_ours/"
else
    echo "❌ Dataset directory not found after extraction"
    echo "Please check the extraction process"
    exit 1
fi

echo ""
echo "🎉 Setup complete! Happy training! 🎉"