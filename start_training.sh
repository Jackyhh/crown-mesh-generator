#!/bin/bash

# Script to start training once ShapeNet dataset is downloaded
# This script will be ready to run after the dataset download completes

echo "=== Shape As Points Training Setup ==="
echo "Checking dataset availability..."

if [ ! -d "data/shapenet_psr" ]; then
    echo "⚠️  ShapeNet dataset not found at data/shapenet_psr"
    echo "Please wait for the dataset download to complete"
    exit 1
fi

echo "✅ ShapeNet dataset found!"
echo "Setting environment variables..."

export MKL_SERVICE_FORCE_INTEL=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Starting training with config: configs/learning_based/noise_large/ours.yaml"
echo "This will train a model for large noise point cloud reconstruction"
echo ""
echo "Training output will be saved to: out/shapenet/noise_025_ours/"
echo "Training logs, models, and visualizations will be available there"
echo ""
echo "Press Ctrl+C to stop training at any time"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Start training with output to both terminal and log file
echo "Training logs will be saved to: logs/training_$(date +%Y%m%d_%H%M%S).log"
echo "You can also monitor the logs in real-time here in the terminal"
echo ""

# Start training in background with nohup - survives terminal disconnection
LOGFILE="logs/training_$(date +%Y%m%d_%H%M%S).log"
echo "Starting training in background..."
echo "Log file: $LOGFILE"
echo "PID will be saved to: logs/training.pid"

nohup /root/anaconda3/envs/py310-dmc/bin/python train.py configs/learning_based/noise_large/ours.yaml > "$LOGFILE" 2>&1 &
TRAINING_PID=$!
echo $TRAINING_PID > logs/training.pid

echo ""
echo "✅ Training started in background!"
echo "   PID: $TRAINING_PID"
echo "   Log: $LOGFILE"
echo ""
echo "To monitor progress: tail -f $LOGFILE"
echo "To check if running: ps -p $TRAINING_PID"
echo "To stop training: kill $TRAINING_PID"
echo ""
echo "Training will continue even if you disconnect from the server."