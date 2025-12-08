#!/usr/bin/env bash
#
# Build Docker image and train GIN model
#

set -e

PROJECT_DIR="/home/belovedemperor/class/Comp Sec/ComputerSecurityProjectGit"

echo "================================================================"
echo "PDF Malware Detection - GIN Training Pipeline"
echo "================================================================"
echo

# Step 1: Build Docker image
echo "[Step 1/3] Building Docker image..."
echo "------------------------------------"
cd "$PROJECT_DIR"
docker build -t pdf-malware-detection .

echo
echo "[Step 2/3] Testing Docker + GPU..."
echo "------------------------------------"
docker run --rm --device nvidia.com/gpu=all pdf-malware-detection nvidia-smi

echo
echo "[Step 3/3] Training GIN model..."
echo "------------------------------------"
docker run --rm \
    --device nvidia.com/gpu=all \
    -v "$PROJECT_DIR/data:/workspace/data:ro" \
    -v "$PROJECT_DIR/models:/workspace/models" \
    -v "$PROJECT_DIR/scripts:/workspace/scripts:ro" \
    -v "$PROJECT_DIR/lib:/workspace/lib:ro" \
    pdf-malware-detection \
    python3 scripts/trainGIN.py \
        --train-dir /workspace/data/train/org_after_prebert \
        --test-dir /workspace/data/test/org_after_prebert \
        --output /workspace/models/GIN-trained.pth \
        --epochs 50 \
        --batch-size 64 \
        --lr 0.001 \
        --weight-decay 0.0001 \
        --hidden-size 256

echo
echo "================================================================"
echo "TRAINING COMPLETE"
echo "================================================================"
echo "Model saved to: $PROJECT_DIR/models/GIN-trained.pth"
echo "Training history: $PROJECT_DIR/models/GIN-trained_history.json"
echo "================================================================"
