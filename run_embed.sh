#!/usr/bin/env bash
#
# Run AORG embedding generation using nix-shell
#
# This script uses NixOS nix-shell to provide all required Python packages
# with proper CUDA support and library linking.
#

set -e  # Exit on error

echo "================================================================"
echo "PDF Malware Detection - AORG Embedding Generation"
echo "================================================================"
echo

# Configuration
PROJECT_DIR="/home/belovedemperor/class/Comp Sec/ComputerSecurityProjectGit"
SEMESTER_DIR="/home/belovedemperor/class/Comp Sec/semester project/PDFObj2Vec"

BERT_MODEL="${SEMESTER_DIR}/models/BERT20k.pth"
VOCAB="${SEMESTER_DIR}/models/vocab-20k"

# Check if models exist
if [ ! -f "$BERT_MODEL" ]; then
    echo "ERROR: BERT model not found at: $BERT_MODEL"
    exit 1
fi

if [ ! -f "$VOCAB" ]; then
    echo "ERROR: Vocabulary not found at: $VOCAB"
    exit 1
fi

echo "[Step 1/2] Processing TRAIN set..."
echo "------------------------------------"
nix-shell \
    -p python313Packages.torch \
       python313Packages.transformers \
       python313Packages.torch-geometric \
       python313Packages.scikit-learn \
       python313Packages.tqdm \
       python313Packages.numpy \
       python313Packages.pandas \
    --run "cd '${PROJECT_DIR}' && python scripts/embed_aorg.py \
        -r data/train \
        -v '${VOCAB}' \
        -b '${BERT_MODEL}' \
        -label data/train/labels.csv"

echo
echo "[Step 2/2] Processing TEST set..."
echo "------------------------------------"
nix-shell \
    -p python313Packages.torch \
       python313Packages.transformers \
       python313Packages.torch-geometric \
       python313Packages.scikit-learn \
       python313Packages.tqdm \
       python313Packages.numpy \
       python313Packages.pandas \
    --run "cd '${PROJECT_DIR}' && python scripts/embed_aorg.py \
        -r data/test \
        -v '${VOCAB}' \
        -b '${BERT_MODEL}' \
        -label data/test/labels.csv"

echo
echo "================================================================"
echo "AORG EMBEDDING GENERATION COMPLETE"
echo "================================================================"
echo "Train AORGs: ${PROJECT_DIR}/data/train/org_after_prebert/"
echo "Test AORGs:  ${PROJECT_DIR}/data/test/org_after_prebert/"
echo "================================================================"
