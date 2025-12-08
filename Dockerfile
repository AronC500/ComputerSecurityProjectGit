# PDF Malware Detection - Training Environment
# Python 3.10 + PyTorch + DGL for GIN training

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1
RUN pip3 install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric (required for loading .pt files)
RUN pip3 install torch_geometric==2.4.0

# Install torchdata (required by DGL)
RUN pip3 install torchdata==0.7.1

# Install DGL for CUDA 12.1
RUN pip3 install dgl==2.1.0+cu121 -f https://data.dgl.ai/wheels/cu121/repo.html

# Install scientific computing packages
RUN pip3 install \
    numpy==1.26.4 \
    scipy==1.12.0 \
    pandas==2.2.0 \
    scikit-learn==1.5.0 \
    matplotlib==3.9.0 \
    seaborn==0.13.0 \
    tqdm==4.66.0 \
    pyyaml==6.0.1 \
    pydantic==2.10.5 \
    psutil

# Set working directory
WORKDIR /workspace

# Set Python to use UTF-8 encoding
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Default command
CMD ["/bin/bash"]
