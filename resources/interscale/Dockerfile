FROM python:3.13

# Install apt packages
RUN apt-get update && \
    apt-get install -y procps git && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip (26+ has a significantly improved resolver)
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU-only (must come before torch-geometric)
RUN pip install --no-cache-dir \
    "torch>=2" \
    --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric (depends on torch being installed first)
RUN pip install --no-cache-dir torch-geometric

# Pin numpy early so subsequent resolvers don't fight over it
RUN pip install --no-cache-dir "numpy>=1.19,<2"

# Install core scientific stack
RUN pip install --no-cache-dir \
    anndata \
    pandas \
    "scipy>=1.7" \
    scikit-learn \
    scanpy

# Install torch extensions
RUN pip install --no-cache-dir \
    "lightning>=2" \
    "torchmetrics>=1.8"

# Install bio-specific packages
RUN pip install --no-cache-dir \
    "scvi-tools>=0.19" \
    squidpy

# Install geome with the SAME index URL
RUN pip install --no-cache-dir \
    geome \
    --index-url https://download.pytorch.org/whl/cpu

# Install utilities
RUN pip install --no-cache-dir \
    wandb \
    yacs \
    session-info2 \
    "fast-array-utils[accel]" \
    jupyterlab

# Install interscale
RUN pip install --no-cache-dir git+https://github.com/theislab/interscale.git
