FROM python:3.13

# Install apt packages
RUN apt-get update && \
    apt-get install -y procps git curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install PyTorch (CPU build for Linux)
RUN uv pip install --system \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install PyG extension wheels (must match torch version)
RUN uv pip install --system \
    torch-scatter torch-sparse torch-cluster \
    --find-links https://data.pyg.org/whl/torch-2.10.0+cpu.html

# Install torch-geometric and core ML dependencies
RUN uv pip install --system \
    torch-geometric \
    pytorch-lightning \
    wandb \
    yacs \
    scvi-tools

# Install geome
RUN uv pip install --system geome

# Install interscale from local source (pulls in all remaining dependencies from pyproject.toml)
WORKDIR /opt/interscale
COPY . .
RUN uv pip install --system .
