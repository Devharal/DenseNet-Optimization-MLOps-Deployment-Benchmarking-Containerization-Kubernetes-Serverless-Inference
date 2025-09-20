# # Use NVIDIA CUDA base image with Python 3.10
# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04



# # Set environment variables
# ENV PYTHONUNBUFFERED=1
# ENV DEBIAN_FRONTEND=noninteractive
# ENV CUDA_VISIBLE_DEVICES=0
# ENV VIRTUAL_ENV=/opt/venv
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-dev \
#     python3-venv \
#     curl \
#     wget \
#     git \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# # Create virtual environment and upgrade pip
# RUN python3 -m venv $VIRTUAL_ENV \
#     && $VIRTUAL_ENV/bin/pip install --upgrade pip uv

# # Copy requirements and install Python dependencies inside venv
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# # Copy source code
# COPY main.py .

# # Create necessary directories
# RUN mkdir -p /app/results/profiles /app/results/models /app/logs/tensorboard

# # Set permissions
# RUN chmod +x main.py

# # Expose TensorBoard port
# EXPOSE 6006

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD python3 -c "import torch; print('Health check passed')" || exit 1

# # Default command
# CMD ["python3", "main.py", "--output-dir", "/app/results", "--gpu-enabled", "true"]


# Use NVIDIA CUDA base image with Python 3.10
# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# # Set environment variables
# ENV PYTHONUNBUFFERED=1
# ENV DEBIAN_FRONTEND=noninteractive
# ENV CUDA_VISIBLE_DEVICES=0

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-dev \
#     python3-pip \
#     curl \
#     wget \
#     git \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*


# # Copy requirements and install Python dependencies
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# # Copy source code
# COPY main.py .

# # Create necessary directories with proper permissions
# RUN mkdir -p /app/results/profiles /app/results/models /app/logs/tensorboard && \
#     chmod 755 /app/results /app/logs && \
#     chmod -R 755 /app/results /app/logs

# # Set permissions
# RUN chmod +x main.py

# # Expose TensorBoard port
# EXPOSE 6006

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD python3 -c "import torch; print('Health check passed')" || exit 1

# # Default command
# CMD ["python3", "main.py", "--output-dir", "/app/results", "--gpu-enabled", "true"]

# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Define virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-venv \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create venv and upgrade pip
RUN python3 -m venv $VIRTUAL_ENV && \
    $VIRTUAL_ENV/bin/pip install --upgrade pip

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY main.py .

# Create necessary directories
RUN mkdir -p /app/results/profiles /app/results/models /app/logs/tensorboard && \
    chmod -R 755 /app/results /app/logs

# Expose TensorBoard port
EXPOSE 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; import pandas; print('Health check passed')" || exit 1

# Default command
CMD ["python3", "main.py", "--output-dir", "/app/results", "--gpu-enabled", "true"]
