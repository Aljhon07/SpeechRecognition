# # Use Python 3.11 as base image
# FROM python:3.11-slim
FROM sephirah07/lingua-speech:base
# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     wget \
#     build-essential \
#     pkg-config \
#     cmake \
#     curl \
#     gcc \
#     g++ \
#     libc6-dev \
#     git \
#     alsa-utils \
#     && rm -rf /var/lib/apt/lists/*

# # Try to install sentencepiece from wheel first, fallback to source build
# RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
#     (pip install --no-cache-dir sentencepiece || \
#     (echo "Wheel installation failed, building from source..." && \
#      git clone https://github.com/google/sentencepiece.git /tmp/sentencepiece && \
#      cd /tmp/sentencepiece && \
#      mkdir build && \
#      cd build && \
#      cmake .. && \
#      make -j$(nproc) && \
#      make install && \
#      ldconfig && \
#      cd ../python && \
#      pip install . && \
#      cd / && \
#      rm -rf /tmp/sentencepiece))

# # Set working directory
WORKDIR /app

# # Copy requirements first for better caching
# COPY requirements.txt ./

# # Install Python dependencies
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# Copy only the required source directories
COPY src/ ./src/
COPY inference/ ./inference/
COPY tools/ ./tools/
COPY config.py ./
COPY __init__.py ./

# Create required directories for data persistence
RUN mkdir -p /app/output /app/logs /app/uploads /app/commonvoice

# RUN apt-get purge -y \
#     build-essential \
#     gcc \
#     g++ \
#     cmake \
#     pkg-config \
#     git \
#     libc6-dev \
#     && apt-get autoremove -y && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

# Expose the Flask app port
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_ENV=production
ENV DOCKER_ENV=1
ENV POST_PROCESS_WITH_AI=false

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Default command to run the Flask app
CMD ["python", "-m", "inference.app"]
