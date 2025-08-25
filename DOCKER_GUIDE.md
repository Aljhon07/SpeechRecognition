# Docker Build and Deployment Guide

## Overview

This guide covers building and deploying the Speech Recognition application with Whisper integration using Docker.

## System Requirements

- Docker Desktop 4.0+ or Docker Engine 20.10+
- Docker Compose 2.0+
- Minimum 4GB RAM
- 10GB free disk space (for model downloads and data)

## Build Instructions

### 1. Quick Build (Recommended)

```powershell
# Clone and navigate to project
cd d:\Projects\SpeechRecognition

# Build the Docker image
docker-compose build
```

### 2. Build with Cache Optimization

```powershell
# Build with no cache (clean build)
docker-compose build --no-cache

# Build with parallel processing
docker-compose build --parallel
```

### 3. Environment-Specific Builds

#### Development Build

```powershell
docker-compose -f docker-compose.dev.yml build
```

#### Production Build

```powershell
docker-compose -f docker-compose.prod.yml build
```

## Deployment Options

### 1. Standard Deployment

```powershell
# Start the main application
docker-compose up -d speech-recognition

# View logs
docker-compose logs -f speech-recognition

# Stop the application
docker-compose down
```

### 2. Development Deployment

```powershell
# Start with live reloading
docker-compose -f docker-compose.dev.yml up speech-recognition-dev
```

### 3. Production Deployment

```powershell
# Start production-optimized container
docker-compose -f docker-compose.prod.yml up -d
```

### 4. Training Deployment

```powershell
# Preprocess data
docker-compose --profile preprocess up

# Train model
docker-compose --profile training up
```

## Environment Variables

Create a `.env` file in the project root:

```env
# Required: Google GenAI API Key
GENAI_API_KEY=your-api-key-here

# Optional: Flask settings
FLASK_ENV=production
FLASK_DEBUG=0
```

## Troubleshooting

### Common Issues

1. **Build fails with "sentencepiece" error**

   - The Dockerfile includes fallback compilation from source
   - Ensure you have sufficient disk space (>2GB during build)

2. **Whisper model download timeout**

   - The build includes automatic Whisper model caching
   - For slow connections, increase Docker build timeout:

   ```powershell
   $env:DOCKER_BUILDKIT=1
   docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1
   ```

3. **Port 8080 already in use**

   ```powershell
   # Find process using port 8080
   netstat -ano | findstr :8080

   # Change port in docker-compose.yml
   ports:
     - "8081:8080"  # Use port 8081 instead
   ```

4. **Out of memory during build**
   - Increase Docker Desktop memory limit to 6GB+
   - Or use multi-stage build optimization

### Performance Optimization

1. **Enable BuildKit** (faster builds)

   ```powershell
   $env:DOCKER_BUILDKIT=1
   docker-compose build
   ```

2. **Use .dockerignore** (already included)

   - Excludes unnecessary files from build context
   - Reduces build time and image size

3. **Layer Caching**
   - Requirements are installed before code copy
   - Code changes don't rebuild Python dependencies

## Container Management

### Useful Commands

```powershell
# View running containers
docker-compose ps

# Check container health
docker-compose exec speech-recognition curl http://localhost:8080/

# Access container shell
docker-compose exec speech-recognition bash

# View resource usage
docker stats

# Clean up unused containers/images
docker system prune -a
```

### Volume Management

```powershell
# Backup trained models
docker cp speech-recognition-app:/app/output ./backup/

# Import pretrained models
docker cp ./models/ speech-recognition-app:/app/inference/models/
```

## Security Considerations

1. **API Key Security**

   - Never commit `.env` files
   - Use Docker secrets in production
   - Rotate API keys regularly

2. **Network Security**

   - Run behind reverse proxy (nginx/traefik)
   - Use HTTPS in production
   - Configure firewall rules

3. **Container Security**
   - Use non-root user (TODO: implement)
   - Regular security updates
   - Scan images for vulnerabilities

## Production Checklist

- [ ] Environment variables configured
- [ ] SSL/TLS certificates in place
- [ ] Backup strategy for models and data
- [ ] Monitoring and logging configured
- [ ] Resource limits set appropriately
- [ ] Health checks functioning
- [ ] Load balancer configured (if multiple instances)
