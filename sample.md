### Unified Docker Implementation

The Speech Recognition project uses a single, flexible Docker setup:

**Docker Files:**

- `Dockerfile` - Container configuration
- `docker-compose.yml` - Single compose file for all use cases
- `requirements.txt` - Python dependencies
- `.dockerignore` - Build exclusions
- `.env` - Environment configuration
- `docker-run.sh` / `docker-run.bat` - Management scripts

**Container Structure:**

```
/app/
├── src/           # Core source code (copied into container)
├── inference/     # Inference and Flask app (copied into container)
├── tools/         # Utility modules (copied into container)
├── config.py      # Configuration file (copied into container)
└── (mounted volumes)
    ├── output/    # Model outputs and checkpoints
    ├── logs/      # Training and application logs
    ├── uploads/   # Audio file uploads
    └── commonvoice/ # Training dataset
```

**Flexible Configuration:**

- **Development Mode**: Source code mounted as volumes for live editing
- **Production Mode**: Comment out source volume mounts (code from image only)
- **Environment**: Configure via `.env` file (FLASK_ENV, FLASK_DEBUG)

**Usage:**

```bash
# Build and start
./docker-run.sh build
./docker-run.sh start

# Training
./docker-run.sh train

# View logs
./docker-run.sh logs

# Access app at http://localhost:8080
```

**Switching Modes:**

- Edit `.env` file to change FLASK_ENV (development/production)
- Comment/uncomment source volume mounts in docker-compose.yml for production
