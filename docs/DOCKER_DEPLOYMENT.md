# Docker Deployment Guide

This guide covers deploying the IT Support Chatbot using Docker and Docker Compose.

## Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose (usually included with Docker Desktop)
- Your Anthropic API key

## Quick Start

### 1. Prepare Files

```bash
# Clone or create project directory
mkdir it-support-chatbot
cd it-support-chatbot

# Copy all project files
# - Dockerfile
# - docker-compose.yml
# - requirements.txt
# - it_support_chatbot_claude_api.py
# - app_claude.py

# Create .env file
cp .env.template .env
# Edit .env and add your API key
```

### 2. Add Documents

```bash
# Create documents folder
mkdir documents

# Copy your PDF files
cp /path/to/your/pdfs/*.pdf documents/
```

### 3. Build and Run

```bash
# Build the image
docker-compose build

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f
```

### 4. Access the Application

Open browser: http://localhost:8501

## Docker Commands

### Build
```bash
# Build image
docker-compose build

# Build without cache
docker-compose build --no-cache

# Build with specific compose file
docker-compose -f docker-compose.yml build
```

### Run
```bash
# Start in detached mode
docker-compose up -d

# Start with logs visible
docker-compose up

# Start specific service
docker-compose up chatbot
```

### Stop
```bash
# Stop services
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove everything including volumes
docker-compose down -v
```

### Logs
```bash
# View logs
docker-compose logs

# Follow logs
docker-compose logs -f

# Logs for specific service
docker-compose logs -f chatbot

# Last 100 lines
docker-compose logs --tail=100
```

### Shell Access
```bash
# Access container shell
docker-compose exec chatbot bash

# Run command in container
docker-compose exec chatbot python --version
```

## Configuration

### Environment Variables

Edit `.env` file:

```bash
ANTHROPIC_API_KEY=your-key-here
CLAUDE_MODEL=claude-sonnet-4-20250514
TEMPERATURE=0.7
```

### Volume Mounts

The docker-compose.yml mounts:

1. **Documents** (read-only):
   ```yaml
   - ./documents:/app/documents:ro
   ```

2. **Vector Index** (persistent):
   ```yaml
   - ./index_faiss:/app/index_faiss
   ```

3. **Logs** (optional):
   ```yaml
   - ./logs:/app/logs
   ```

### Resource Limits

Adjust in docker-compose.yml:

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
```

## Production Deployment

### 1. Security

**API Keys**:
```bash
# Use Docker secrets
docker secret create anthropic_key ./api_key.txt

# Reference in compose
secrets:
  - anthropic_key
```

**Network Security**:
```yaml
# Use internal network
networks:
  internal:
    internal: true
  external:
```

### 2. SSL/TLS

Use reverse proxy (nginx, traefik):

```yaml
# nginx example
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
```

### 3. Health Monitoring

**Health checks**:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

**Monitoring Stack** (optional):
```bash
# Add Prometheus + Grafana
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

### 4. Backup

**Vector Index**:
```bash
# Backup
tar -czf index_backup.tar.gz index_faiss/

# Restore
tar -xzf index_backup.tar.gz
```

**Documents**:
```bash
# Backup documents
cp -r documents/ documents_backup_$(date +%Y%m%d)/
```

### 5. Updates

```bash
# Pull latest code
git pull

# Rebuild
docker-compose build --no-cache

# Restart with new image
docker-compose up -d --force-recreate
```

## Multi-Platform Deployment

### Build for Multiple Platforms

```bash
# Setup buildx
docker buildx create --use

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t it-support-chatbot:latest \
  --push .
```

### Platform-Specific Images

```yaml
# In docker-compose.yml
services:
  chatbot:
    platform: linux/amd64  # or linux/arm64
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs chatbot

# Common issues:
# 1. API key not set
# 2. Documents folder missing
# 3. Port 8501 already in use
```

### Out of Memory

```bash
# Increase memory limit
# Edit docker-compose.yml
memory: 8G  # Increase from 4G

# Check container stats
docker stats
```

### Slow Performance

```bash
# Check resource usage
docker stats

# Rebuild index
docker-compose exec chatbot rm -rf index_faiss
docker-compose restart
```

### Permission Issues

```bash
# Fix volume permissions
sudo chown -R 1000:1000 documents/
sudo chown -R 1000:1000 index_faiss/
```

## Advanced Configurations

### Custom Entrypoint

```dockerfile
# In Dockerfile
ENTRYPOINT ["custom-script.sh"]
CMD ["web"]
```

### Multiple Instances

```bash
# Scale service
docker-compose up -d --scale chatbot=3

# Use load balancer
# Add nginx/traefik configuration
```

### GPU Support (if needed)

```yaml
# In docker-compose.yml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Kubernetes Deployment (Optional)

For large-scale deployments:

```bash
# Convert compose to k8s
kompose convert

# Apply to cluster
kubectl apply -f chatbot-deployment.yaml
```

## Monitoring & Logging

### Application Logs

```bash
# Structured logging
docker-compose logs -f chatbot | jq .

# Export logs
docker-compose logs chatbot > chatbot_logs.txt
```

### Performance Metrics

```bash
# Container stats
docker stats chatbot

# Resource usage over time
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

## Maintenance

### Regular Tasks

1. **Weekly**:
   - Check logs for errors
   - Monitor resource usage
   - Backup vector index

2. **Monthly**:
   - Update dependencies
   - Review and update documents
   - Check for security updates

3. **Quarterly**:
   - Rebuild index from scratch
   - Performance optimization
   - Security audit

### Cleanup

```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Full cleanup
docker system prune -a --volumes
```

## Support

For issues:
1. Check logs: `docker-compose logs -f`
2. Verify configuration: `docker-compose config`
3. Review documentation: README.md

---

**Last Updated**: November 2025
