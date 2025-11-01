# Docker Setup Scripts for FastAPI User Management API

## Prerequisites
Ensure Docker Desktop is installed and running on your system.

### For Windows:
- Download Docker Desktop from https://docs.docker.com/desktop/windows/install/
- Install and start Docker Desktop
- Verify installation: `docker --version`

### For macOS:
- Download Docker Desktop from https://docs.docker.com/desktop/mac/install/
- Install and start Docker Desktop
- Verify installation: `docker --version`

### For Linux:
- Install Docker Engine: https://docs.docker.com/engine/install/
- Start Docker service: `sudo systemctl start docker`
- Verify installation: `docker --version`

## Quick Start Commands

### 1. Build and Run with Docker Compose (Recommended)
```bash
# Start the application in production mode
docker-compose up --build

# Start in development mode with hot reload
docker-compose --profile dev up --build fastapi-dev

# Run in background (detached mode)
docker-compose up -d --build
```

### 2. Build and Run with Docker Commands
```bash
# Build the image
docker build -t fastapi-user-management .

# Run the container with volume mounts for data persistence
docker run -d \
  --name fastapi-app \
  -p 8000:8000 \
  -v ${PWD}/uploads:/app/uploads \
  -v ${PWD}/app.db:/app/app.db \
  -v ${PWD}/templates:/app/templates \
  -v ${PWD}/models_predict:/app/models_predict \
  -v ${PWD}/mriscans:/app/mriscans \
  fastapi-user-management

# View logs
docker logs fastapi-app

# Stop the container
docker stop fastapi-app
docker rm fastapi-app
```

### 3. Development Commands
```bash
# View running containers
docker ps

# View all containers
docker ps -a

# View logs
docker-compose logs -f

# Execute commands inside container
docker exec -it fastapi-app bash

# Stop all services
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Rebuild without cache
docker-compose build --no-cache
```

## Troubleshooting

### Common Issues:

1. **Port already in use**:
   ```bash
   # Check what's using port 8000
   netstat -an | findstr :8000
   # Change port in docker-compose.yml or stop conflicting service
   ```

2. **Permission issues**:
   - On Linux/macOS: `sudo chown -R $(whoami):$(whoami) uploads/`
   - On Windows: Ensure Docker has access to the project directory

3. **Database issues**:
   - Delete `app.db` to reset database
   - Ensure database file has proper permissions

4. **Memory issues**:
   - Increase Docker Desktop memory allocation
   - Close other applications

### Verification Steps:
1. Check if Docker is running: `docker --version`
2. Test container health: `docker ps` (should show healthy status)
3. Test API: Visit `http://localhost:8000/docs`
4. Check logs: `docker-compose logs fastapi-app`

## Production Deployment

For production deployment, consider:

1. **Environment Variables**: Create `.env` file for sensitive data
2. **Reverse Proxy**: Use Nginx or Traefik
3. **SSL/TLS**: Configure HTTPS certificates
4. **Database**: Use PostgreSQL or MySQL instead of SQLite
5. **Monitoring**: Add logging and monitoring solutions
6. **Backup**: Implement data backup strategies

Example production docker-compose override:
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  fastapi-app:
    restart: always
    environment:
      - ENVIRONMENT=production
    volumes:
      - /var/log/fastapi:/app/logs
```

Run with: `docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d`