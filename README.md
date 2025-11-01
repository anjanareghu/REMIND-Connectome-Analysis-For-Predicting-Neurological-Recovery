# FastAPI AI-Powered Health Management System - Docker Distribution

A comprehensive FastAPI application with user authentication, profile management, AI-powered health insights, stroke analysis, and recovery tracking, fully containerized with Docker.

## 🌟 Features

- **User Registration & Authentication**: JWT-based authentication system
- **Comprehensive Profile Management**: Detailed user profiles with medical and lifestyle information
- **AI-Powered Health Advice**: Google Gemini AI integration for personalized health recommendations
- **Stroke Analysis & Risk Assessment**: MRI analysis tracking and stroke risk evaluation
- **Recovery Success Stories**: Inspiring patient recovery journeys with detailed case studies
- **Interactive Dashboard**: Complete web interface with real-time health insights
- **Fully Dockerized**: Easy deployment with Docker and Docker Compose

## 🐳 Docker Quick Start

### Prerequisites
- Docker Desktop installed and running
- At least 4GB RAM available for Docker

### 1. Start the Application
```bash
# Production mode (recommended)
docker-compose up --build

# Development mode with hot reload
docker-compose --profile dev up --build fastapi-dev

# Run in background
docker-compose up -d --build
```

### 2. Access the Application
- **API Documentation**: http://localhost:8000/docs
- **Application**: http://localhost:8000/login
- **Dashboard**: http://localhost:8000/dashboard

### 3. Stop the Application
```bash
# Stop services
docker-compose down

# Stop and remove volumes (resets database)
docker-compose down -v
```

## 📁 Project Structure

```
fastapi-docker-distribution/
├── app/                        # FastAPI application code
│   ├── routers/               # API endpoints
│   ├── models/                # Database models
│   ├── schemas/               # Data validation schemas
│   ├── database/              # Database configuration
│   ├── services/              # Business logic services
│   └── auth.py                # Authentication utilities
├── templates/                 # HTML templates
├── models_predict/            # ML models for stroke prediction
├── uploads/                   # File upload directories
│   ├── mri_scans/            # MRI scan uploads
│   └── profile_pictures/     # Profile picture uploads
├── mriscans/                  # Sample MRI data
├── Dockerfile                 # Docker image configuration
├── docker-compose.yml         # Docker services configuration
├── .dockerignore             # Docker build exclusions
├── main.py                   # Application entry point
├── requirements.txt          # Python dependencies
├── DOCKER_SETUP.md          # Detailed Docker guide
└── README.md                 # This file
```

## 🚀 Usage

### First Time Setup
1. **Start the application**: `docker-compose up --build`
2. **Open browser**: Navigate to http://localhost:8000/login
3. **Register**: Create a new user account
4. **Complete profile**: Fill out your health profile
5. **Explore dashboard**: Access AI-powered health insights

### API Endpoints

#### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user info

#### Profile Management
- `POST /api/profile/create` - Create user profile
- `GET /api/profile/me` - Get user profile
- `PUT /api/profile/update` - Update profile
- `POST /api/profile/upload-picture` - Upload profile picture

#### AI Dashboard
- `GET /api/dashboard/summary` - Dashboard overview
- `POST /api/dashboard/health-advice` - AI health advice
- `GET /api/dashboard/stroke-analyses` - Stroke analysis history
- `GET /api/dashboard/recovered-patients` - Recovery stories

### Development Mode

For development with hot reload:
```bash
docker-compose --profile dev up --build fastapi-dev
```
Access at: http://localhost:8001

## 🔧 Configuration

### Environment Variables
Create a `.env` file for custom configuration:
```env
SECRET_KEY=your-secret-key-here
GOOGLE_API_KEY=your-gemini-api-key
DATABASE_URL=sqlite:///./app.db
```

### Custom Ports
Edit `docker-compose.yml` to change ports:
```yaml
ports:
  - "8080:8000"  # Change 8080 to your preferred port
```

## 📊 Health Features

### AI-Powered Insights
- Personalized health recommendations
- Stroke risk assessment
- Recovery timeline predictions
- Lifestyle optimization suggestions

### Data Collection
- Medical history and conditions
- Lifestyle factors (exercise, diet, sleep)
- Biometric data (height, weight, BMI)
- Emergency contact information

## 🛠️ Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Check what's using port 8000
   netstat -an | findstr :8000
   # Stop conflicting service or change port in docker-compose.yml
   ```

2. **Docker not running**:
   - Start Docker Desktop
   - Verify with: `docker --version`

3. **Memory issues**:
   - Increase Docker Desktop memory allocation
   - Close other applications

4. **Permission errors**:
   - Ensure Docker has access to the project directory
   - On Linux/macOS: `sudo chown -R $(whoami):$(whoami) uploads/`

### View Logs
```bash
# View all logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# View specific service logs
docker-compose logs fastapi-app
```

### Reset Database
```bash
# Stop services and remove volumes
docker-compose down -v

# Restart (will create fresh database)
docker-compose up --build
```

## 📋 Production Deployment

For production deployment:

1. **Set environment variables** for sensitive data
2. **Use external database** (PostgreSQL/MySQL)
3. **Configure reverse proxy** (Nginx/Traefik)
4. **Enable HTTPS** with SSL certificates
5. **Set up monitoring** and logging
6. **Implement backup strategy**

Example production override:
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  fastapi-app:
    restart: always
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://user:pass@db:5432/dbname
```

Run with: `docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d`

## 🔒 Security

- JWT tokens with 30-minute expiration
- Password hashing with bcrypt
- Input validation with Pydantic
- Non-root container user
- File upload restrictions

## 📖 API Documentation

Once running, access interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤝 Support

For issues or questions:
1. Check the logs: `docker-compose logs`
2. Review `DOCKER_SETUP.md` for detailed troubleshooting
3. Ensure all prerequisites are met
4. Verify Docker Desktop is running and has sufficient resources

## 📄 License

This project is provided as-is for demonstration and learning purposes.

---

**Quick Start Command**: `docker-compose up --build`  
**Access URL**: http://localhost:8000/docs