from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routers import auth, users, files, profile, dashboard, stroke_analysis
from app.database.database import engine, Base
from app.services.data_seeder import initialize_dummy_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize dummy data
initialize_dummy_data()

app = FastAPI(
    title="FastAPI User Management API",
    description="A comprehensive FastAPI application with user authentication, profile management, stroke analysis, and AI-powered health insights",
    version="1.0.0"
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"REQUEST: {request.method} {request.url}")
    response = await call_next(request)
    print(f"RESPONSE: {response.status_code}")
    return response

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://localhost:8000"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(profile.router, prefix="/api/profile", tags=["Profile"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(stroke_analysis.router, tags=["Stroke Analysis"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(files.router, prefix="/api/files", tags=["Files"])

@app.get("/")
async def root():
    return {"message": "FastAPI User Management API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/login")
async def login_page():
    """Serve the login page."""
    return FileResponse("templates/login.html")

@app.get("/profile/create")
async def profile_create_page():
    """Serve the profile creation page."""
    return FileResponse("templates/profile_create.html")

@app.get("/profile")
async def profile_view_page():
    """Serve the profile view page."""
    return FileResponse("templates/profile_view.html")

@app.get("/profile/edit")
async def profile_edit_page():
    """Serve the profile edit page."""
    return FileResponse("templates/profile_edit.html")

@app.get("/dashboard")
async def dashboard_page():
    """Serve the dashboard page."""
    return FileResponse("templates/dashboard.html")

@app.get("/stroke-analysis")
async def stroke_analysis_page():
    """Serve the stroke analysis page."""
    return FileResponse("templates/stroke_analysis.html")
