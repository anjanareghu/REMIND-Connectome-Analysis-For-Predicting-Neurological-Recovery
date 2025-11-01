@echo off
REM FastAPI Docker Quick Start Script for Windows
REM This script helps you quickly start the FastAPI application using Docker

echo =====================================
echo   FastAPI Docker Quick Start
echo =====================================
echo.

REM Check if Docker is running
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not installed!
    echo Please install Docker Desktop and make sure it's running.
    echo Download from: https://docs.docker.com/desktop/windows/install/
    pause
    exit /b 1
)

echo Docker is running... ✓
echo.

echo Starting FastAPI application...
echo This may take a few minutes on first run (downloading dependencies)
echo.

REM Start the application
docker-compose up --build

echo.
echo Application stopped.
pause