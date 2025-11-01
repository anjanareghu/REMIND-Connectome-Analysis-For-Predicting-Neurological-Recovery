#!/bin/bash
# FastAPI Docker Quick Start Script for Linux/macOS
# This script helps you quickly start the FastAPI application using Docker

echo "====================================="
echo "   FastAPI Docker Quick Start"
echo "====================================="
echo

# Check if Docker is running
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed!"
    echo "Please install Docker first:"
    echo "- Linux: https://docs.docker.com/engine/install/"
    echo "- macOS: https://docs.docker.com/desktop/mac/install/"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "ERROR: Docker is not running!"
    echo "Please start Docker and try again."
    exit 1
fi

echo "Docker is running... ✓"
echo

echo "Starting FastAPI application..."
echo "This may take a few minutes on first run (downloading dependencies)"
echo

# Start the application
docker-compose up --build

echo
echo "Application stopped."