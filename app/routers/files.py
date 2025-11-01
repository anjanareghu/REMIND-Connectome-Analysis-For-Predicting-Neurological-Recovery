from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import os
import uuid
import shutil
from pathlib import Path

from app.database.database import get_db
from app.models.models import User, UploadedFile
from app.schemas.schemas import FileUploadResponse
from app.routers.auth import get_current_user

router = APIRouter()

# Create uploads directory if it doesn't exist
UPLOAD_DIRECTORY = "uploads"
Path(UPLOAD_DIRECTORY).mkdir(exist_ok=True)

@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a file (requires authentication)."""
    # Generate unique filename
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)
    
    # Save file to disk
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving file: {str(e)}"
        )
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # Save file metadata to database
    db_file = UploadedFile(
        filename=unique_filename,
        original_filename=file.filename,
        file_path=file_path,
        file_size=file_size,
        content_type=file.content_type,
        uploaded_by=current_user.id
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    
    return db_file

@router.get("/", response_model=List[FileUploadResponse])
async def get_user_files(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all files uploaded by the current user."""
    files = db.query(UploadedFile).filter(UploadedFile.uploaded_by == current_user.id).all()
    return files

@router.get("/{file_id}", response_model=FileUploadResponse)
async def get_file_info(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get information about a specific file."""
    file_record = db.query(UploadedFile).filter(
        UploadedFile.id == file_id,
        UploadedFile.uploaded_by == current_user.id
    ).first()
    
    if file_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    return file_record

@router.delete("/{file_id}")
async def delete_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a file (only the uploader can delete their files)."""
    file_record = db.query(UploadedFile).filter(
        UploadedFile.id == file_id,
        UploadedFile.uploaded_by == current_user.id
    ).first()
    
    if file_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    # Delete file from disk
    try:
        if os.path.exists(file_record.file_path):
            os.remove(file_record.file_path)
    except Exception as e:
        # Log the error but don't fail the request
        print(f"Error deleting file from disk: {e}")
    
    # Delete from database
    db.delete(file_record)
    db.commit()
    
    return {"message": "File deleted successfully"}
