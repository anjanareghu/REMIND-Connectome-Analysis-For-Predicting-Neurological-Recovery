from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from datetime import datetime, date
from typing import Optional
import os
import uuid
import shutil
from pathlib import Path

from app.database.database import get_db
from app.models.models import User, UserProfile, UploadedFile
from app.schemas.schemas import ProfileCreate, ProfileUpdate, ProfileResponse
from app.routers.auth import get_current_user

router = APIRouter()

# Create profile pictures directory if it doesn't exist
PROFILE_PICS_DIRECTORY = "uploads/profile_pictures"
Path(PROFILE_PICS_DIRECTORY).mkdir(parents=True, exist_ok=True)

def calculate_age(birth_date: date) -> int:
    """Calculate age from date of birth."""
    today = date.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

def calculate_bmi(height_cm: Optional[int], weight_kg: Optional[int]) -> Optional[float]:
    """Calculate BMI from height and weight."""
    if height_cm and weight_kg:
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 2)
    return None

def get_profile_by_user_id(db: Session, user_id: int):
    """Get user profile by user ID."""
    return db.query(UserProfile).filter(UserProfile.user_id == user_id).first()

@router.post("/create", response_model=ProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_profile(
    profile: ProfileCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create user profile (first-time setup)."""
    # Check if profile already exists
    existing_profile = get_profile_by_user_id(db, current_user.id)
    if existing_profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Profile already exists. Use update endpoint instead."
        )
    
    # Convert age to date_of_birth (approximate)
    from datetime import date
    current_year = date.today().year
    birth_year = current_year - profile.age
    approximate_birth_date = date(birth_year, 1, 1)
    
    # Create new profile with mapped fields
    db_profile = UserProfile(
        user_id=current_user.id,
        full_name=f"{profile.first_name} {profile.last_name}",
        date_of_birth=approximate_birth_date,
        phone_number=None,  # Not collected in current form
        address=None,  # Not collected in current form
        emergency_contact_name=None,  # Not collected in current form
        emergency_contact_phone=None,  # Not collected in current form
        blood_type=profile.blood_type,
        allergies=None,  # Not collected in current form
        medical_conditions=f"Hypertension: {profile.hypertension}, Diabetes: {profile.diabetes}, Heart Disease: {profile.heart_disease}, Atrial Fibrillation: {profile.atrial_fibrillation}",
        current_medications=profile.medications,
        medical_history=f"Family History - Stroke: {profile.family_history_stroke}, Heart Disease: {profile.family_history_heart_disease}",
        height_cm=int(profile.height) if profile.height else None,
        weight_kg=int(profile.weight) if profile.weight else None,
        exercise_frequency=profile.exercise_frequency,
        smoking_status=profile.smoking_status,
        alcohol_consumption=profile.alcohol_consumption,
        diet_type=profile.diet_type,
        sleep_hours=int(profile.sleep_hours) if profile.sleep_hours else None,
        stress_level=profile.stress_level
    )
    
    db.add(db_profile)
    
    # Mark user profile as completed
    current_user.profile_completed = True
    
    db.commit()
    db.refresh(db_profile)
    
    # Calculate age and BMI for response
    age = calculate_age(db_profile.date_of_birth)
    bmi = calculate_bmi(db_profile.height_cm, db_profile.weight_kg)
    
    # Create response with calculated fields
    response_data = ProfileResponse(
        id=db_profile.id,
        user_id=db_profile.user_id,
        full_name=db_profile.full_name,
        date_of_birth=db_profile.date_of_birth,
        age=age,
        profile_picture=db_profile.profile_picture,
        phone_number=db_profile.phone_number,
        address=db_profile.address,
        emergency_contact_name=db_profile.emergency_contact_name,
        emergency_contact_phone=db_profile.emergency_contact_phone,
        blood_type=db_profile.blood_type,
        allergies=db_profile.allergies,
        medical_conditions=db_profile.medical_conditions,
        current_medications=db_profile.current_medications,
        medical_history=db_profile.medical_history,
        height_cm=db_profile.height_cm,
        weight_kg=db_profile.weight_kg,
        bmi=bmi,
        exercise_frequency=db_profile.exercise_frequency,
        smoking_status=db_profile.smoking_status,
        alcohol_consumption=db_profile.alcohol_consumption,
        diet_type=db_profile.diet_type,
        sleep_hours=db_profile.sleep_hours,
        stress_level=db_profile.stress_level,
        created_at=db_profile.created_at,
        updated_at=db_profile.updated_at
    )
    
    return response_data

@router.get("/me", response_model=ProfileResponse)
async def get_my_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's profile."""
    profile = get_profile_by_user_id(db, current_user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found. Please create your profile first."
        )
    
    # Calculate age and BMI
    age = calculate_age(profile.date_of_birth) if profile.date_of_birth else None
    bmi = calculate_bmi(profile.height_cm, profile.weight_kg)
    
    # Extract first and last name from full_name
    name_parts = profile.full_name.split() if profile.full_name else []
    first_name = name_parts[0] if name_parts else None
    last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else None
    
    # Create response with calculated fields
    response_data = ProfileResponse(
        id=profile.id,
        user_id=profile.user_id,
        first_name=first_name,
        last_name=last_name,
        full_name=profile.full_name,
        age=age,
        gender=profile.gender,
        date_of_birth=profile.date_of_birth,
        profile_picture=profile.profile_picture,
        phone_number=profile.phone_number,
        address=profile.address,
        emergency_contact_name=profile.emergency_contact_name,
        emergency_contact_phone=profile.emergency_contact_phone,
        blood_type=profile.blood_type,
        allergies=profile.allergies,
        medical_conditions=profile.medical_conditions,
        current_medications=profile.current_medications,
        medical_history=profile.medical_history,
        height=float(profile.height_cm) if profile.height_cm else None,
        weight=float(profile.weight_kg) if profile.weight_kg else None,
        height_cm=profile.height_cm,
        weight_kg=profile.weight_kg,
        bmi=bmi,
        exercise_frequency=profile.exercise_frequency,
        smoking_status=profile.smoking_status,
        alcohol_consumption=profile.alcohol_consumption,
        diet_type=profile.diet_type,
        sleep_hours=profile.sleep_hours,
        stress_level=profile.stress_level,
        additional_notes=None,  # Not stored in current model
        created_at=profile.created_at,
        updated_at=profile.updated_at
    )
    
    return response_data

@router.put("/update", response_model=ProfileResponse)
async def update_profile(
    profile_update: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user profile."""
    profile = get_profile_by_user_id(db, current_user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found. Please create your profile first."
        )
    
    # Handle first_name and last_name combination to full_name
    if profile_update.first_name is not None or profile_update.last_name is not None:
        first_name = profile_update.first_name or (profile.full_name.split()[0] if profile.full_name else "")
        last_name = profile_update.last_name or (profile.full_name.split()[-1] if profile.full_name and len(profile.full_name.split()) > 1 else "")
        profile.full_name = f"{first_name} {last_name}".strip()
    
    # Handle age to date_of_birth conversion
    if profile_update.age is not None:
        from datetime import date
        current_year = date.today().year
        birth_year = current_year - profile_update.age
        profile.date_of_birth = date(birth_year, 1, 1)
    
    # Handle height and weight (frontend sends as height/weight, backend stores as height_cm/weight_kg)
    if profile_update.height is not None:
        profile.height_cm = int(profile_update.height)
    if profile_update.weight is not None:
        profile.weight_kg = int(profile_update.weight)
    
    # Handle gender field specifically
    if profile_update.gender is not None:
        profile.gender = profile_update.gender
    
    # Update other fields
    update_data = profile_update.dict(exclude_unset=True, exclude={'first_name', 'last_name', 'age', 'height', 'weight', 'gender'})
    for field, value in update_data.items():
        if hasattr(profile, field):
            setattr(profile, field, value)
    
    db.commit()
    db.refresh(profile)
    
    # Calculate age and BMI for response
    age = calculate_age(profile.date_of_birth) if profile.date_of_birth else None
    bmi = calculate_bmi(profile.height_cm, profile.weight_kg)
    
    # Extract first and last name from full_name
    name_parts = profile.full_name.split() if profile.full_name else []
    first_name = name_parts[0] if name_parts else None
    last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else None
    
    # Create response with calculated fields
    response_data = ProfileResponse(
        id=profile.id,
        user_id=profile.user_id,
        first_name=first_name,
        last_name=last_name,
        full_name=profile.full_name,
        age=age,
        gender=profile.gender,
        date_of_birth=profile.date_of_birth,
        profile_picture=profile.profile_picture,
        phone_number=profile.phone_number,
        address=profile.address,
        emergency_contact_name=profile.emergency_contact_name,
        emergency_contact_phone=profile.emergency_contact_phone,
        blood_type=profile.blood_type,
        allergies=profile.allergies,
        medical_conditions=profile.medical_conditions,
        current_medications=profile.current_medications,
        medical_history=profile.medical_history,
        height=float(profile.height_cm) if profile.height_cm else None,
        weight=float(profile.weight_kg) if profile.weight_kg else None,
        height_cm=profile.height_cm,
        weight_kg=profile.weight_kg,
        bmi=bmi,
        exercise_frequency=profile.exercise_frequency,
        smoking_status=profile.smoking_status,
        alcohol_consumption=profile.alcohol_consumption,
        diet_type=profile.diet_type,
        sleep_hours=profile.sleep_hours,
        stress_level=profile.stress_level,
        created_at=profile.created_at,
        updated_at=profile.updated_at
    )
    
    return response_data

@router.post("/upload-picture")
async def upload_profile_picture(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload profile picture."""
    # Check if profile exists
    profile = get_profile_by_user_id(db, current_user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found. Please create your profile first."
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    # Generate unique filename
    file_extension = Path(file.filename).suffix
    unique_filename = f"profile_{current_user.id}_{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(PROFILE_PICS_DIRECTORY, unique_filename)
    
    # Remove old profile picture if exists
    if profile.profile_picture and os.path.exists(profile.profile_picture):
        try:
            os.remove(profile.profile_picture)
        except Exception as e:
            print(f"Error removing old profile picture: {e}")
    
    # Save new file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving file: {str(e)}"
        )
    
    # Update profile with new picture path
    profile.profile_picture = file_path
    db.commit()
    
    return {"message": "Profile picture uploaded successfully", "file_path": file_path}

@router.delete("/delete-picture")
async def delete_profile_picture(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete profile picture."""
    profile = get_profile_by_user_id(db, current_user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found"
        )
    
    if not profile.profile_picture:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No profile picture found"
        )
    
    # Remove file from disk
    try:
        if os.path.exists(profile.profile_picture):
            os.remove(profile.profile_picture)
    except Exception as e:
        print(f"Error deleting profile picture: {e}")
    
    # Update database
    profile.profile_picture = None
    db.commit()
    
    return {"message": "Profile picture deleted successfully"}

@router.get("/check")
async def check_profile_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Check if user has completed their profile."""
    return {
        "profile_completed": current_user.profile_completed,
        "redirect_to": None if current_user.profile_completed else "/profile/create"
    }
