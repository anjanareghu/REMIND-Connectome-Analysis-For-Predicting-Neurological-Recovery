from sqlalchemy import Column, Integer, String, DateTime, Boolean, Date, Text
from sqlalchemy.sql import func
from app.database.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    profile_completed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, unique=True)  # Foreign key to User
    full_name = Column(String, nullable=False)
    date_of_birth = Column(Date, nullable=False)
    gender = Column(String, nullable=True)  # Male, Female, Other
    profile_picture = Column(String, nullable=True)  # File path to profile picture
    phone_number = Column(String, nullable=True)
    address = Column(Text, nullable=True)
    emergency_contact_name = Column(String, nullable=True)
    emergency_contact_phone = Column(String, nullable=True)
    
    # Medical Information
    blood_type = Column(String, nullable=True)
    allergies = Column(Text, nullable=True)
    medical_conditions = Column(Text, nullable=True)
    current_medications = Column(Text, nullable=True)
    medical_history = Column(Text, nullable=True)
    
    # Lifestyle Information
    height_cm = Column(Integer, nullable=True)
    weight_kg = Column(Integer, nullable=True)
    exercise_frequency = Column(String, nullable=True)  # daily, weekly, monthly, rarely, never
    smoking_status = Column(String, nullable=True)  # never, former, current
    alcohol_consumption = Column(String, nullable=True)  # never, rarely, moderate, frequent
    diet_type = Column(String, nullable=True)  # omnivore, vegetarian, vegan, etc.
    sleep_hours = Column(Integer, nullable=True)
    stress_level = Column(String, nullable=True)  # low, moderate, high
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class UploadedFile(Base):
    __tablename__ = "uploaded_files"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String, nullable=False)
    uploaded_by = Column(Integer, nullable=False)  # User ID
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class StrokeAnalysis(Base):
    __tablename__ = "stroke_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)  # Foreign key to User
    mri_file_id = Column(Integer, nullable=True)  # Foreign key to UploadedFile
    analysis_type = Column(String, nullable=False)  # "stroke_detection", "risk_assessment", etc.
    confidence_score = Column(Integer, nullable=False)  # 0-100
    findings = Column(Text, nullable=False)  # AI analysis results
    recommendations = Column(Text, nullable=True)  # Treatment recommendations
    severity_level = Column(String, nullable=False)  # "low", "medium", "high", "critical"
    brain_regions_affected = Column(Text, nullable=True)  # JSON string of affected regions
    lesion_volume_ml = Column(Integer, nullable=True)  # Lesion volume in milliliters
    status = Column(String, default="pending")  # "pending", "completed", "reviewed"
    reviewed_by_doctor = Column(Boolean, default=False)
    doctor_notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class RecoveredPatient(Base):
    __tablename__ = "recovered_patients"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String, nullable=False)  # Anonymized name
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)  # "male", "female", "other"
    stroke_type = Column(String, nullable=False)  # "ischemic", "hemorrhagic", "tia"
    initial_severity = Column(String, nullable=False)  # "mild", "moderate", "severe"
    treatment_duration_days = Column(Integer, nullable=False)
    recovery_percentage = Column(Integer, nullable=False)  # 0-100
    treatment_methods = Column(Text, nullable=False)  # JSON string of treatments used
    success_story = Column(Text, nullable=False)  # Patient's recovery story
    before_condition = Column(Text, nullable=False)  # Initial condition description
    after_condition = Column(Text, nullable=False)  # Final condition description
    key_factors = Column(Text, nullable=False)  # JSON string of success factors
    location = Column(String, nullable=False)  # City/Country for privacy
    recovery_completed_date = Column(Date, nullable=False)
    
    # Additional detailed information for full reports
    initial_symptoms = Column(Text, nullable=True)  # JSON string of initial symptoms
    treatment_timeline = Column(Text, nullable=True)  # JSON string of treatment milestones
    rehabilitation_details = Column(Text, nullable=True)  # JSON string of rehab activities
    support_system = Column(Text, nullable=True)  # Family/medical support details
    lifestyle_changes = Column(Text, nullable=True)  # Changes made during recovery
    current_status = Column(Text, nullable=True)  # Current health status
    inspiring_quote = Column(Text, nullable=True)  # Patient's inspiring message
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
