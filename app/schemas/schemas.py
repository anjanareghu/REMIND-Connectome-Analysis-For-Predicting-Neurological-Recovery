from pydantic import BaseModel, EmailStr, validator
from datetime import datetime, date
from typing import Optional
from enum import Enum

# Enums for validation
class ExerciseFrequency(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    rarely = "rarely"
    never = "never"

class SmokingStatus(str, Enum):
    never = "never"
    former = "former"
    current = "current"

class AlcoholConsumption(str, Enum):
    never = "never"
    rarely = "rarely"
    moderate = "moderate"
    frequent = "frequent"

class DietType(str, Enum):
    omnivore = "omnivore"
    vegetarian = "vegetarian"
    vegan = "vegan"
    keto = "keto"
    paleo = "paleo"
    mediterranean = "mediterranean"
    other = "other"

class StressLevel(str, Enum):
    low = "low"
    moderate = "moderate"
    high = "high"

# User schemas
class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    profile_completed: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    profile_completed: bool
    redirect_to: Optional[str] = None

class TokenData(BaseModel):
    username: Optional[str] = None

# Profile schemas
class ProfileCreate(BaseModel):
    # Personal Information
    first_name: str
    last_name: str
    age: int
    gender: str
    
    # Physical Measurements
    height: Optional[float] = None  # in cm
    weight: Optional[float] = None  # in kg
    bmi: Optional[float] = None
    waist_circumference: Optional[float] = None
    
    # Medical Information
    blood_type: Optional[str] = None
    family_history_stroke: Optional[str] = None
    family_history_heart_disease: Optional[str] = None
    medications: Optional[str] = None
    
    # Health Conditions
    hypertension: Optional[bool] = False
    diabetes: Optional[bool] = False
    heart_disease: Optional[bool] = False
    atrial_fibrillation: Optional[bool] = False
    
    # Lifestyle Information
    smoking_status: Optional[str] = None
    alcohol_consumption: Optional[str] = None
    exercise_frequency: Optional[str] = None
    stress_level: Optional[int] = None
    sleep_hours: Optional[float] = None
    diet_type: Optional[str] = None
    
    # Additional Information
    additional_notes: Optional[str] = None

    @validator('age')
    def validate_age(cls, v):
        if v < 1 or v > 120:
            raise ValueError('Age must be between 1 and 120')
        return v
    
    @validator('height')
    def validate_height(cls, v):
        if v is not None and (v < 50 or v > 300):
            raise ValueError('Height must be between 50 and 300 cm')
        return v
    
    @validator('weight')
    def validate_weight(cls, v):
        if v is not None and (v < 10 or v > 500):
            raise ValueError('Weight must be between 10 and 500 kg')
        return v
    
    @validator('stress_level')
    def validate_stress_level(cls, v):
        if v is not None and (v < 1 or v > 10):
            raise ValueError('Stress level must be between 1 and 10')
        return v
    
    @validator('sleep_hours')
    def validate_sleep_hours(cls, v):
        if v is not None and (v < 0 or v > 24):
            raise ValueError('Sleep hours must be between 0 and 24')
        return v

class ProfileUpdate(BaseModel):
    # Personal Information
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    
    # Contact Information
    phone_number: Optional[str] = None
    address: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    
    # Physical Measurements
    height: Optional[float] = None  # in cm
    weight: Optional[float] = None  # in kg
    
    # Medical Information
    blood_type: Optional[str] = None
    allergies: Optional[str] = None
    medical_conditions: Optional[str] = None
    current_medications: Optional[str] = None
    medical_history: Optional[str] = None
    
    # Lifestyle Information
    exercise_frequency: Optional[str] = None
    smoking_status: Optional[str] = None
    alcohol_consumption: Optional[str] = None
    diet_type: Optional[str] = None
    sleep_hours: Optional[int] = None
    stress_level: Optional[str] = None
    additional_notes: Optional[str] = None

class ProfileResponse(BaseModel):
    id: int
    user_id: int
    
    # Personal Information
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    date_of_birth: Optional[date] = None
    
    # Contact Information
    profile_picture: Optional[str] = None
    phone_number: Optional[str] = None
    address: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    
    # Physical Measurements
    height: Optional[float] = None
    weight: Optional[float] = None
    height_cm: Optional[int] = None
    weight_kg: Optional[int] = None
    bmi: Optional[float] = None
    
    # Medical Information
    blood_type: Optional[str] = None
    allergies: Optional[str] = None
    medical_conditions: Optional[str] = None
    current_medications: Optional[str] = None
    medical_history: Optional[str] = None
    
    # Lifestyle Information
    exercise_frequency: Optional[str] = None
    smoking_status: Optional[str] = None
    alcohol_consumption: Optional[str] = None
    diet_type: Optional[str] = None
    sleep_hours: Optional[int] = None
    stress_level: Optional[str] = None
    additional_notes: Optional[str] = None
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# File schemas
class FileUploadResponse(BaseModel):
    id: int
    filename: str
    original_filename: str
    file_size: int
    content_type: str
    uploaded_by: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Stroke Analysis schemas
class StrokeAnalysisResponse(BaseModel):
    id: int
    user_id: int
    mri_file_id: Optional[int] = None
    analysis_type: str
    confidence_score: int
    findings: str
    recommendations: Optional[str] = None
    severity_level: str
    brain_regions_affected: Optional[str] = None
    lesion_volume_ml: Optional[int] = None
    status: str
    reviewed_by_doctor: bool
    doctor_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class StrokeAnalysisCreate(BaseModel):
    mri_file_id: Optional[int] = None
    analysis_type: str = "stroke_detection"
    findings: str
    recommendations: Optional[str] = None
    severity_level: str
    brain_regions_affected: Optional[str] = None
    lesion_volume_ml: Optional[int] = None

# Recovered Patient schemas
class RecoveredPatientResponse(BaseModel):
    id: int
    patient_name: str
    age: int
    gender: str
    stroke_type: str
    initial_severity: str
    treatment_duration_days: int
    recovery_percentage: int
    treatment_methods: str
    success_story: str
    before_condition: str
    after_condition: str
    key_factors: str
    location: str
    recovery_completed_date: date
    created_at: datetime
    
    class Config:
        from_attributes = True

# Health Advice schemas
class HealthAdviceRequest(BaseModel):
    focus_area: Optional[str] = None  # "stroke_prevention", "general", "recovery", etc.

class HealthAdviceResponse(BaseModel):
    advice_type: str
    title: str
    content: str
    recommendations: list[str]
    risk_factors: list[str]
    lifestyle_tips: list[str]
    generated_at: datetime

# Dashboard Summary schemas
class DashboardSummary(BaseModel):
    user_profile_completed: bool
    total_analyses: int
    latest_analysis: Optional[StrokeAnalysisResponse] = None
    health_score: int  # 0-100 based on profile
    risk_level: str  # "low", "medium", "high"
    recommendations_count: int
    recovered_patients_count: int
