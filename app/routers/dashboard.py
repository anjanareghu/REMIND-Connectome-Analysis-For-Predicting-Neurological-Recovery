from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, date

from app.database.database import get_db
from app.models.models import User, UserProfile, StrokeAnalysis, RecoveredPatient
from app.schemas.schemas import (
    HealthAdviceRequest, HealthAdviceResponse, StrokeAnalysisResponse, 
    RecoveredPatientResponse, DashboardSummary
)
from app.routers.auth import get_current_user
from app.services.gemini_service import gemini_advisor
from app.services.data_seeder import seed_sample_stroke_analyses, seed_recovered_patients

router = APIRouter()

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

def get_user_profile_dict(user: User, profile: UserProfile) -> dict:
    """Convert user profile to dictionary for AI processing."""
    if not profile:
        return {"age": "unknown"}
    
    age = calculate_age(profile.date_of_birth)
    bmi = calculate_bmi(profile.height_cm, profile.weight_kg)
    
    return {
        "age": age,
        "height_cm": profile.height_cm,
        "weight_kg": profile.weight_kg,
        "bmi": bmi,
        "exercise_frequency": profile.exercise_frequency,
        "sleep_hours": profile.sleep_hours,
        "smoking_status": profile.smoking_status,
        "alcohol_consumption": profile.alcohol_consumption,
        "diet_type": profile.diet_type,
        "stress_level": profile.stress_level,
        "blood_type": profile.blood_type,
        "allergies": profile.allergies,
        "medical_conditions": profile.medical_conditions,
        "current_medications": profile.current_medications,
        "medical_history": profile.medical_history
    }

def calculate_health_score(profile_dict: dict) -> int:
    """Calculate a health score based on user profile (0-100)."""
    score = 50  # Base score
    
    # BMI scoring
    bmi = profile_dict.get("bmi")
    if bmi:
        if 18.5 <= bmi <= 24.9:
            score += 20
        elif 25 <= bmi <= 29.9:
            score += 10
        elif bmi > 30:
            score -= 20
    
    # Exercise scoring
    exercise = profile_dict.get("exercise_frequency")
    if exercise == "daily":
        score += 15
    elif exercise == "weekly":
        score += 10
    elif exercise == "never":
        score -= 15
    
    # Sleep scoring
    sleep_hours = profile_dict.get("sleep_hours")
    if sleep_hours and 7 <= sleep_hours <= 9:
        score += 10
    elif sleep_hours and (sleep_hours < 6 or sleep_hours > 10):
        score -= 10
    
    # Smoking penalty
    if profile_dict.get("smoking_status") == "current":
        score -= 25
    elif profile_dict.get("smoking_status") == "former":
        score -= 5
    
    # Stress penalty
    if profile_dict.get("stress_level") == "high":
        score -= 15
    elif profile_dict.get("stress_level") == "low":
        score += 10
    
    # Ensure score is within bounds
    return max(0, min(100, score))

def get_risk_level(health_score: int) -> str:
    """Determine risk level based on health score."""
    if health_score >= 80:
        return "low"
    elif health_score >= 60:
        return "medium"
    else:
        return "high"

@router.get("/summary", response_model=DashboardSummary)
async def get_dashboard_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get dashboard summary with user's health overview."""
    
    # Get user profile
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    # Get user's stroke analyses
    analyses = db.query(StrokeAnalysis).filter(StrokeAnalysis.user_id == current_user.id).all()
    latest_analysis = analyses[-1] if analyses else None
    
    # Calculate health metrics
    if profile:
        profile_dict = get_user_profile_dict(current_user, profile)
        health_score = calculate_health_score(profile_dict)
        risk_level = get_risk_level(health_score)
    else:
        health_score = 50
        risk_level = "medium"
    
    # Count recovered patients
    recovered_count = db.query(RecoveredPatient).count()
    
    return DashboardSummary(
        user_profile_completed=current_user.profile_completed,
        total_analyses=len(analyses),
        latest_analysis=latest_analysis,
        health_score=health_score,
        risk_level=risk_level,
        recommendations_count=5,  # Could be dynamic based on AI recommendations
        recovered_patients_count=recovered_count
    )

@router.post("/health-advice", response_model=HealthAdviceResponse)
async def get_health_advice(
    request: HealthAdviceRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get AI-generated personalized health advice."""
    
    # Get user profile
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found. Please complete your profile first."
        )
    
    # Convert profile to dictionary for AI processing
    profile_dict = get_user_profile_dict(current_user, profile)
    
    # Get AI-generated advice
    focus_area = request.focus_area or "general"
    advice_data = await gemini_advisor.generate_health_advice(profile_dict, focus_area)
    
    return HealthAdviceResponse(
        advice_type=advice_data["advice_type"],
        title=advice_data["title"],
        content=advice_data["content"],
        recommendations=advice_data["recommendations"],
        risk_factors=advice_data["risk_factors"],
        lifestyle_tips=advice_data["lifestyle_tips"],
        generated_at=datetime.now()
    )

@router.get("/stroke-analyses", response_model=List[StrokeAnalysisResponse])
async def get_stroke_analyses(
    limit: int = Query(default=10, le=50),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's stroke analysis history."""
    
    # Get only actual analyses performed by the user (no dummy data)
    analyses = (
        db.query(StrokeAnalysis)
        .filter(StrokeAnalysis.user_id == current_user.id)
        .order_by(StrokeAnalysis.created_at.desc())
        .limit(limit)
        .all()
    )
    
    return analyses

@router.get("/stroke-analyses/{analysis_id}", response_model=StrokeAnalysisResponse)
async def get_stroke_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific stroke analysis details."""
    
    analysis = (
        db.query(StrokeAnalysis)
        .filter(
            StrokeAnalysis.id == analysis_id,
            StrokeAnalysis.user_id == current_user.id
        )
        .first()
    )
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stroke analysis not found"
        )
    
    return analysis

@router.get("/recovered-patients", response_model=List[RecoveredPatientResponse])
async def get_recovered_patients(
    limit: int = Query(default=20, le=100),
    stroke_type: Optional[str] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get list of recovered patients for inspiration and success stories."""
    
    # Ensure dummy data exists
    if not db.query(RecoveredPatient).first():
        seed_recovered_patients()
    
    # Build query with filters
    query = db.query(RecoveredPatient)
    
    if stroke_type:
        query = query.filter(RecoveredPatient.stroke_type == stroke_type)
    
    if severity:
        query = query.filter(RecoveredPatient.initial_severity == severity)
    
    patients = (
        query.order_by(RecoveredPatient.recovery_percentage.desc())
        .limit(limit)
        .all()
    )
    
    return patients

@router.get("/recovered-patients/{patient_id}", response_model=RecoveredPatientResponse)
async def get_recovered_patient_details(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed information about a specific recovered patient."""
    
    patient = db.query(RecoveredPatient).filter(RecoveredPatient.id == patient_id).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recovered patient not found"
        )
    
    return patient

@router.get("/stroke-risk-assessment")
async def get_stroke_risk_assessment(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get AI-powered stroke risk assessment based on user profile."""
    
    # Get user profile
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found. Please complete your profile first."
        )
    
    # Convert profile to dictionary for AI processing
    profile_dict = get_user_profile_dict(current_user, profile)
    
    # Get AI-generated risk assessment
    risk_assessment = await gemini_advisor.analyze_stroke_risk(profile_dict)
    
    return {
        "user_id": current_user.id,
        "assessment_date": datetime.now(),
        "risk_analysis": risk_assessment,
        "profile_based_score": calculate_health_score(profile_dict)
    }

@router.get("/recovery-insights")
async def get_recovery_insights(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get AI-generated recovery insights based on analysis history."""
    
    # Get user's analysis history
    analyses = (
        db.query(StrokeAnalysis)
        .filter(StrokeAnalysis.user_id == current_user.id)
        .order_by(StrokeAnalysis.created_at.desc())
        .limit(10)
        .all()
    )
    
    if not analyses:
        return {
            "message": "No analysis history available",
            "recommendation": "Complete a stroke risk assessment to get personalized insights"
        }
    
    # Convert analyses to dictionaries for AI processing
    analysis_history = []
    for analysis in analyses:
        analysis_history.append({
            "created_at": analysis.created_at.isoformat(),
            "findings": analysis.findings,
            "severity_level": analysis.severity_level,
            "confidence_score": analysis.confidence_score
        })
    
    # Get AI-generated insights
    insights = await gemini_advisor.generate_recovery_insights(analysis_history)
    
    return {
        "user_id": current_user.id,
        "analysis_count": len(analyses),
        "insights": insights,
        "generated_at": datetime.now()
    }

@router.get("/statistics")
async def get_dashboard_statistics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get general statistics for the dashboard."""
    
    # User's personal stats
    user_analyses = db.query(StrokeAnalysis).filter(StrokeAnalysis.user_id == current_user.id).count()
    
    # Global stats
    total_recovered = db.query(RecoveredPatient).count()
    avg_recovery_rate = db.query(RecoveredPatient).with_entities(
        func.avg(RecoveredPatient.recovery_percentage)
    ).scalar() or 0
    
    # Recovery by severity stats
    recovery_by_severity = {}
    severities = ["mild", "moderate", "severe"]
    for severity in severities:
        avg_recovery = db.query(RecoveredPatient).filter(
            RecoveredPatient.initial_severity == severity
        ).with_entities(
            func.avg(RecoveredPatient.recovery_percentage)
        ).scalar()
        recovery_by_severity[severity] = round(avg_recovery or 0, 1)
    
    return {
        "user_stats": {
            "total_analyses": user_analyses,
            "profile_completed": current_user.profile_completed
        },
        "global_stats": {
            "total_recovered_patients": total_recovered,
            "average_recovery_rate": round(avg_recovery_rate, 1),
            "recovery_by_severity": recovery_by_severity
        },
        "generated_at": datetime.now()
    }

@router.get("/success-stories", response_model=List[RecoveredPatientResponse])
def get_success_stories(
    limit: int = Query(default=6, le=20),
    severity_filter: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get recovered patient success stories for dashboard display.
    Only shows stories if user has completed at least one stroke analysis,
    and filters by similar stroke characteristics."""
    
    # Check if user has completed at least one stroke analysis
    user_analyses = db.query(StrokeAnalysis).filter(
        StrokeAnalysis.user_id == current_user.id
    ).all()
    
    if not user_analyses:
        # Return empty list if no analysis completed yet
        return []
    
    # Get the latest stroke analysis for filtering
    latest_analysis = db.query(StrokeAnalysis).filter(
        StrokeAnalysis.user_id == current_user.id
    ).order_by(StrokeAnalysis.created_at.desc()).first()
    
    # Get user profile for demographic filtering
    user_profile = db.query(UserProfile).filter(
        UserProfile.user_id == current_user.id
    ).first()
    
    # Base query
    query = db.query(RecoveredPatient)
    
    # Filter by similar stroke characteristics if we have analysis data
    if latest_analysis:
        # Filter by same severity level or one level above/below for hope
        target_severities = [latest_analysis.severity_level]
        if latest_analysis.severity_level == "low":
            target_severities.extend(["mild"])
        elif latest_analysis.severity_level == "medium":
            target_severities.extend(["mild", "moderate"])
        elif latest_analysis.severity_level == "high":
            target_severities.extend(["moderate", "severe"])
        elif latest_analysis.severity_level == "critical":
            target_severities.extend(["severe"])
        
        query = query.filter(RecoveredPatient.initial_severity.in_(target_severities))
    
    # Apply additional severity filter if provided
    if severity_filter:
        query = query.filter(RecoveredPatient.initial_severity == severity_filter)
    
    # Filter by similar age group if profile exists
    if user_profile and user_profile.date_of_birth:
        from datetime import date
        age = (date.today() - user_profile.date_of_birth).days // 365
        # Filter patients within ±15 years age range
        query = query.filter(
            RecoveredPatient.age.between(max(0, age - 15), age + 15)
        )
    
    # Filter by same gender if profile exists
    if user_profile and user_profile.gender:
        query = query.filter(RecoveredPatient.gender == user_profile.gender.lower())
    
    # Order by recovery percentage (highest first) and limit results
    recovered_patients = query.order_by(
        RecoveredPatient.recovery_percentage.desc(),
        RecoveredPatient.created_at.desc()
    ).limit(limit).all()
    
    # If no patients found with filters, get some general success stories
    if not recovered_patients:
        # Remove strict filters and try again
        query = db.query(RecoveredPatient)
        if latest_analysis:
            # At least keep severity filter
            target_severities = [latest_analysis.severity_level]
            if latest_analysis.severity_level in ["low", "medium"]:
                target_severities.extend(["mild", "moderate"])
            elif latest_analysis.severity_level in ["high", "critical"]:
                target_severities.extend(["moderate", "severe"])
            query = query.filter(RecoveredPatient.initial_severity.in_(target_severities))
        
        recovered_patients = query.order_by(
            RecoveredPatient.recovery_percentage.desc(),
            RecoveredPatient.created_at.desc()
        ).limit(limit).all()
    
    # If still no patients found, seed some data
    if not recovered_patients:
        seed_recovered_patients()
        recovered_patients = db.query(RecoveredPatient).order_by(
            RecoveredPatient.recovery_percentage.desc(),
            RecoveredPatient.created_at.desc()
        ).limit(limit).all()
    
    return recovered_patients

@router.get("/success-stories/{patient_id}/detailed-report")
async def get_detailed_recovery_report(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate detailed recovery report for a specific patient using Gemini AI."""
    
    # Get the patient data
    patient = db.query(RecoveredPatient).filter(RecoveredPatient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get the latest stroke analysis for context (if any)
    latest_analysis = db.query(StrokeAnalysis).filter(
        StrokeAnalysis.user_id == current_user.id
    ).order_by(StrokeAnalysis.created_at.desc()).first()
    
    # Prepare patient data for Gemini
    patient_data = {
        "patient_name": patient.patient_name,
        "age": patient.age,
        "gender": patient.gender,
        "stroke_type": patient.stroke_type,
        "initial_severity": patient.initial_severity,
        "treatment_duration_days": patient.treatment_duration_days,
        "recovery_percentage": patient.recovery_percentage,
        "location": patient.location,
        "success_story": patient.success_story,
        "before_condition": patient.before_condition,
        "after_condition": patient.after_condition,
        "recovery_completed_date": patient.recovery_completed_date.isoformat() if patient.recovery_completed_date else None,
        # Handle new fields gracefully (might be None for existing records)
        "initial_symptoms": patient.initial_symptoms if hasattr(patient, 'initial_symptoms') and patient.initial_symptoms else "Classic stroke symptoms",
        "support_system": patient.support_system if hasattr(patient, 'support_system') and patient.support_system else "Strong family and medical support",
        "lifestyle_changes": patient.lifestyle_changes if hasattr(patient, 'lifestyle_changes') and patient.lifestyle_changes else "Adopted healthier lifestyle",
        "current_status": patient.current_status if hasattr(patient, 'current_status') and patient.current_status else "Living well post-recovery",
        "inspiring_quote": patient.inspiring_quote if hasattr(patient, 'inspiring_quote') and patient.inspiring_quote else "Recovery is possible with determination."
    }
    
    # Prepare latest analysis data for context
    analysis_context = None
    if latest_analysis:
        analysis_context = {
            "severity_level": latest_analysis.severity_level,
            "confidence_score": latest_analysis.confidence_score,
            "brain_regions_affected": latest_analysis.brain_regions_affected,
            "created_at": latest_analysis.created_at.isoformat()
        }
    
    # Generate detailed report using Gemini
    detailed_report = await gemini_advisor.generate_detailed_recovery_report(
        patient_data, analysis_context
    )
    
    # Add patient basic info to the response
    return {
        "patient_info": {
            "id": patient.id,
            "name": patient.patient_name,
            "age": patient.age,
            "gender": patient.gender,
            "stroke_type": patient.stroke_type,
            "initial_severity": patient.initial_severity,
            "recovery_percentage": patient.recovery_percentage,
            "location": patient.location,
            "recovery_date": patient.recovery_completed_date
        },
        "detailed_report": detailed_report,
        "generated_at": datetime.now()
    }
