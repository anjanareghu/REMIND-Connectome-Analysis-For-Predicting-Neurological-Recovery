from sqlalchemy.orm import Session
from datetime import datetime, date, timedelta
import json
import random
from app.database.database import SessionLocal
from app.models.models import RecoveredPatient, StrokeAnalysis

def seed_recovered_patients():
    """Seed the database with dummy recovered patient data."""
    
    db = SessionLocal()
    
    # Check if data already exists
    if db.query(RecoveredPatient).first():
        print("Recovered patients data already exists. Skipping seeding.")
        db.close()
        return
    
    recovered_patients_data = [
        {
            "patient_name": "Sarah M.",
            "age": 58,
            "gender": "female",
            "stroke_type": "ischemic",
            "initial_severity": "moderate",
            "treatment_duration_days": 180,
            "recovery_percentage": 95,
            "treatment_methods": json.dumps([
                "Physical therapy", "Speech therapy", "Occupational therapy", 
                "Medication therapy", "Cognitive rehabilitation"
            ]),
            "success_story": "Sarah experienced an ischemic stroke that affected her speech and right-side mobility. Through dedicated rehabilitation and family support, she regained 95% of her function and returned to her job as a teacher within 6 months.",
            "before_condition": "Severe speech difficulties, partial paralysis on right side, difficulty with fine motor skills",
            "after_condition": "Clear speech with minimal hesitation, full mobility restored, returned to teaching full-time",
            "key_factors": json.dumps([
                "Early intervention", "Consistent therapy attendance", "Strong family support", 
                "Positive attitude", "Gradual return to activities"
            ]),
            "location": "Seattle, WA",
            "recovery_completed_date": date.today() - timedelta(days=30),
            "initial_symptoms": json.dumps([
                "Sudden speech difficulty", "Right arm weakness", "Confusion", "Dizziness"
            ]),
            "treatment_timeline": json.dumps([
                "Day 1: Emergency room, CT scan, clot-dissolving medication",
                "Week 1: Stabilization, speech assessment",
                "Month 1: Intensive physical therapy begun",
                "Month 3: Speech therapy showing major improvements",
                "Month 6: Return to work with accommodations"
            ]),
            "rehabilitation_details": json.dumps([
                "Daily speech exercises", "Right-hand strength training", "Balance work", "Cognitive exercises"
            ]),
            "support_system": "Husband and two adult children provided daily support. Colleagues created a substitute teaching network.",
            "lifestyle_changes": "Adopted heart-healthy diet, started regular walking routine, stress management techniques, medication adherence",
            "current_status": "Teaching full-time, leading stroke survivor support group, excellent quality of life",
            "inspiring_quote": "Every small victory in recovery is actually a huge triumph. Celebrate each step forward."
        },
        {
            "patient_name": "James T.",
            "age": 45,
            "gender": "male",
            "stroke_type": "hemorrhagic",
            "initial_severity": "severe",
            "treatment_duration_days": 365,
            "recovery_percentage": 85,
            "treatment_methods": json.dumps([
                "Emergency neurosurgery", "Intensive care", "Physical therapy", 
                "Occupational therapy", "Cognitive rehabilitation", "Family counseling"
            ]),
            "success_story": "James suffered a severe hemorrhagic stroke requiring emergency surgery. His recovery journey was long but inspiring, demonstrating that determination and proper medical care can overcome even severe strokes.",
            "before_condition": "Coma for 10 days, left-side paralysis, severe cognitive impairment, unable to speak",
            "after_condition": "Walking with minimal assistance, clear speech, returned to modified work duties, independent living",
            "key_factors": json.dumps([
                "Emergency surgery", "Intensive rehabilitation", "Family dedication", 
                "Work accommodation", "Mental health support"
            ]),
            "location": "Austin, TX",
            "recovery_completed_date": date.today() - timedelta(days=60),
            "initial_symptoms": json.dumps([
                "Sudden severe headache", "Loss of consciousness", "Vomiting", "Left side paralysis"
            ]),
            "treatment_timeline": json.dumps([
                "Day 1: Emergency surgery to relieve brain pressure",
                "Days 2-10: Coma, intensive care monitoring",
                "Month 1: Awakening, beginning basic rehabilitation",
                "Month 6: Walking with assistance, speech returning",
                "Month 12: Independent mobility, return to work"
            ]),
            "rehabilitation_details": json.dumps([
                "Intensive physical therapy", "Speech pathology", "Occupational therapy", "Cognitive training"
            ]),
            "support_system": "Wife took leave from work, children helped with daily activities, employer provided flexible work arrangements.",
            "lifestyle_changes": "Complete diet overhaul, regular exercise program, stress reduction, blood pressure monitoring",
            "current_status": "Working part-time as software developer, driving again, active in stroke advocacy",
            "inspiring_quote": "I learned that the brain's ability to heal is remarkable. Never underestimate your potential for recovery."
        },
        {
            "patient_name": "Maria L.",
            "age": 62,
            "gender": "female",
            "stroke_type": "ischemic",
            "initial_severity": "mild",
            "treatment_duration_days": 90,
            "recovery_percentage": 98,
            "treatment_methods": json.dumps([
                "Medication therapy", "Physical therapy", "Lifestyle modifications", 
                "Dietary counseling", "Regular monitoring"
            ]),
            "success_story": "Maria's quick recognition of stroke symptoms and immediate medical attention led to minimal brain damage. Her proactive approach to recovery and lifestyle changes resulted in excellent outcomes.",
            "before_condition": "Mild weakness on left side, slight speech slurring, balance issues",
            "after_condition": "Full strength restored, clear speech, excellent balance, improved overall health",
            "key_factors": json.dumps([
                "Fast medical response", "Immediate treatment", "Lifestyle changes", 
                "Regular exercise", "Medication compliance"
            ]),
            "location": "Miami, FL",
            "recovery_completed_date": date.today() - timedelta(days=60)
        },
        {
            "patient_name": "Robert K.",
            "age": 71,
            "gender": "male",
            "stroke_type": "ischemic",
            "initial_severity": "moderate",
            "treatment_duration_days": 240,
            "recovery_percentage": 88,
            "treatment_methods": json.dumps([
                "Thrombolytic therapy", "Physical therapy", "Occupational therapy", 
                "Cardiac rehabilitation", "Nutritional counseling"
            ]),
            "success_story": "Despite his age, Robert's commitment to rehabilitation and healthy lifestyle changes allowed him to regain independence and enjoy retirement activities with his grandchildren.",
            "before_condition": "Right-side weakness, difficulty walking, problems with coordination and balance",
            "after_condition": "Walking independently, driving again, actively playing with grandchildren",
            "key_factors": json.dumps([
                "Age-appropriate therapy", "Family motivation", "Gradual progression", 
                "Heart-healthy lifestyle", "Social engagement"
            ]),
            "location": "Denver, CO",
            "recovery_completed_date": date.today() - timedelta(days=45)
        },
        {
            "patient_name": "Lisa P.",
            "age": 39,
            "gender": "female",
            "stroke_type": "ischemic",
            "initial_severity": "severe",
            "treatment_duration_days": 300,
            "recovery_percentage": 92,
            "treatment_methods": json.dumps([
                "Emergency surgery", "Intensive rehabilitation", "Speech therapy", 
                "Cognitive therapy", "Art therapy", "Peer counseling"
            ]),
            "success_story": "Lisa's stroke at a young age was devastating, but her youth and determination, combined with innovative therapies, led to remarkable recovery. She now advocates for stroke awareness.",
            "before_condition": "Severe aphasia, right-side paralysis, cognitive difficulties, depression",
            "after_condition": "Fluent speech, walking normally, returned to creative work, became stroke advocate",
            "key_factors": json.dumps([
                "Young age advantage", "Innovative therapies", "Creative rehabilitation", 
                "Peer support", "Advocacy involvement"
            ]),
            "location": "Portland, OR",
            "recovery_completed_date": date.today() - timedelta(days=120)
        },
        {
            "patient_name": "Michael D.",
            "age": 55,
            "gender": "male",
            "stroke_type": "hemorrhagic",
            "initial_severity": "moderate",
            "treatment_duration_days": 210,
            "recovery_percentage": 90,
            "treatment_methods": json.dumps([
                "Minimally invasive surgery", "Physical therapy", "Aquatic therapy", 
                "Technology-assisted rehabilitation", "Mindfulness training"
            ]),
            "success_story": "Michael's recovery was enhanced by cutting-edge rehabilitation technologies and alternative therapies. His engineering background helped him optimize his recovery process.",
            "before_condition": "Memory issues, left-side weakness, difficulty with problem-solving tasks",
            "after_condition": "Sharp memory, full strength, returned to engineering work, improved stress management",
            "key_factors": json.dumps([
                "Technology integration", "Analytical approach", "Alternative therapies", 
                "Stress management", "Professional support"
            ]),
            "location": "San Francisco, CA",
            "recovery_completed_date": date.today() - timedelta(days=75)
        }
    ]
    
    for patient_data in recovered_patients_data:
        patient = RecoveredPatient(**patient_data)
        db.add(patient)
    
    db.commit()
    print(f"Successfully seeded {len(recovered_patients_data)} recovered patients")
    db.close()

def seed_sample_stroke_analyses(user_id: int):
    """Seed sample stroke analysis data for a user."""
    
    db = SessionLocal()
    
    # Check if user already has analyses
    if db.query(StrokeAnalysis).filter(StrokeAnalysis.user_id == user_id).first():
        print(f"Stroke analyses already exist for user {user_id}. Skipping seeding.")
        db.close()
        return
    
    sample_analyses = [
        {
            "user_id": user_id,
            "analysis_type": "stroke_risk_assessment",
            "confidence_score": 92,
            "findings": "Low risk of stroke detected. No significant abnormalities found in brain imaging. Blood flow patterns appear normal with good vessel integrity.",
            "recommendations": "Continue current healthy lifestyle. Regular exercise and balanced diet recommended. Schedule follow-up in 6 months.",
            "severity_level": "low",
            "brain_regions_affected": json.dumps([]),
            "lesion_volume_ml": 0,
            "status": "completed",
            "reviewed_by_doctor": True,
            "doctor_notes": "Excellent results. Patient shows very low stroke risk factors."
        },
        {
            "user_id": user_id,
            "analysis_type": "preventive_screening",
            "confidence_score": 89,
            "findings": "Mild arterial narrowing detected in left carotid artery (15% stenosis). No acute findings. Early atherosclerotic changes consistent with age.",
            "recommendations": "Increase cardiovascular exercise, consider statin therapy, monitor blood pressure more frequently. Dietary modifications recommended.",
            "severity_level": "low",
            "brain_regions_affected": json.dumps(["left_carotid_artery"]),
            "lesion_volume_ml": None,
            "status": "completed",
            "reviewed_by_doctor": True,
            "doctor_notes": "Mild changes that require monitoring but not immediate intervention."
        },
        {
            "user_id": user_id,
            "analysis_type": "follow_up_assessment",
            "confidence_score": 94,
            "findings": "Improvement noted from previous scan. Carotid artery stenosis remains stable at 15%. Good collateral circulation observed.",
            "recommendations": "Continue current medication regimen. Lifestyle modifications showing positive results. Next follow-up in 3 months.",
            "severity_level": "low",
            "brain_regions_affected": json.dumps(["left_carotid_artery"]),
            "lesion_volume_ml": None,
            "status": "completed",
            "reviewed_by_doctor": False,
            "doctor_notes": None
        }
    ]
    
    for i, analysis_data in enumerate(sample_analyses):
        # Space out the analyses over the past few months
        created_date = datetime.now() - timedelta(days=30 * (len(sample_analyses) - i))
        analysis = StrokeAnalysis(**analysis_data)
        analysis.created_at = created_date
        analysis.updated_at = created_date
        db.add(analysis)
    
    db.commit()
    print(f"Successfully seeded {len(sample_analyses)} stroke analyses for user {user_id}")
    db.close()

def initialize_dummy_data():
    """Initialize all dummy data."""
    print("Initializing dummy data...")
    seed_recovered_patients()
    print("Dummy data initialization complete!")

if __name__ == "__main__":
    initialize_dummy_data()
