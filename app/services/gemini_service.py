import google.generativeai as genai
from typing import Dict, List, Optional
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables. AI features will use fallback responses.")
else:
    genai.configure(api_key=GEMINI_API_KEY)


class GeminiHealthAdvisor:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')

    async def generate_health_advice(self, user_profile: Dict, focus_area: str = "general") -> Dict:
        """Generate personalized health advice based on user profile."""

        # Construct prompt based on user profile
        prompt = self._build_health_advice_prompt(user_profile, focus_area)

        try:
            response = self.model.generate_content(prompt)
            advice_data = self._parse_health_advice_response(response.text, focus_area)
            return advice_data
        except Exception:
            # Fallback advice if API fails
            return self._get_fallback_advice(focus_area)

    async def analyze_stroke_risk(self, user_profile: Dict) -> Dict:
        """Analyze stroke risk based on user profile."""

        prompt = self._build_stroke_risk_prompt(user_profile)

        try:
            response = self.model.generate_content(prompt)
            risk_analysis = self._parse_stroke_risk_response(response.text)
            return risk_analysis
        except Exception:
            return self._get_fallback_stroke_risk()

    async def generate_recovery_insights(self, analysis_history: List[Dict]) -> Dict:
        """Generate recovery insights based on analysis history."""

        prompt = self._build_recovery_insights_prompt(analysis_history)

        try:
            response = self.model.generate_content(prompt)
            insights = self._parse_recovery_insights_response(response.text)
            return insights
        except Exception:
            return self._get_fallback_recovery_insights()

    def _build_health_advice_prompt(self, profile: Dict, focus_area: str) -> str:
        """Build prompt for health advice generation."""

        base_info = f"""
        Generate personalized health advice for a {profile.get('age', 'unknown')} year old person with the following profile:

        Physical Stats:
        - Height: {profile.get('height_cm', 'not provided')} cm
        - Weight: {profile.get('weight_kg', 'not provided')} kg
        - BMI: {profile.get('bmi', 'not calculated')}

        Lifestyle:
        - Exercise: {profile.get('exercise_frequency', 'not specified')}
        - Sleep: {profile.get('sleep_hours', 'not specified')} hours per night
        - Smoking: {profile.get('smoking_status', 'not specified')}
        - Alcohol: {profile.get('alcohol_consumption', 'not specified')}
        - Diet: {profile.get('diet_type', 'not specified')}
        - Stress Level: {profile.get('stress_level', 'not specified')}

        Medical Info:
        - Blood Type: {profile.get('blood_type', 'not provided')}
        - Allergies: {profile.get('allergies', 'none specified')}
        - Medical Conditions: {profile.get('medical_conditions', 'none specified')}
        - Current Medications: {profile.get('current_medications', 'none specified')}
        """

        if focus_area == "stroke_prevention":
            focus_instruction = """
            Focus specifically on STROKE PREVENTION advice. Provide:
            1. Specific lifestyle modifications to reduce stroke risk
            2. Dietary recommendations for brain health
            3. Exercise routines that improve cardiovascular health
            4. Warning signs to watch for
            5. When to seek medical attention
            """
        elif focus_area == "recovery":
            focus_instruction = """
            Focus on RECOVERY and REHABILITATION advice. Provide:
            1. Recovery-focused exercise recommendations
            2. Nutrition for healing and brain health
            3. Mental health and cognitive recovery tips
            4. Lifestyle adaptations for recovery
            5. Long-term health maintenance strategies
            """
        else:
            focus_instruction = """
            Focus on GENERAL HEALTH OPTIMIZATION. Provide:
            1. Overall wellness recommendations
            2. Preventive care suggestions
            3. Lifestyle improvements for better health
            4. Risk factor management
            5. Health monitoring recommendations
            """

        prompt = base_info + focus_instruction + """

        Please provide the response in the following JSON format:
        {
            "advice_type": "stroke_prevention/general/recovery",
            "title": "Personalized Health Advice",
            "content": "Main advice content (2-3 paragraphs)",
            "recommendations": ["specific recommendation 1", "specific recommendation 2", "specific recommendation 3"],
            "risk_factors": ["risk factor 1", "risk factor 2", "risk factor 3"],
            "lifestyle_tips": ["tip 1", "tip 2", "tip 3", "tip 4", "tip 5"]
        }

        Make the advice specific to their profile and actionable.
        """

        return prompt

    def _build_stroke_risk_prompt(self, profile: Dict) -> str:
        """Build prompt for stroke risk analysis."""

        return f"""
        Analyze stroke risk for a person with this profile:

        Age: {profile.get('age', 'unknown')}
        BMI: {profile.get('bmi', 'unknown')}
        Exercise: {profile.get('exercise_frequency', 'unknown')}
        Smoking: {profile.get('smoking_status', 'unknown')}
        Alcohol: {profile.get('alcohol_consumption', 'unknown')}
        Stress: {profile.get('stress_level', 'unknown')}
        Medical conditions: {profile.get('medical_conditions', 'none')}

        Provide a stroke risk assessment in JSON format:
        {{
            "risk_level": "low/medium/high",
            "risk_score": 0-100,
            "primary_risk_factors": ["factor1", "factor2"],
            "protective_factors": ["factor1", "factor2"],
            "recommendations": ["rec1", "rec2", "rec3"]
        }}
        """

    def _build_recovery_insights_prompt(self, analysis_history: List[Dict]) -> str:
        """Build prompt for recovery insights."""

        history_summary = "Analysis history:\n"
        for analysis in analysis_history[-5:]:  # Last 5 analyses
            history_summary += f"- {analysis.get('created_at', 'unknown date')}: {analysis.get('findings', 'no findings')}\n"

        return f"""
        Based on this stroke analysis history, provide recovery insights:

        {history_summary}

        Generate insights in JSON format:
        {{
            "trend": "improving/stable/concerning",
            "progress_summary": "Overall progress description",
            "next_steps": ["step1", "step2", "step3"],
            "timeline_estimate": "Estimated recovery timeline",
            "focus_areas": ["area1", "area2", "area3"]
        }}
        """

    def _parse_health_advice_response(self, response_text: str, focus_area: str) -> Dict:
        """Parse and validate Gemini response for health advice."""
        try:
            # Try to extract JSON from the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                advice_data = json.loads(json_str)

                # Ensure all required fields are present
                advice_data.setdefault('advice_type', focus_area)
                advice_data.setdefault('title', 'Personalized Health Advice')
                advice_data.setdefault('content', 'Health advice content')
                advice_data.setdefault('recommendations', [])
                advice_data.setdefault('risk_factors', [])
                advice_data.setdefault('lifestyle_tips', [])

                return advice_data
            else:
                return self._get_fallback_advice(focus_area)

        except json.JSONDecodeError:
            return self._get_fallback_advice(focus_area)

    def _parse_stroke_risk_response(self, response_text: str) -> Dict:
        """Parse stroke risk analysis response."""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self._get_fallback_stroke_risk()

        except json.JSONDecodeError:
            return self._get_fallback_stroke_risk()

    def _parse_recovery_insights_response(self, response_text: str) -> Dict:
        """Parse recovery insights response."""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self._get_fallback_recovery_insights()

        except json.JSONDecodeError:
            return self._get_fallback_recovery_insights()

    def _get_fallback_advice(self, focus_area: str) -> Dict:
        """Provide fallback health advice if API fails."""

        if focus_area == "stroke_prevention":
            return {
                "advice_type": "stroke_prevention",
                "title": "Stroke Prevention Guidelines",
                "content": "Stroke prevention focuses on managing risk factors and maintaining a healthy lifestyle. Regular exercise, a balanced diet, and stress management are key components of stroke prevention.",
                "recommendations": [
                    "Exercise for at least 30 minutes, 5 days a week",
                    "Maintain a diet rich in fruits, vegetables, and whole grains",
                    "Monitor blood pressure regularly",
                    "Limit alcohol consumption and avoid smoking"
                ],
                "risk_factors": [
                    "High blood pressure",
                    "High cholesterol",
                    "Diabetes",
                    "Smoking",
                    "Excessive alcohol consumption"
                ],
                "lifestyle_tips": [
                    "Take regular walks or engage in cardio exercises",
                    "Eat omega-3 rich foods like fish and nuts",
                    "Practice stress-reduction techniques like meditation",
                    "Get 7-9 hours of quality sleep each night",
                    "Stay hydrated throughout the day"
                ]
            }
        else:
            return {
                "advice_type": "general",
                "title": "General Health Recommendations",
                "content": "Maintaining good health requires a balanced approach to diet, exercise, and lifestyle choices. Focus on building sustainable habits that support your overall well-being.",
                "recommendations": [
                    "Maintain a balanced diet with plenty of fruits and vegetables",
                    "Exercise regularly to maintain cardiovascular health",
                    "Get adequate sleep and manage stress levels",
                    "Schedule regular health check-ups"
                ],
                "risk_factors": [
                    "Sedentary lifestyle",
                    "Poor diet",
                    "Chronic stress",
                    "Lack of sleep"
                ],
                "lifestyle_tips": [
                    "Incorporate physical activity into daily routine",
                    "Choose whole foods over processed options",
                    "Practice mindfulness or relaxation techniques",
                    "Maintain social connections",
                    "Stay consistent with healthy habits"
                ]
            }

    def _get_fallback_stroke_risk(self) -> Dict:
        """Provide fallback stroke risk assessment."""
        return {
            "risk_level": "medium",
            "risk_score": 40,
            "primary_risk_factors": ["Age", "Lifestyle factors"],
            "protective_factors": ["Regular check-ups", "Health awareness"],
            "recommendations": [
                "Consult with healthcare provider for personalized assessment",
                "Maintain regular exercise routine",
                "Monitor blood pressure and cholesterol"
            ]
        }

    def _get_fallback_recovery_insights(self) -> Dict:
        """Provide fallback recovery insights."""
        return {
            "trend": "stable",
            "progress_summary": "Recovery progress appears stable. Continue with current treatment plan.",
            "next_steps": [
                "Continue regular therapy sessions",
                "Maintain medication regimen",
                "Follow up with healthcare provider"
            ],
            "timeline_estimate": "Recovery timelines vary by individual. Consult your healthcare team for personalized estimates.",
            "focus_areas": ["Physical therapy", "Medication adherence", "Lifestyle modifications"]
        }

    async def generate_detailed_recovery_report(self, patient_data: Dict, latest_stroke_analysis: Optional[Dict] = None) -> Dict:
        """Generate a detailed recovery report for a specific patient."""

        prompt = self._build_recovery_report_prompt(patient_data, latest_stroke_analysis)

        try:
            response = self.model.generate_content(prompt)
            detailed_report = self._parse_recovery_report_response(response.text, patient_data)
            return detailed_report
        except Exception:
            return self._get_fallback_recovery_report(patient_data)

    def _build_recovery_report_prompt(self, patient_data: Dict, latest_analysis: Optional[Dict]) -> str:
        """Build prompt for generating detailed recovery report."""

        analysis_context = ""
        if latest_analysis:
            analysis_context = f"""

RECENT STROKE ANALYSIS CONTEXT:
- Analysis Date: {latest_analysis.get('created_at', 'Recent')}
- Severity: {latest_analysis.get('severity_level', 'Not specified')}
- Confidence: {latest_analysis.get('confidence_score', 'N/A')}%
- Affected Regions: {latest_analysis.get('brain_regions_affected', 'Various')}
- Current Treatment Focus: Based on this analysis type
"""

        return f"""
You are a medical writing specialist creating an inspiring and detailed recovery report for stroke patients and their families. Based on the patient profile below, generate a comprehensive, medically accurate, and emotionally inspiring recovery story.

PATIENT PROFILE:
- Name: {patient_data.get('patient_name', 'Patient')}
- Age: {patient_data.get('age', 'Adult')}
- Gender: {patient_data.get('gender', 'Not specified')}
- Stroke Type: {patient_data.get('stroke_type', 'Not specified')}
- Initial Severity: {patient_data.get('initial_severity', 'Not specified')}
- Treatment Duration: {patient_data.get('treatment_duration_days', 'Several months')} days
- Recovery Percentage: {patient_data.get('recovery_percentage', 80)}%
- Location: {patient_data.get('location', 'Global')}
- Current Condition: {patient_data.get('after_condition', 'Improving')}
{analysis_context}

Generate a detailed recovery report in JSON format with these exact fields:

{{
    "detailed_timeline": [
        {{"phase": "Initial Emergency", "duration": "Days 1-3", "description": "Emergency response and stabilization", "key_events": ["event1", "event2"]}},
        {{"phase": "Acute Treatment", "duration": "Days 4-14", "description": "Intensive medical treatment", "key_events": ["event1", "event2"]}},
        {{"phase": "Early Rehabilitation", "duration": "Weeks 3-8", "description": "Beginning recovery activities", "key_events": ["event1", "event2"]}},
        {{"phase": "Active Recovery", "duration": "Months 2-6", "description": "Intensive rehabilitation", "key_events": ["event1", "event2"]}},
        {{"phase": "Ongoing Support", "duration": "6+ months", "description": "Continued improvement", "key_events": ["event1", "event2"]}}
    ],
    "treatment_details": {{
        "medical_interventions": ["intervention1", "intervention2", "intervention3"],
        "therapies": ["Physical therapy", "Speech therapy", "Occupational therapy"],
        "medications": ["medication_type1", "medication_type2"],
        "innovative_treatments": ["treatment1", "treatment2"]
    }},
    "daily_life_impact": {{
        "before_stroke": "Description of life before stroke",
        "during_recovery": "Challenges and adaptations during recovery",
        "current_status": "Current daily life and capabilities"
    }},
    "support_system": {{
        "family_role": "How family supported recovery",
        "medical_team": "Key medical professionals involved",
        "community_support": "Community and peer support received"
    }},
    "inspiring_moments": [
        {{"milestone": "First major achievement", "description": "Detailed description", "emotional_impact": "How it felt"}},
        {{"milestone": "Breakthrough moment", "description": "Detailed description", "emotional_impact": "How it felt"}},
        {{"milestone": "Return to independence", "description": "Detailed description", "emotional_impact": "How it felt"}}
    ],
    "lessons_learned": [
        "Important lesson about recovery",
        "Advice for other stroke survivors",
        "Key insight about resilience"
    ],
    "current_goals": [
        "Short-term goal",
        "Long-term aspiration",
        "Personal dream"
    ],
    "message_to_others": "Inspiring message to other stroke patients and families (2-3 sentences)",
    "recovery_tips": [
        {{"category": "Physical", "tip": "Practical physical recovery tip"}},
        {{"category": "Mental", "tip": "Mental health and mindset tip"}},
        {{"category": "Social", "tip": "Social and family relationship tip"}},
        {{"category": "Practical", "tip": "Daily life management tip"}}
    ]
}}

Make this medically accurate, emotionally authentic, and inspiring. Focus on real challenges and genuine progress. Include specific details that make the story feel real and relatable.
"""

    def _parse_recovery_report_response(self, response_text: str, patient_data: Dict) -> Dict:
        """Parse the Gemini response for recovery report."""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._get_fallback_recovery_report(patient_data)
        except Exception:
            return self._get_fallback_recovery_report(patient_data)

    def _get_fallback_recovery_report(self, patient_data: Dict) -> Dict:
        """Provide fallback recovery report if AI generation fails."""
        name = patient_data.get('patient_name', 'Patient')
        return {
            "detailed_timeline": [
                {"phase": "Initial Emergency", "duration": "Days 1-3", "description": "Emergency response and stabilization", "key_events": ["Immediate medical attention", "Initial assessment", "Family notification"]},
                {"phase": "Acute Treatment", "duration": "Days 4-14", "description": "Intensive medical treatment", "key_events": ["Medication regimen started", "Vital signs stabilized", "Treatment plan developed"]},
                {"phase": "Early Rehabilitation", "duration": "Weeks 3-8", "description": "Beginning recovery activities", "key_events": ["First therapy session", "Basic movement recovery", "Communication improvements"]},
                {"phase": "Active Recovery", "duration": "Months 2-6", "description": "Intensive rehabilitation", "key_events": ["Regular therapy sessions", "Significant improvements", "Independence milestones"]},
                {"phase": "Ongoing Support", "duration": "6+ months", "description": "Continued improvement", "key_events": ["Return to daily activities", "Ongoing monitoring", "Quality of life improvements"]}
            ],
            "treatment_details": {
                "medical_interventions": ["Acute stroke medications", "Blood pressure management", "Anticoagulation therapy"],
                "therapies": ["Physical therapy", "Speech therapy", "Occupational therapy"],
                "medications": ["Stroke prevention medications", "Rehabilitation support medications"],
                "innovative_treatments": ["Modern rehabilitation techniques", "Technology-assisted therapy"]
            },
            "daily_life_impact": {
                "before_stroke": f"{name} lived an active, independent life with normal daily routines and responsibilities.",
                "during_recovery": "Recovery involved learning to adapt daily activities, with gradual improvements in mobility and communication.",
                "current_status": f"{name} has successfully adapted to life post-stroke and maintains an active, fulfilling lifestyle."
            },
            "support_system": {
                "family_role": "Family provided constant emotional support and practical assistance throughout recovery.",
                "medical_team": "Dedicated team of doctors, nurses, and therapists provided comprehensive care.",
                "community_support": "Local support groups and community resources aided in the recovery journey."
            },
            "inspiring_moments": [
                {"milestone": "First words after stroke", "description": "Speaking the first clear words was an emotional breakthrough for both patient and family.", "emotional_impact": "Overwhelming joy and hope for the future."},
                {"milestone": "Walking independently", "description": "Taking the first independent steps marked a major physical recovery milestone.", "emotional_impact": "Sense of freedom and returning confidence."},
                {"milestone": "Returning home", "description": "Moving back home represented the achievement of significant independence and recovery goals.", "emotional_impact": "Deep satisfaction and gratitude for the recovery journey."}
            ],
            "lessons_learned": [
                "Recovery takes time and patience is essential",
                "Small improvements add up to significant progress",
                "Support from others makes an enormous difference"
            ],
            "current_goals": [
                "Continue improving physical strength",
                "Maintain social connections and activities",
                "Help others in their recovery journey"
            ],
            "message_to_others": f"Recovery is possible with determination and support. Every small step forward is a victory worth celebrating.",
            "recovery_tips": [
                {"category": "Physical", "tip": "Consistent daily exercises, even small ones, contribute to significant improvements over time."},
                {"category": "Mental", "tip": "Maintain a positive outlook and celebrate small victories throughout the recovery process."},
                {"category": "Social", "tip": "Stay connected with family and friends - their support is invaluable for emotional recovery."},
                {"category": "Practical", "tip": "Adapt your environment to support independence while staying safe in daily activities."}
            ]
        }


# Global instance
gemini_advisor = GeminiHealthAdvisor()
import google.generativeai as genai
from typing import Dict, List, Optional
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables. AI features will use fallback responses.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

class GeminiHealthAdvisor:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def generate_health_advice(self, user_profile: Dict, focus_area: str = "general") -> Dict:
        """Generate personalized health advice based on user profile."""
        
        # Construct prompt based on user profile
        prompt = self._build_health_advice_prompt(user_profile, focus_area)
        
        try:
            response = self.model.generate_content(prompt)
            advice_data = self._parse_health_advice_response(response.text, focus_area)
            return advice_data
        except Exception as e:
            # Fallback advice if API fails
            return self._get_fallback_advice(focus_area)
    
    async def analyze_stroke_risk(self, user_profile: Dict) -> Dict:
        """Analyze stroke risk based on user profile."""
        
        prompt = self._build_stroke_risk_prompt(user_profile)
        
        try:
            response = self.model.generate_content(prompt)
            risk_analysis = self._parse_stroke_risk_response(response.text)
            return risk_analysis
        except Exception as e:
            return self._get_fallback_stroke_risk()
    
    async def generate_recovery_insights(self, analysis_history: List[Dict]) -> Dict:
        """Generate recovery insights based on analysis history."""
        
        prompt = self._build_recovery_insights_prompt(analysis_history)
        
        try:
            response = self.model.generate_content(prompt)
            insights = self._parse_recovery_insights_response(response.text)
            return insights
        except Exception as e:
            return self._get_fallback_recovery_insights()
    
    def _build_health_advice_prompt(self, profile: Dict, focus_area: str) -> str:
        """Build prompt for health advice generation."""
        
        base_info = f"""
        Generate personalized health advice for a {profile.get('age', 'unknown')} year old person with the following profile:
        
        Physical Stats:
        - Height: {profile.get('height_cm', 'not provided')} cm
        - Weight: {profile.get('weight_kg', 'not provided')} kg
        - BMI: {profile.get('bmi', 'not calculated')}
        
        Lifestyle:
        - Exercise: {profile.get('exercise_frequency', 'not specified')}
        - Sleep: {profile.get('sleep_hours', 'not specified')} hours per night
        - Smoking: {profile.get('smoking_status', 'not specified')}
        - Alcohol: {profile.get('alcohol_consumption', 'not specified')}
        - Diet: {profile.get('diet_type', 'not specified')}
        - Stress Level: {profile.get('stress_level', 'not specified')}
        
        Medical Info:
        - Blood Type: {profile.get('blood_type', 'not provided')}
        - Allergies: {profile.get('allergies', 'none specified')}
        - Medical Conditions: {profile.get('medical_conditions', 'none specified')}
        - Current Medications: {profile.get('current_medications', 'none specified')}
        """
        
        if focus_area == "stroke_prevention":
            focus_instruction = """
            Focus specifically on STROKE PREVENTION advice. Provide:
            1. Specific lifestyle modifications to reduce stroke risk
            2. Dietary recommendations for brain health
            3. Exercise routines that improve cardiovascular health
            4. Warning signs to watch for
            5. When to seek medical attention
            """
        elif focus_area == "recovery":
            focus_instruction = """
            Focus on RECOVERY and REHABILITATION advice. Provide:
            1. Recovery-focused exercise recommendations
            2. Nutrition for healing and brain health
            3. Mental health and cognitive recovery tips
            4. Lifestyle adaptations for recovery
            5. Long-term health maintenance strategies
            """
        else:
            focus_instruction = """
            Focus on GENERAL HEALTH OPTIMIZATION. Provide:
            1. Overall wellness recommendations
            2. Preventive care suggestions
            3. Lifestyle improvements for better health
            4. Risk factor management
            5. Health monitoring recommendations
            """
        
        prompt = base_info + focus_instruction + """
        
        Please provide the response in the following JSON format:
        {
            "advice_type": "stroke_prevention/general/recovery",
            "title": "Personalized Health Advice",
            "content": "Main advice content (2-3 paragraphs)",
            "recommendations": ["specific recommendation 1", "specific recommendation 2", "specific recommendation 3"],
            "risk_factors": ["risk factor 1", "risk factor 2", "risk factor 3"],
            "lifestyle_tips": ["tip 1", "tip 2", "tip 3", "tip 4", "tip 5"]
        }
        
        Make the advice specific to their profile and actionable.
        """
        
        return prompt
    
    def _build_stroke_risk_prompt(self, profile: Dict) -> str:
        """Build prompt for stroke risk analysis."""
        
        return f"""
        Analyze stroke risk for a person with this profile:
        
        Age: {profile.get('age', 'unknown')}
        BMI: {profile.get('bmi', 'unknown')}
        Exercise: {profile.get('exercise_frequency', 'unknown')}
        Smoking: {profile.get('smoking_status', 'unknown')}
        Alcohol: {profile.get('alcohol_consumption', 'unknown')}
        Stress: {profile.get('stress_level', 'unknown')}
        Medical conditions: {profile.get('medical_conditions', 'none')}
        
        Provide a stroke risk assessment in JSON format:
        {{
            "risk_level": "low/medium/high",
            "risk_score": 0-100,
            "primary_risk_factors": ["factor1", "factor2"],
            "protective_factors": ["factor1", "factor2"],
            "recommendations": ["rec1", "rec2", "rec3"]
        }}
        """
    
    def _build_recovery_insights_prompt(self, analysis_history: List[Dict]) -> str:
        """Build prompt for recovery insights."""
        
        history_summary = "Analysis history:\n"
        for analysis in analysis_history[-5:]:  # Last 5 analyses
            history_summary += f"- {analysis.get('created_at', 'unknown date')}: {analysis.get('findings', 'no findings')}\n"
        
        return f"""
        Based on this stroke analysis history, provide recovery insights:
        
        {history_summary}
        
        Generate insights in JSON format:
        {{
            "trend": "improving/stable/concerning",
            "progress_summary": "Overall progress description",
            "next_steps": ["step1", "step2", "step3"],
            "timeline_estimate": "Estimated recovery timeline",
            "focus_areas": ["area1", "area2", "area3"]
        }}
        """
    
    def _parse_health_advice_response(self, response_text: str, focus_area: str) -> Dict:
        """Parse and validate Gemini response for health advice."""
        try:
            # Try to extract JSON from the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                advice_data = json.loads(json_str)
                
                # Ensure all required fields are present
                advice_data.setdefault('advice_type', focus_area)
                advice_data.setdefault('title', 'Personalized Health Advice')
                advice_data.setdefault('content', 'Health advice content')
                advice_data.setdefault('recommendations', [])
                advice_data.setdefault('risk_factors', [])
                advice_data.setdefault('lifestyle_tips', [])
                
                return advice_data
            else:
                return self._get_fallback_advice(focus_area)
                
        except json.JSONDecodeError:
            return self._get_fallback_advice(focus_area)
    
    def _parse_stroke_risk_response(self, response_text: str) -> Dict:
        """Parse stroke risk analysis response."""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self._get_fallback_stroke_risk()
                
        except json.JSONDecodeError:
            return self._get_fallback_stroke_risk()
    
    def _parse_recovery_insights_response(self, response_text: str) -> Dict:
        """Parse recovery insights response."""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self._get_fallback_recovery_insights()
                
        except json.JSONDecodeError:
            return self._get_fallback_recovery_insights()
    
    def _get_fallback_advice(self, focus_area: str) -> Dict:
        """Provide fallback health advice if API fails."""
        
        if focus_area == "stroke_prevention":
            return {
                "advice_type": "stroke_prevention",
                "title": "Stroke Prevention Guidelines",
                "content": "Stroke prevention focuses on managing risk factors and maintaining a healthy lifestyle. Regular exercise, a balanced diet, and stress management are key components of stroke prevention.",
                "recommendations": [
                    "Exercise for at least 30 minutes, 5 days a week",
                    "Maintain a diet rich in fruits, vegetables, and whole grains",
                    "Monitor blood pressure regularly",
                    "Limit alcohol consumption and avoid smoking"
                ],
                "risk_factors": [
                    "High blood pressure",
                    "High cholesterol",
                    "Diabetes",
                    "Smoking",
                    "Excessive alcohol consumption"
                ],
                "lifestyle_tips": [
                    "Take regular walks or engage in cardio exercises",
                    "Eat omega-3 rich foods like fish and nuts",
                    "Practice stress-reduction techniques like meditation",
                    "Get 7-9 hours of quality sleep each night",
                    "Stay hydrated throughout the day"
                ]
            }
        else:
            return {
                "advice_type": "general",
                "title": "General Health Recommendations",
                "content": "Maintaining good health requires a balanced approach to diet, exercise, and lifestyle choices. Focus on building sustainable habits that support your overall well-being.",
                "recommendations": [
                    "Maintain a balanced diet with plenty of fruits and vegetables",
                    "Exercise regularly to maintain cardiovascular health",
                    "Get adequate sleep and manage stress levels",
                    "Schedule regular health check-ups"
                ],
                "risk_factors": [
                    "Sedentary lifestyle",
                    "Poor diet",
                    "Chronic stress",
                    "Lack of sleep"
                ],
                "lifestyle_tips": [
                    "Incorporate physical activity into daily routine",
                    "Choose whole foods over processed options",
                    "Practice mindfulness or relaxation techniques",
                    "Maintain social connections",
                    "Stay consistent with healthy habits"
                ]
            }
    
    def _get_fallback_stroke_risk(self) -> Dict:
        """Provide fallback stroke risk assessment."""
        return {
            "risk_level": "medium",
            "risk_score": 40,
            "primary_risk_factors": ["Age", "Lifestyle factors"],
            "protective_factors": ["Regular check-ups", "Health awareness"],
            "recommendations": [
                "Consult with healthcare provider for personalized assessment",
                "Maintain regular exercise routine",
                "Monitor blood pressure and cholesterol"
            ]
        }
    
    def _get_fallback_recovery_insights(self) -> Dict:
        """Provide fallback recovery insights."""
        return {
            "trend": "stable",
            "progress_summary": "Recovery progress appears stable. Continue with current treatment plan.",
            "next_steps": [
                "Continue regular therapy sessions",
                "Maintain medication regimen",
                "Follow up with healthcare provider"
            ],
            "timeline_estimate": "Recovery timelines vary by individual. Consult your healthcare team for personalized estimates.",
            "focus_areas": ["Physical therapy", "Medication adherence", "Lifestyle modifications"]
        }

    async def generate_detailed_recovery_report(self, patient_data: Dict, latest_stroke_analysis: Optional[Dict] = None) -> Dict:
        """Generate a detailed recovery report for a specific patient."""
        
        prompt = self._build_recovery_report_prompt(patient_data, latest_stroke_analysis)
        
        try:
            response = self.model.generate_content(prompt)
            detailed_report = self._parse_recovery_report_response(response.text, patient_data)
            return detailed_report
        except Exception as e:
            return self._get_fallback_recovery_report(patient_data)
    
    def _build_recovery_report_prompt(self, patient_data: Dict, latest_analysis: Optional[Dict]) -> str:
        """Build prompt for generating detailed recovery report."""
        
        analysis_context = ""
        if latest_analysis:
            analysis_context = f"""
            
RECENT STROKE ANALYSIS CONTEXT:
- Analysis Date: {latest_analysis.get('created_at', 'Recent')}
- Severity: {latest_analysis.get('severity_level', 'Not specified')}
- Confidence: {latest_analysis.get('confidence_score', 'N/A')}%
- Affected Regions: {latest_analysis.get('brain_regions_affected', 'Various')}
- Current Treatment Focus: Based on this analysis type
"""
        
        return f"""
You are a medical writing specialist creating an inspiring and detailed recovery report for stroke patients and their families. Based on the patient profile below, generate a comprehensive, medically accurate, and emotionally inspiring recovery story.

PATIENT PROFILE:
- Name: {patient_data.get('patient_name', 'Patient')}
- Age: {patient_data.get('age', 'Adult')}
- Gender: {patient_data.get('gender', 'Not specified')}
- Stroke Type: {patient_data.get('stroke_type', 'Not specified')}
- Initial Severity: {patient_data.get('initial_severity', 'Not specified')}
- Treatment Duration: {patient_data.get('treatment_duration_days', 'Several months')} days
- Recovery Percentage: {patient_data.get('recovery_percentage', 80)}%
- Location: {patient_data.get('location', 'Global')}
- Current Condition: {patient_data.get('after_condition', 'Improving')}
{analysis_context}

Generate a detailed recovery report in JSON format with these exact fields:

{{
    "detailed_timeline": [
        {{"phase": "Initial Emergency", "duration": "Days 1-3", "description": "Emergency response and stabilization", "key_events": ["event1", "event2"]}},
        {{"phase": "Acute Treatment", "duration": "Days 4-14", "description": "Intensive medical treatment", "key_events": ["event1", "event2"]}},
        {{"phase": "Early Rehabilitation", "duration": "Weeks 3-8", "description": "Beginning recovery activities", "key_events": ["event1", "event2"]}},
        {{"phase": "Active Recovery", "duration": "Months 2-6", "description": "Intensive rehabilitation", "key_events": ["event1", "event2"]}},
        {{"phase": "Ongoing Support", "duration": "6+ months", "description": "Continued improvement", "key_events": ["event1", "event2"]}}
    ],
    "treatment_details": {{
        "medical_interventions": ["intervention1", "intervention2", "intervention3"],
        "therapies": ["Physical therapy", "Speech therapy", "Occupational therapy"],
        "medications": ["medication_type1", "medication_type2"],
        "innovative_treatments": ["treatment1", "treatment2"]
    }},
    "daily_life_impact": {{
        "before_stroke": "Description of life before stroke",
        "during_recovery": "Challenges and adaptations during recovery",
        "current_status": "Current daily life and capabilities"
    }},
    "support_system": {{
        "family_role": "How family supported recovery",
        "medical_team": "Key medical professionals involved",
        "community_support": "Community and peer support received"
    }},
    "inspiring_moments": [
        {{"milestone": "First major achievement", "description": "Detailed description", "emotional_impact": "How it felt"}},
        {{"milestone": "Breakthrough moment", "description": "Detailed description", "emotional_impact": "How it felt"}},
        {{"milestone": "Return to independence", "description": "Detailed description", "emotional_impact": "How it felt"}}
    ],
    "lessons_learned": [
        "Important lesson about recovery",
        "Advice for other stroke survivors",
        "Key insight about resilience"
    ],
    "current_goals": [
        "Short-term goal",
        "Long-term aspiration",
        "Personal dream"
    ],
    "message_to_others": "Inspiring message to other stroke patients and families (2-3 sentences)",
    "recovery_tips": [
        {{"category": "Physical", "tip": "Practical physical recovery tip"}},
        {{"category": "Mental", "tip": "Mental health and mindset tip"}},
        {{"category": "Social", "tip": "Social and family relationship tip"}},
        {{"category": "Practical", "tip": "Daily life management tip"}}
    ]
}}

Make this medically accurate, emotionally authentic, and inspiring. Focus on real challenges and genuine progress. Include specific details that make the story feel real and relatable.
"""

    def _parse_recovery_report_response(self, response_text: str, patient_data: Dict) -> Dict:
        """Parse the Gemini response for recovery report."""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._get_fallback_recovery_report(patient_data)
        except:
            return self._get_fallback_recovery_report(patient_data)
    
    def _get_fallback_recovery_report(self, patient_data: Dict) -> Dict:
        """Provide fallback recovery report if AI generation fails."""
        name = patient_data.get('patient_name', 'Patient')
        return {
            "detailed_timeline": [
                {"phase": "Initial Emergency", "duration": "Days 1-3", "description": "Emergency response and stabilization", "key_events": ["Immediate medical attention", "Initial assessment", "Family notification"]},
                {"phase": "Acute Treatment", "duration": "Days 4-14", "description": "Intensive medical treatment", "key_events": ["Medication regimen started", "Vital signs stabilized", "Treatment plan developed"]},
                {"phase": "Early Rehabilitation", "duration": "Weeks 3-8", "description": "Beginning recovery activities", "key_events": ["First therapy session", "Basic movement recovery", "Communication improvements"]},
                {"phase": "Active Recovery", "duration": "Months 2-6", "description": "Intensive rehabilitation", "key_events": ["Regular therapy sessions", "Significant improvements", "Independence milestones"]},
                {"phase": "Ongoing Support", "duration": "6+ months", "description": "Continued improvement", "key_events": ["Return to daily activities", "Ongoing monitoring", "Quality of life improvements"]}
            ],
            "treatment_details": {
                "medical_interventions": ["Acute stroke medications", "Blood pressure management", "Anticoagulation therapy"],
                "therapies": ["Physical therapy", "Speech therapy", "Occupational therapy"],
                "medications": ["Stroke prevention medications", "Rehabilitation support medications"],
                "innovative_treatments": ["Modern rehabilitation techniques", "Technology-assisted therapy"]
            },
            "daily_life_impact": {
                "before_stroke": f"{name} lived an active, independent life with normal daily routines and responsibilities.",
                "during_recovery": "Recovery involved learning to adapt daily activities, with gradual improvements in mobility and communication.",
                "current_status": f"{name} has successfully adapted to life post-stroke and maintains an active, fulfilling lifestyle."
            },
            "support_system": {
                "family_role": "Family provided constant emotional support and practical assistance throughout recovery.",
                "medical_team": "Dedicated team of doctors, nurses, and therapists provided comprehensive care.",
                "community_support": "Local support groups and community resources aided in the recovery journey."
            },
            "inspiring_moments": [
                {"milestone": "First words after stroke", "description": "Speaking the first clear words was an emotional breakthrough for both patient and family.", "emotional_impact": "Overwhelming joy and hope for the future."},
                {"milestone": "Walking independently", "description": "Taking the first independent steps marked a major physical recovery milestone.", "emotional_impact": "Sense of freedom and returning confidence."},
                {"milestone": "Returning home", "description": "Moving back home represented the achievement of significant independence and recovery goals.", "emotional_impact": "Deep satisfaction and gratitude for the recovery journey."}
            ],
            "lessons_learned": [
                "Recovery takes time and patience is essential",
                "Small improvements add up to significant progress",
                "Support from others makes an enormous difference"
            ],
            "current_goals": [
                "Continue improving physical strength",
                "Maintain social connections and activities",
                "Help others in their recovery journey"
            ],
            "message_to_others": f"Recovery is possible with determination and support. Every small step forward is a victory worth celebrating.",
            "recovery_tips": [
                {"category": "Physical", "tip": "Consistent daily exercises, even small ones, contribute to significant improvements over time."},
                {"category": "Mental", "tip": "Maintain a positive outlook and celebrate small victories throughout the recovery process."},
                {"category": "Social", "tip": "Stay connected with family and friends - their support is invaluable for emotional recovery."},
                {"category": "Practical", "tip": "Adapt your environment to support independence while staying safe in daily activities."}
            ]
        }

# Global instance
gemini_advisor = GeminiHealthAdvisor()
