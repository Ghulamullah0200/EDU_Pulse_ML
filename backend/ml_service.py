import os
import joblib
import json
import pandas as pd
from datetime import datetime
from models import StudentInput, PredictionResponse

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_pipeline")
MODEL_PATH = os.path.join(MODEL_DIR, "student_risk_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.csv")

class MLService:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.feature_importance = None
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        if os.path.exists(IMPORTANCE_PATH):
            self.feature_importance = pd.read_csv(IMPORTANCE_PATH)

    def get_risk_level(self, probability: float) -> str:
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"

    def get_intervention(self, risk_level: str) -> str:
        if risk_level == "Low":
            return "Standard monitoring. Continue current academic plan."
        elif risk_level == "Medium":
            return "Schedule a check-in with the academic counselor. Recommend peer tutoring."
        else:
            return "Immediate intervention required. Mandatory counselor meeting and personalized study support plan."

    def extract_important_factors(self, student_df: pd.DataFrame) -> list:
        # Simplified: highlight areas where student is struggling compared to general population
        factors = []
        if student_df['attendance_percentage'].iloc[0] < 75:
            factors.append({"factor": "Attendance", "message": "Attendance is below 75%"})
        if student_df['previous_gpa'].iloc[0] < 2.5:
            factors.append({"factor": "Previous GPA", "message": "Previous GPA is relatively low"})
        if student_df['absences'].iloc[0] > 7:
            factors.append({"factor": "Absences", "message": "High number of absences"})
        if student_df['assignment_average'].iloc[0] < 60:
            factors.append({"factor": "Assignments", "message": "Assignment average is low"})
        if student_df['midterm_score'].iloc[0] < 60:
            factors.append({"factor": "Midterm", "message": "Midterm score is low"})
            
        if not factors:
            factors.append({"factor": "Overall Profile", "message": "Model identified subtle risk patterns across multiple features"})
            
        return factors

    def predict_single(self, student: StudentInput) -> PredictionResponse:
        if not self.model:
            raise Exception("Model is not loaded")
            
        student_dict = student.model_dump()
        df = pd.DataFrame([student_dict])
        
        # Predict
        probability = float(self.model.predict_proba(df)[0][1])
        prediction_val = int(self.model.predict(df)[0])
        prediction_str = "At Risk" if prediction_val == 1 else "Not At Risk"
        
        risk_level = self.get_risk_level(probability)
        intervention = self.get_intervention(risk_level)
        factors = self.extract_important_factors(df)
        
        version = self.metadata.get("model_version", "1.0.0") if self.metadata else "1.0.0"
        
        return PredictionResponse(
            prediction=prediction_str,
            probability=probability,
            risk_level=risk_level,
            model_version=version,
            prediction_date=datetime.now().isoformat(),
            important_factors=factors,
            recommended_intervention=intervention
        )

ml_service = MLService()
