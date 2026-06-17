from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class StudentInput(BaseModel):
    age: int = Field(..., ge=15, le=40)
    gender: str = Field(..., pattern="^(Male|Female)$")
    department: str = Field(..., pattern="^(Engineering|Social Sciences|Business|Computer Science)$")
    semester: int = Field(..., ge=1, le=10)
    study_hours_per_week: float = Field(..., ge=0.0, le=80.0)
    attendance_percentage: float = Field(..., ge=0.0, le=100.0)
    assignment_average: float = Field(..., ge=0.0, le=100.0)
    midterm_score: float = Field(..., ge=0.0, le=100.0)
    previous_gpa: float = Field(..., ge=0.0, le=4.0)
    internet_access: str = Field(..., pattern="^(Yes|No)$")
    extra_academic_support: str = Field(..., pattern="^(Yes|No)$")
    part_time_job: str = Field(..., pattern="^(Yes|No)$")
    extracurricular_hours_per_week: float = Field(..., ge=0.0, le=40.0)
    absences: int = Field(..., ge=0, le=100)

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str
    model_version: str
    prediction_date: str
    important_factors: List[dict]
    recommended_intervention: str

class BatchPredictionRequest(BaseModel):
    students: List[StudentInput]

class StudentRecord(StudentInput):
    student_id: str
    created_at: Optional[datetime] = None
