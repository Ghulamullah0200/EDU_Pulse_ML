from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from pydantic import BaseModel, Field
import json
import os

from models import StudentInput, PredictionResponse, BatchPredictionRequest
from ml_service import ml_service
from database import db_service

app = FastAPI(
    title="EduPulse AI API",
    description="Backend API for the Student Academic Risk and Performance Intelligence System",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to check API key or token (simplified for MVP)
async def verify_token(authorization: Optional[str] = Header(None)):
    pass


@app.get("/")
def read_root():
    return {"message": "Welcome to EduPulse AI API"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": ml_service.model is not None,
        "database": "SQLite",
        "database_connected": db_service.is_connected()
    }


# ── Predictions ───────────────────────────────────────────────────
@app.post("/predict/single", response_model=PredictionResponse)
def predict_single(student: StudentInput, authorization: Optional[str] = Header(None)):
    try:
        response = ml_service.predict_single(student)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(request: BatchPredictionRequest):
    try:
        responses = [ml_service.predict_single(s) for s in request.students]
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Metrics ───────────────────────────────────────────────────────
@app.get("/metrics")
def get_metrics():
    if not ml_service.metadata:
        raise HTTPException(status_code=404, detail="Metrics not available")
    return ml_service.metadata


@app.get("/feature-importance")
def get_feature_importance():
    if ml_service.feature_importance is None:
        raise HTTPException(status_code=404, detail="Feature importance not available")
    return ml_service.feature_importance.to_dict(orient="records")


@app.get("/dashboard/summary")
def get_dashboard_summary():
    total, _ = db_service.list_students(limit=1)
    return {
        "total_students": total,
        "at_risk_students": 1920,
        "high_risk_students": 450,
        "average_attendance": 82.4,
        "average_final_score": 70.1,
        "current_model_f1": ml_service.metadata.get("metrics", [{}])[3].get("f1_score", 0.91) if ml_service.metadata else 0.91
    }


# ── Dataset Explorer (paginated from SQLite) ──────────────────────
@app.get("/dataset")
def get_dataset(skip: int = 0, limit: int = 100):
    try:
        total, records = db_service.get_dataset_page(skip, limit)
        return {"total": total, "skip": skip, "limit": limit, "data": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Student Management (all SQLite) ──────────────────────────────
@app.get("/students")
def list_students(limit: int = 200, offset: int = 0):
    try:
        total, records = db_service.list_students(limit, offset)
        return {"total": total, "data": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/students/{student_id}")
def get_student(student_id: str):
    record = db_service.get_student(student_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
    return record


class EnrollStudentRequest(BaseModel):
    name: str = Field(..., min_length=2)
    age: int = Field(..., ge=15, le=40)
    gender: str
    department: str
    semester: int = Field(..., ge=1, le=10)
    study_hours_per_week: float = Field(default=0.0, ge=0.0)
    attendance_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    assignment_average: float = Field(default=0.0, ge=0.0, le=100.0)
    midterm_score: float = Field(default=0.0, ge=0.0, le=100.0)
    previous_gpa: float = Field(default=0.0, ge=0.0, le=4.0)
    internet_access: str = Field(default="Yes")
    extra_academic_support: str = Field(default="No")
    part_time_job: str = Field(default="No")
    extracurricular_hours_per_week: float = Field(default=0.0, ge=0.0)
    absences: int = Field(default=0, ge=0)


@app.post("/students/enroll")
def enroll_student(req: EnrollStudentRequest):
    try:
        data = req.model_dump()
        student = db_service.create_student(data)
        return {"message": "Student enrolled successfully", "student": student}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "uvicorn", "main:app",
                    "--reload", "--host", "0.0.0.0", "--port", "8000"])
