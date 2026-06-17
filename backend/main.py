from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional, List
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import json
import os

from models import StudentInput, PredictionResponse, BatchPredictionRequest
from ml_service import ml_service
from database import db_service

# Path to the dataset CSV (relative to project root)
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_pipeline", "student_performance_expanded.csv")

app = FastAPI(
    title="EduPulse AI API",
    description="Backend API for the Student Academic Risk and Performance Intelligence System",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to check API key or token (simplified for MVP)
async def verify_token(authorization: Optional[str] = Header(None)):
    # In production, this would verify the Supabase JWT token
    pass

@app.get("/")
def read_root():
    return {"message": "Welcome to EduPulse AI API"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": ml_service.model is not None,
        "database_connected": db_service.is_connected()
    }

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
        responses = [ml_service.predict_single(student) for student in request.students]
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    return {
        "total_students": 12000,
        "at_risk_students": 1920,
        "high_risk_students": 450,
        "average_attendance": 82.4,
        "average_final_score": 70.1,
        "current_model_f1": ml_service.metadata.get("metrics", [{}])[3].get("f1_score", 0.91) if ml_service.metadata else 0.91
    }

# ── Dataset Explorer ──────────────────────────────────────────────
@app.get("/dataset")
def get_dataset(skip: int = 0, limit: int = 100):
    try:
        df = pd.read_csv(DATASET_PATH)
        total = len(df)
        subset = df.iloc[skip : skip + limit]
        # Replace NaN with None so JSON serialization doesn't break
        subset = subset.where(pd.notna(subset), None)
        records = subset.to_dict(orient="records")
        return {"total": total, "skip": skip, "limit": limit, "data": records}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset file not found at {DATASET_PATH}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Student Management ────────────────────────────────────────────
@app.get("/students")
def list_students():
    """Return student records from the dataset CSV (first 200 for directory)."""
    try:
        df = pd.read_csv(DATASET_PATH)
        df = df.where(pd.notna(df), None)
        records = df.head(200).to_dict(orient="records")
        return {"total": len(df), "data": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/students/{student_id}")
def get_student(student_id: str):
    """Get a single student's full profile by ID."""
    try:
        df = pd.read_csv(DATASET_PATH)
        df = df.where(pd.notna(df), None)
        row = df[df["student_id"] == student_id]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
        return row.iloc[0].to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    """Add a new student row to the dataset CSV and return the record."""
    try:
        df = pd.read_csv(DATASET_PATH)

        # Generate next student ID
        max_num = df["student_id"].str.extract(r"(\d+)").astype(int).max().iloc[0]
        new_id = f"STU{max_num + 1}"

        new_row = {
            "student_id": new_id,
            "age": req.age,
            "gender": req.gender,
            "department": req.department,
            "semester": req.semester,
            "study_hours_per_week": req.study_hours_per_week,
            "attendance_percentage": req.attendance_percentage,
            "assignment_average": req.assignment_average,
            "midterm_score": req.midterm_score,
            "previous_gpa": req.previous_gpa,
            "internet_access": req.internet_access,
            "extra_academic_support": req.extra_academic_support,
            "part_time_job": req.part_time_job,
            "extracurricular_hours_per_week": req.extracurricular_hours_per_week,
            "absences": req.absences,
            "final_score": 0.0,
            "at_risk": 0,
            "risk_label": "Pending",
            "data_origin": "Enrolled"
        }

        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(DATASET_PATH, index=False)

        # Also add the name for the response (name is not in the CSV schema,
        # so we return it separately for the frontend)
        new_row["name"] = req.name
        return {"message": "Student enrolled successfully", "student": new_row}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
