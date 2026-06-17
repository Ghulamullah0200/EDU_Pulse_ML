from fastapi.testclient import TestClient
import sys
import os

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from main import app
from models import StudentInput

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_valid_prediction():
    payload = {
        "age": 20,
        "gender": "Female",
        "department": "Engineering",
        "semester": 4,
        "study_hours_per_week": 15.5,
        "attendance_percentage": 85.0,
        "assignment_average": 80.0,
        "midterm_score": 75.0,
        "previous_gpa": 3.2,
        "internet_access": "Yes",
        "extra_academic_support": "No",
        "part_time_job": "No",
        "extracurricular_hours_per_week": 5.0,
        "absences": 2
    }
    response = client.post("/predict/single", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "risk_level" in data
    assert "important_factors" in data

def test_invalid_input_validation():
    # Age out of bounds (under 15)
    payload = {
        "age": 10,
        "gender": "Female",
        "department": "Engineering",
        "semester": 4,
        "study_hours_per_week": 15.5,
        "attendance_percentage": 85.0,
        "assignment_average": 80.0,
        "midterm_score": 75.0,
        "previous_gpa": 3.2,
        "internet_access": "Yes",
        "extra_academic_support": "No",
        "part_time_job": "No",
        "extracurricular_hours_per_week": 5.0,
        "absences": 2
    }
    response = client.post("/predict/single", json=payload)
    assert response.status_code == 422 # Unprocessable Entity (validation error)

def test_get_metrics():
    response = client.get("/metrics")
    if response.status_code == 200:
        data = response.json()
        assert "best_model" in data
        assert "metrics" in data

def test_get_feature_importance():
    response = client.get("/feature-importance")
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "feature" in data[0]
            assert "importance" in data[0]
