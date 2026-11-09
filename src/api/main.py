
"""FastAPI application for WorkforceIQ."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from src.api.prediction import classify_attrition
from src.api.schemas import EmployeeInput, PredictionResponse

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "workforceiq_pipeline.joblib"

# Load trained model artifact
artifact = joblib.load(MODEL_PATH)

pipeline = artifact["pipeline"]
CLASSIFICATION_THRESHOLD = artifact["threshold"]
MODEL_NAME = artifact["model_name"]

# Initialize FastAPI
app = FastAPI(
    title="WorkforceIQ API",
    description="Employee Attrition Prediction API",
    version="1.0.0",
)


@app.get("/")
def root():
    """API welcome endpoint."""
    return {
        "message": "Welcome to WorkforceIQ API",
        "model": MODEL_NAME,
        "threshold": CLASSIFICATION_THRESHOLD,
    }


@app.get("/health")
def health_check():
    """API health endpoint."""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(employee: EmployeeInput):
    """Predict employee attrition."""

    # Convert validated input into DataFrame
    employee_data = pd.DataFrame([employee.model_dump()])

    # Predict attrition probability
    probability = float(pipeline.predict_proba(employee_data)[0][1])

    # Apply selected classification threshold
    prediction = classify_attrition(
        probability,
        CLASSIFICATION_THRESHOLD,
    )

    return PredictionResponse(
        attrition_prediction=prediction,
        attrition_probability=probability,
        classification_threshold=CLASSIFICATION_THRESHOLD,
        model_name=MODEL_NAME,
    )
