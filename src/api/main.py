
"""FastAPI application for WorkforceIQ API."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI

from src.api.prediction import classify_attrition
from src.api.schemas import (
    EmployeeInput,
    FeatureContribution,
    PredictionResponse,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "workforceiq_pipeline.joblib"

# Load trained model artifact
artifact = joblib.load(MODEL_PATH)

pipeline = artifact["pipeline"]
CLASSIFICATION_THRESHOLD = artifact["threshold"]
MODEL_NAME = artifact["model_name"]

# Extract pipeline components
preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

# Initialize SHAP explainer
shap_explainer = shap.TreeExplainer(classifier)

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

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
    """Health endpoint."""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(employee: EmployeeInput):
    """Predict employee attrition with SHAP explanations."""

    # Convert validated input into DataFrame
    employee_data = pd.DataFrame([employee.model_dump()])

    # Predict attrition probability
    probability = float(pipeline.predict_proba(employee_data)[0][1])

    # Apply classification threshold
    prediction = classify_attrition(
        probability,
        CLASSIFICATION_THRESHOLD,
    )

    # Transform employee data for SHAP
    transformed_employee = preprocessor.transform(employee_data)

    # Calculate SHAP values
    shap_values = shap_explainer.shap_values(
        transformed_employee
    )

    # Extract SHAP values for attrition class (Yes)
    shap_array = np.asarray(shap_values)

    if isinstance(shap_values, list):
        employee_shap_values = shap_values[1][0]
    elif shap_array.ndim == 3:
        employee_shap_values = shap_array[0, :, 1]
    else:
        employee_shap_values = shap_array[0]

    # Select top 10 features by absolute SHAP impact
    top_indices = np.argsort(
        np.abs(employee_shap_values)
    )[::-1][:10]

    # Prepare feature contributions
    feature_contributions = [
        FeatureContribution(
            feature=str(feature_names[index]),
            impact=float(employee_shap_values[index]),
        )
        for index in top_indices
    ]

    # Return prediction and explanation
    return PredictionResponse(
        attrition_prediction=prediction,
        attrition_probability=probability,
        classification_threshold=CLASSIFICATION_THRESHOLD,
        model_name=MODEL_NAME,
        feature_contributions=feature_contributions,
    )
