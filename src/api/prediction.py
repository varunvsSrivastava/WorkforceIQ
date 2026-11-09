"""Prediction utilities for WorkforceIQ API."""

def classify_attrition(probability: float, threshold: float) -> str:
    """Classify attrition using the configured probability threshold."""
    return "Yes" if probability >= threshold else "No"
