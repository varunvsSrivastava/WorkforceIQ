"""Generate individual employee attrition predictions."""

from pathlib import Path

import pandas as pd

from src.data_loader import load_dataset
from src.preprocessing.prepare_features import prepare_features
from src.preprocessing.split_data import split_dataset
from src.pipeline.build_pipeline import build_pipeline
from src.models.random_forest import build_random_forest

OPTIMAL_THRESHOLD = 0.24

def classify_prediction(actual: int, predicted: int) -> str:
    """Classify an employee prediction outcome."""

    if actual == 1 and predicted == 1:
        return "True Positive"
    if actual == 0 and predicted == 0:
        return "True Negative"
    if actual == 0 and predicted == 1:
        return "False Positive"
    return "False Negative"

def predict_employee(test_position: int = 0) -> dict:
    """Predict attrition for one employee from the test dataset."""

    df = load_dataset()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    pipeline = build_pipeline(build_random_forest())
    pipeline.fit(X_train, y_train)

    if not 0 <= test_position < len(X_test):
        raise IndexError(
            f"test_position must be between 0 and {len(X_test) - 1}"
        )

    employee = X_test.iloc[[test_position]]
    actual = int(y_test.iloc[test_position])

    probability = float(pipeline.predict_proba(employee)[0, 1])
    predicted = int(probability >= OPTIMAL_THRESHOLD)

    outcome = classify_prediction(actual, predicted)

    return {
        "test_position": test_position,
        "actual_attrition": actual,
        "predicted_attrition": predicted,
        "attrition_probability": probability,
        "threshold": OPTIMAL_THRESHOLD,
        "prediction_outcome": outcome,
    }

def main():
    result = predict_employee(test_position=0)

    print("\nIndividual Employee Prediction")
    print("-" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
