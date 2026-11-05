"""Optimize Logistic Regression classification threshold using OOF predictions."""

import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.data_loader import load_dataset
from src.preprocessing.prepare_features import prepare_features
from src.preprocessing.split_data import split_dataset
from src.models.logistic_regression import build_logistic_regression
from src.pipeline.build_pipeline import build_pipeline


def evaluate_threshold(y_true, probabilities, threshold):
    """Evaluate classification metrics at a given threshold."""

    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    matrix = confusion_matrix(y_true, predictions)

    print(f"\nThreshold: {threshold:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(matrix)


def main():
    # Load and prepare dataset
    df = load_dataset()
    X, y = prepare_features(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Build tuned Logistic Regression pipeline
    pipeline = build_pipeline(build_logistic_regression())

    # Define cross-validation strategy
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # Generate out-of-fold probabilities on training data
    oof_probabilities = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    # Find threshold maximizing F1
    precision, recall, thresholds = precision_recall_curve(
        y_train,
        oof_probabilities,
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]

    print("Logistic Regression Threshold Optimization")
    print(f"OOF Best F1: {f1_scores[best_index]:.4f}")
    print(f"Optimal Threshold: {best_threshold:.4f}")

    # Compare default and optimized thresholds on OOF predictions
    print("\n--- Default Threshold ---")
    evaluate_threshold(y_train, oof_probabilities, 0.5)

    print("\n--- Optimized Threshold ---")
    evaluate_threshold(y_train, oof_probabilities, best_threshold)


if __name__ == "__main__":
    main()
