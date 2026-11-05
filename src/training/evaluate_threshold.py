"""Evaluate Logistic Regression at different thresholds on the test set."""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.data_loader import load_dataset
from src.preprocessing.prepare_features import prepare_features
from src.preprocessing.split_data import split_dataset
from src.models.logistic_regression import build_logistic_regression
from src.pipeline.build_pipeline import build_pipeline


def evaluate_threshold(y_true, probabilities, threshold):
    """Evaluate metrics at a specified threshold."""

    predictions = (probabilities >= threshold).astype(int)

    print(f"\nThreshold: {threshold:.4f}")
    print(f"Accuracy:  {accuracy_score(y_true, predictions):.4f}")
    print(f"Precision: {precision_score(y_true, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, predictions, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_true, predictions, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, predictions))


def main():
    # Load and prepare dataset
    df = load_dataset()
    X, y = prepare_features(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Build and train Logistic Regression pipeline
    model = build_pipeline(build_logistic_regression())
    model.fit(X_train, y_train)

    # Predict probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    # Threshold-independent metrics
    print("Logistic Regression Test Evaluation")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, probabilities):.4f}")

    # Compare thresholds
    print("\n--- Default Threshold ---")
    evaluate_threshold(y_test, probabilities, 0.5)

    print("\n--- Optimized Threshold ---")
    evaluate_threshold(y_test, probabilities, 0.3678)


if __name__ == "__main__":
    main()
