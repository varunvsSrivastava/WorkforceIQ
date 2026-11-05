"""
Evaluate tuned WorkforceIQ models using OOF-optimized thresholds.
"""

import pandas as pd

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
from src.pipeline.build_pipeline import build_pipeline

from src.models.logistic_regression import build_logistic_regression
from src.models.decision_tree import build_decision_tree
from src.models.random_forest import build_random_forest
from src.models.gradient_boosting import build_gradient_boosting


def get_tuned_models():
    """Return tuned model pipelines."""

    models = {
        "Logistic Regression": build_pipeline(build_logistic_regression()),
        "Decision Tree": build_pipeline(build_decision_tree()),
        "Random Forest": build_pipeline(build_random_forest()),
        "Gradient Boosting": build_pipeline(build_gradient_boosting()),
    }

    tuned_params = {
        "Logistic Regression": {
            "classifier__C": 1,
            "classifier__class_weight": None,
            "classifier__solver": "lbfgs",
        },
        "Decision Tree": {
            "classifier__class_weight": "balanced",
            "classifier__max_depth": 3,
            "classifier__min_samples_leaf": 4,
            "classifier__min_samples_split": 2,
        },
        "Random Forest": {
            "classifier__n_estimators": 200,
            "classifier__max_depth": None,
            "classifier__max_features": "log2",
            "classifier__min_samples_split": 5,
            "classifier__min_samples_leaf": 2,
            "classifier__class_weight": None,
        },
        "Gradient Boosting": {
            "classifier__learning_rate": 0.1,
            "classifier__max_depth": 2,
            "classifier__min_samples_leaf": 2,
            "classifier__min_samples_split": 2,
            "classifier__n_estimators": 200,
            "classifier__subsample": 0.8,
        },
    }

    for name, pipeline in models.items():
        pipeline.set_params(**tuned_params[name])

    return models


def evaluate_predictions(y_true, probabilities, threshold):
    """Evaluate probabilities using a specified threshold."""

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def main():
    """Evaluate all tuned models on the held-out test set."""

    dataset = load_dataset()
    X, y = prepare_features(dataset)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    models = get_tuned_models()

    optimized_thresholds = {
        "Logistic Regression": 0.3678,
        "Decision Tree": 0.2051,
        "Random Forest": 0.2400,
        "Gradient Boosting": 0.2257,
    }

    results = []

    print("\nHeld-Out Test Evaluation")
    print("=" * 70)
    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))

    for name, pipeline in models.items():
        print(f"\n{name}")
        print("-" * 50)

        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        threshold_05 = evaluate_predictions(
            y_test, probabilities, 0.5
        )

        optimized_threshold = optimized_thresholds[name]

        threshold_opt = evaluate_predictions(
            y_test, probabilities, optimized_threshold
        )

        for threshold_name, threshold, metrics in [
            ("Default", 0.5, threshold_05),
            ("Optimized", optimized_threshold, threshold_opt),
        ]:
            row = {
                "Model": name,
                "Threshold Type": threshold_name,
                "Threshold": threshold,
                **metrics,
            }
            results.append(row)

            print(f"\n{threshold_name} threshold: {threshold:.4f}")
            for metric in [
                "Accuracy", "Precision", "Recall",
                "F1", "ROC-AUC", "PR-AUC",
            ]:
                print(f"{metric:12s}: {metrics[metric]:.4f}")

            print(
                "Confusion Matrix: "
                f"TN={metrics['TN']}, FP={metrics['FP']}, "
                f"FN={metrics['FN']}, TP={metrics['TP']}"
            )

    results_df = pd.DataFrame(results)

    print("\n\nFinal Test Results Summary")
    print("=" * 100)
    print(results_df.to_string(index=False))

    output_path = "reports/threshold_evaluation.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
