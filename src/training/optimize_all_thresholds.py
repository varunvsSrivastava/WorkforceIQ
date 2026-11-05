"""
Optimize classification thresholds for all tuned WorkforceIQ models
using out-of-fold probabilities from the training set.
"""

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
from src.pipeline.build_pipeline import build_pipeline
from src.models.logistic_regression import build_logistic_regression
from src.models.decision_tree import build_decision_tree
from src.models.random_forest import build_random_forest
from src.models.gradient_boosting import build_gradient_boosting



def get_tuned_models():
    """Return pipelines configured with tuned hyperparameters."""

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
        print(f"\nModel: {name}")
        print("Pipeline steps:", pipeline.named_steps.keys())
        print("Classifier type:", type(pipeline.named_steps["classifier"]))
        print("Classifier:", pipeline.named_steps["classifier"])

    return models



def find_best_threshold(y_true, probabilities):
    """Find the threshold that maximizes F1 score."""

    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    # The final precision/recall point has no corresponding threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_index = np.argmax(f1_scores)

    return thresholds[best_index], f1_scores[best_index]


def evaluate_oof_threshold(y_true, probabilities, threshold):
    """Calculate OOF metrics at a selected threshold."""

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "positive_rate": predictions.mean(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    """Run OOF threshold optimization for all tuned models."""

    dataset = load_dataset()
    X, y = prepare_features(dataset)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    models = get_tuned_models()
    
    for name, model in models.items():
        print(name, model.named_steps["classifier"].get_params())

    print("\nOOF Threshold Optimization")
    print("=" * 70)
    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))
    print("CV folds:", cv.n_splits)

    for name, pipeline in models.items():
        print(f"\n{name}")
        print("-" * 50)

        probabilities = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        threshold, best_f1 = find_best_threshold(
            y_train, probabilities
        )

        metrics = evaluate_oof_threshold(
            y_train, probabilities, threshold
        )

        print(f"Best threshold:       {threshold:.4f}")
        print(f"OOF best F1:         {best_f1:.4f}")
        print(f"OOF precision:       {metrics['precision']:.4f}")
        print(f"OOF recall:           {metrics['recall']:.4f}")
        print(f"OOF F1:               {metrics['f1']:.4f}")
        print(f"Positive prediction:  {metrics['positive_rate']:.4f}")
        print(
            "Confusion Matrix: "
            f"TN={metrics['tn']}, FP={metrics['fp']}, "
            f"FN={metrics['fn']}, TP={metrics['tp']}"
        )


if __name__ == "__main__":
    main()
