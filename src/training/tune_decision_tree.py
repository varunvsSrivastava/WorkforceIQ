"""Tune Decision Tree using GridSearchCV."""

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data_loader import load_dataset
from src.preprocessing.prepare_features import prepare_features
from src.preprocessing.split_data import split_dataset
from src.models.decision_tree import build_decision_tree


def main():
    # Load and prepare dataset
    df = load_dataset()
    X, y = prepare_features(df)

    # Create train-test split
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Build pipeline
    pipeline = build_decision_tree()

    # Define cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # Define parameter grid
    param_grid = {
        "classifier__max_depth": [3, 5, 7, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__class_weight": [None, "balanced"],
    }

    # Configure GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    # Fit on training data only
    grid_search.fit(X_train, y_train)

    # Display results
    print("\n--- Decision Tree Tuning Results ---")
    print(f"Best ROC-AUC: {grid_search.best_score_:.4f}")
    print("Best Parameters:")
    for parameter, value in grid_search.best_params_.items():
        print(f"{parameter}: {value}")

    # Evaluate best model on held-out test set
    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(X_test, y_test)
    test_roc_auc = best_model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import roc_auc_score

    print("\n--- Best Decision Tree Test Results ---")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, test_roc_auc):.4f}")


if __name__ == "__main__":
    main()
