"""Tune Random Forest hyperparameters using GridSearchCV."""

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data_loader import load_dataset
from src.preprocessing.prepare_features import prepare_features
from src.preprocessing.split_data import split_dataset
from src.models.random_forest import build_random_forest
from src.pipeline.build_pipeline import build_pipeline


def main():
    # Load and prepare dataset
    df = load_dataset()
    X, y = prepare_features(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Build pipeline
    pipeline = build_pipeline(build_random_forest())

    # Define cross-validation strategy
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # Define hyperparameter grid
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
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

    # Train and tune model
    grid_search.fit(X_train, y_train)

    # Display results
    print("\nBest ROC-AUC:", round(grid_search.best_score_, 4))
    print("\nBest Parameters:")

    for parameter, value in grid_search.best_params_.items():
        print(f"{parameter}: {value}")

    # Evaluate on test set
    best_model = grid_search.best_estimator_

    test_accuracy = best_model.score(X_test, y_test)
    test_roc_auc = __import__("sklearn.metrics").metrics.roc_auc_score(
        y_test,
        best_model.predict_proba(X_test)[:, 1],
    )

    print("\nBest Random Forest Test Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")


if __name__ == "__main__":
    main()

