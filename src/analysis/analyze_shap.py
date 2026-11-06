"""Calculate SHAP values for Random Forest attrition predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.sparse import issparse

from src.data_loader import load_dataset
from src.preprocessing.prepare_features import prepare_features
from src.preprocessing.split_data import split_dataset
from src.models.random_forest import build_random_forest
from src.pipeline.build_pipeline import build_pipeline


def build_tuned_random_forest():
    """Build Random Forest using Phase 5 selected parameters."""

    model = build_random_forest()

    model.set_params(
        n_estimators=200,
        max_depth=None,
        max_features="log2",
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=None,
    )

    return model


def extract_positive_class_shap_values(shap_values):
    """Extract SHAP values for the positive class (Attrition = Yes)."""

    # Older SHAP versions may return a list for each class
    if isinstance(shap_values, list):
        return shap_values[1]

    # Newer SHAP versions may return 3D arrays
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]

    # Binary output may already be a 2D array
    if shap_values.ndim == 2:
        return shap_values

    raise ValueError(
        f"Unexpected SHAP output shape: {shap_values.shape}"
    )


def analyze_shap():
    """Calculate and save SHAP values for test employees."""

    # Load and prepare dataset
    df = load_dataset()
    X, y = prepare_features(df)

    # Recreate Phase 5 train-test split
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Build and fit tuned pipeline
    pipeline = build_pipeline(build_tuned_random_forest())
    pipeline.fit(X_train, y_train)

    # Extract fitted preprocessor and classifier
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform test data
    X_test_transformed = preprocessor.transform(X_test)

    # SHAP requires dense input for reliable TreeExplainer processing
    if issparse(X_test_transformed):
        X_test_transformed = X_test_transformed.toarray()

    # Retrieve transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Create SHAP explainer
    explainer = shap.TreeExplainer(classifier)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test_transformed)

    # Extract positive class SHAP values
    positive_shap_values = extract_positive_class_shap_values(
        shap_values
    )

    # Validate dimensions
    if positive_shap_values.shape != (
        len(X_test),
        len(feature_names),
    ):
        raise ValueError(
            "SHAP values dimensions do not match test samples "
            "and transformed features."
        )

    # Create SHAP dataframe
    shap_df = pd.DataFrame(
        positive_shap_values,
        columns=feature_names,
        index=X_test.index,
    )

    shap_df.index.name = "employee_index"

    # Save SHAP values
    output_path = Path("reports/shap_values.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shap_df.to_csv(output_path)

    # Display results
    print("\n--- SHAP Analysis ---")
    print("Test samples:", len(X_test))
    print("Transformed features:", len(feature_names))
    print("SHAP values shape:", positive_shap_values.shape)

    print("\n--- Top 10 Features by Mean Absolute SHAP ---")

    mean_abs_shap = np.abs(positive_shap_values).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values(
        by="mean_abs_shap",
        ascending=False,
    )

    print(
        importance_df.head(10).to_string(
            index=False,
            formatters={
                "mean_abs_shap": "{:.6f}".format,
            },
        )
    )

    print("\nSaved SHAP values to:", output_path)

    return shap_df, importance_df


def main():
    """Run SHAP analysis."""

    analyze_shap()


if __name__ == "__main__":
    main()
