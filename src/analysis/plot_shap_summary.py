"""Generate SHAP summary visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
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

def plot_shap_summary():
    """Generate and save SHAP beeswarm summary plot."""

    # Load and prepare dataset
    df = load_dataset()
    X, y = prepare_features(df)

    # Recreate Phase 5 split
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Build and fit tuned pipeline
    pipeline = build_pipeline(build_tuned_random_forest())
    pipeline.fit(X_train, y_train)

    # Extract fitted components
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform test data
    X_test_transformed = preprocessor.transform(X_test)

    if issparse(X_test_transformed):
        X_test_transformed = X_test_transformed.toarray()

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Calculate SHAP values
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_transformed)

    # Extract positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # Remove preprocessing prefixes for readability
    display_feature_names = [
        name.replace("num__", "")
        .replace("nom__", "")
        .replace("ord__", "")
        for name in feature_names
    ]

    # Create SHAP beeswarm plot
    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=display_feature_names,
        max_display=15,
        plot_type="dot",
        show=False,
    )

    plt.title("SHAP Summary Plot: Employee Attrition")
    plt.tight_layout()

    # Save visualization
    output_path = Path("reports/figures/shap_summary.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved SHAP summary plot to: {output_path}")

    plt.show()

def main():
    """Run SHAP summary visualization."""

    plot_shap_summary()

if __name__ == "__main__":
    main()
