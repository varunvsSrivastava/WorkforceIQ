"""Generate SHAP waterfall plot for an individual employee."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

def plot_shap_waterfall():
    """Explain one test employee's attrition prediction."""

    # Load and prepare dataset
    df = load_dataset()
    X, y = prepare_features(df)

    # Recreate Phase 5 train-test split
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Build and fit tuned pipeline
    pipeline = build_pipeline(build_tuned_random_forest())
    pipeline.fit(X_train, y_train)

    # Extract fitted components
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform training and test data
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if issparse(X_train_transformed):
        X_train_transformed = X_train_transformed.toarray()

    if issparse(X_test_transformed):
        X_test_transformed = X_test_transformed.toarray()

    # Feature names
    feature_names = preprocessor.get_feature_names_out()

    display_feature_names = [
        name.replace("num__", "")
        .replace("nom__", "")
        .replace("ord__", "")
        for name in feature_names
    ]

    # Select one employee from test set
    employee_position = 0
    employee_data = X_test_transformed[employee_position]

    # Create probability-based SHAP explainer
    background_data = X_train_transformed[:100]

    explainer = shap.TreeExplainer(
        classifier,
        data=background_data,
        feature_perturbation="interventional",
        model_output="probability",
    )

    # Calculate SHAP values for selected employee
    shap_values = explainer.shap_values(
        employee_data.reshape(1, -1)
    )

    # Extract positive class SHAP values
    if isinstance(shap_values, list):
        employee_shap_values = shap_values[1][0]
    elif shap_values.ndim == 3:
        employee_shap_values = shap_values[0, :, 1]
    else:
        employee_shap_values = shap_values[0]

    # Extract base value for positive class
    base_value = explainer.expected_value

    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]

    # Create SHAP Explanation object
    explanation = shap.Explanation(
        values=employee_shap_values,
        base_values=base_value,
        data=employee_data,
        feature_names=display_feature_names,
    )

    # Generate waterfall plot
    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False,
    )

    plt.title("SHAP Waterfall: Individual Employee Attrition")
    plt.tight_layout()

    # Save visualization
    output_path = Path(
        "reports/figures/shap_waterfall_employee.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    # Model prediction
    employee_probability = pipeline.predict_proba(
        X_test.iloc[[employee_position]]
    )[0, 1]

    employee_actual = y_test.iloc[employee_position]

    print("\n--- Individual Employee SHAP Explanation ---")
    print("Test employee position:", employee_position)
    print("Actual Attrition:", "Yes" if employee_actual == 1 else "No")
    print("Predicted Attrition Probability:", round(employee_probability, 4))
    print("SHAP Base Value:", round(float(base_value), 4))
    print("Sum of SHAP Values:", round(employee_shap_values.sum(), 4))

    print("\nSaved waterfall plot to:", output_path)

    plt.show()

def main():
    """Run individual employee SHAP explanation."""

    plot_shap_waterfall()

if __name__ == "__main__":
    main()
