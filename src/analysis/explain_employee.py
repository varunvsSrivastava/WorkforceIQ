
"""Generate an individual employee SHAP explanation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data_loader import load_dataset
from src.preprocessing.prepare_features import prepare_features
from src.preprocessing.split_data import split_dataset
from src.pipeline.build_pipeline import build_pipeline
from src.models.random_forest import build_random_forest


def explain_employee(test_position: int = 0):
    """Generate SHAP explanation for one employee in the test set."""

    # Load dataset and prepare features.
    df = load_dataset()
    X, y = prepare_features(df)

    # Reproduce the Phase 5 train/test split.
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Build and train the tuned Random Forest pipeline.
    pipeline = build_pipeline(build_random_forest())
    pipeline.fit(X_train, y_train)

    # Validate employee position.
    if not 0 <= test_position < len(X_test):
        raise IndexError(
            f"test_position must be between 0 and {len(X_test) - 1}"
        )

    # Extract fitted preprocessing and classifier.
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    # Transform training and test features.
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # Select background data and employee.
    background = X_train_transformed[:100]
    employee = X_test_transformed[test_position:test_position + 1]

    # Create probability-based SHAP explainer.
    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="probability",
    )

    shap_values = explainer(employee)

    # Extract SHAP values for class 1 (Attrition = Yes).
    values = shap_values.values[0, :, 1]
    base_value = float(
        np.asarray(shap_values.base_values).reshape(1, -1)[0, 1]
    )

    # Get predicted probability for Attrition = Yes.
    predicted_probability = float(
        pipeline.predict_proba(X_test.iloc[[test_position]])[0, 1]
    )

    # Verify SHAP additivity.
    reconstructed_probability = base_value + values.sum()
    additivity_difference = abs(
        predicted_probability - reconstructed_probability
    )

    print("\nIndividual Employee SHAP Explanation")
    print("-" * 45)
    print(f"Test position: {test_position}")
    print(f"Actual attrition: {int(y_test.iloc[test_position])}")
    print(f"Predicted probability: {predicted_probability:.4f}")
    print(f"SHAP base value: {base_value:.4f}")
    print(
        f"SHAP reconstructed probability: "
        f"{reconstructed_probability:.4f}"
    )
    print(f"Additivity difference: {additivity_difference:.6f}")

    # Build single-output SHAP explanation.
    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=X_test_transformed[test_position],
        feature_names=feature_names,
    )

    # Generate waterfall plot.
    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False,
    )

    output_path = Path(
        "reports/figures/employee_shap_waterfall.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"\nSaved employee SHAP waterfall to: {output_path}")

    # Build feature contribution table.
    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values,
    })

    # Display positive and negative contributors.
    positive = contributions.nlargest(5, "shap_value")
    negative = contributions.nsmallest(5, "shap_value")

    print("\nTop Positive Contributors")
    print(positive.to_string(index=False))

    print("\nTop Negative Contributors")
    print(negative.to_string(index=False))

    return {
        "test_position": test_position,
        "predicted_probability": predicted_probability,
        "base_value": base_value,
        "shap_values": values,
        "reconstructed_probability": reconstructed_probability,
        "additivity_difference": additivity_difference,
        "contributions": contributions,
    }


def main():
    explain_employee(test_position=0)


if __name__ == "__main__":
    main()
