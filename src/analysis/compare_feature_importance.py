"""Compare MDI, permutation, and SHAP feature importance."""

from pathlib import Path

import numpy as np
import pandas as pd

NOMINAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

def get_original_feature(transformed_feature):
    """Map transformed feature name to original feature name."""

    if transformed_feature.startswith("num__"):
        return transformed_feature.replace("num__", "", 1)

    if transformed_feature.startswith("ord__"):
        return transformed_feature.replace("ord__", "", 1)

    if transformed_feature.startswith("nom__"):
        encoded_feature = transformed_feature.replace("nom__", "", 1)

        for feature in NOMINAL_FEATURES:
            if encoded_feature.startswith(feature + "_"):
                return feature

    raise ValueError(
        f"Unable to map transformed feature: {transformed_feature}"
    )

def compare_feature_importance():
    """Compare global feature importance across three methods."""

    # Load importance results
    mdi_df = pd.read_csv(
        "reports/random_forest_feature_importance.csv"
    )

    permutation_df = pd.read_csv(
        "reports/permutation_importance.csv"
    )

    shap_df = pd.read_csv(
        "reports/shap_values.csv",
        index_col="employee_index",
    )

    # Aggregate MDI importance to original features
    mdi_df["original_feature"] = mdi_df["feature"].apply(
        get_original_feature
    )

    mdi_aggregated = (
        mdi_df.groupby("original_feature")["importance"]
        .sum()
        .rename("mdi_importance")
    )

    # Permutation importance already uses original features
    permutation_df = permutation_df.set_index("feature")

    permutation_mean = permutation_df["importance_mean"].rename(
        "permutation_importance"
    )

    permutation_std = permutation_df["importance_std"].rename(
        "permutation_std"
    )

    # Aggregate mean absolute SHAP values to original features
    shap_importance = {}

    for feature in shap_df.columns:
        original_feature = get_original_feature(feature)

        mean_abs_shap = np.abs(shap_df[feature]).mean()

        shap_importance[original_feature] = (
            shap_importance.get(original_feature, 0.0)
            + mean_abs_shap
        )

    shap_aggregated = pd.Series(
        shap_importance,
        name="mean_abs_shap",
    )

    # Combine all importance methods
    comparison_df = pd.concat(
        [
            mdi_aggregated,
            permutation_mean,
            permutation_std,
            shap_aggregated,
        ],
        axis=1,
    )

    # Add rankings
    comparison_df["mdi_rank"] = comparison_df[
        "mdi_importance"
    ].rank(ascending=False, method="min")

    comparison_df["permutation_rank"] = comparison_df[
        "permutation_importance"
    ].rank(ascending=False, method="min")

    comparison_df["shap_rank"] = comparison_df[
        "mean_abs_shap"
    ].rank(ascending=False, method="min")

    # Sort by SHAP importance
    comparison_df = comparison_df.sort_values(
        by="mean_abs_shap",
        ascending=False,
    )

    comparison_df = comparison_df.reset_index()
    comparison_df = comparison_df.rename(
        columns={"index": "feature"}
    )

    # Save comparison
    output_path = Path(
        "reports/global_feature_importance_comparison.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(output_path, index=False)

    # Display top features
    print("\n--- Global Feature Importance Comparison ---")

    print(
        comparison_df.head(15).to_string(
            index=False,
            formatters={
                "mdi_importance": "{:.6f}".format,
                "permutation_importance": "{:.6f}".format,
                "permutation_std": "{:.6f}".format,
                "mean_abs_shap": "{:.6f}".format,
            },
        )
    )

    print("\nTotal original features:", len(comparison_df))
    print("Saved comparison to:", output_path)

    return comparison_df

def main():
    """Run global feature importance comparison."""

    compare_feature_importance()

if __name__ == "__main__":
    main()
