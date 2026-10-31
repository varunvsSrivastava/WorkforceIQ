from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_ROOT / "reports" / "eda_summary.txt"


def generate_eda_report():
    df = load_dataset()

    report = []

    # 1. Dataset Overview
    report.append("=" * 60)
    report.append("WORKFORCEIQ - EXPLORATORY DATA ANALYSIS REPORT")
    report.append("=" * 60)

    report.append("\n1. DATASET OVERVIEW")
    report.append(f"Total Records: {df.shape[0]}")
    report.append(f"Total Features: {df.shape[1]}")

    # 2. Attrition Distribution
    report.append("\n2. ATTRITION DISTRIBUTION")

    attrition_counts = df["Attrition"].value_counts()
    attrition_percentages = (
        df["Attrition"].value_counts(normalize=True) * 100
    )

    for category in ["No", "Yes"]:
        report.append(
            f"{category}: {attrition_counts.get(category, 0)} "
            f"({attrition_percentages.get(category, 0):.2f}%)"
        )

    # 3. Numerical Features vs Attrition
    report.append("\n3. NUMERICAL FEATURES VS ATTRITION")

    numerical_features = [
        "Age",
        "MonthlyIncome",
        "DistanceFromHome",
        "TotalWorkingYears",
        "YearsAtCompany",
        "YearsInCurrentRole",
    ]

    median_comparison = (
        df.groupby("Attrition")[numerical_features]
        .median()
    )

    report.append(median_comparison.to_string())

    # 4. Categorical Features vs Attrition
    report.append("\n4. CATEGORICAL FEATURES VS ATTRITION")

    categorical_features = [
        "BusinessTravel",
        "Department",
        "EducationField",
        "Gender",
        "JobRole",
        "MaritalStatus",
        "OverTime",
    ]

    for feature in categorical_features:
        grouped = df.groupby(feature, observed=True)["Attrition"]

        category_counts = grouped.size()
        attrition_rates = grouped.apply(
            lambda values: (values == "Yes").mean() * 100
        )

        results = pd.DataFrame({
            "Employees": category_counts,
            "Attrition Rate (%)": attrition_rates.round(2)
        })

        results = results.sort_values(
            "Attrition Rate (%)",
            ascending=False
        )

        report.append(f"\n{feature}:")
        report.append(results.to_string())

    # 5. Correlation with Attrition
    report.append("\n5. CORRELATION WITH ATTRITION")

    numerical_df = df.select_dtypes(include=["number"])

    excluded_columns = [
        "EmployeeNumber",
        "EmployeeCount",
        "StandardHours",
    ]

    numerical_df = numerical_df.drop(
        columns=excluded_columns,
        errors="ignore"
    )

    numerical_df["Attrition"] = (
        df["Attrition"].map({"No": 0, "Yes": 1})
    )

    correlation_matrix = numerical_df.corr()

    attrition_correlations = (
        correlation_matrix["Attrition"]
        .drop("Attrition")
        .sort_values()
    )

    report.append(attrition_correlations.round(3).to_string())

    # 6. Strong Feature Correlations
    report.append("\n6. STRONG FEATURE CORRELATIONS")

    upper_triangle = np.triu(
        np.ones(correlation_matrix.shape),
        k=1
    ).astype(bool)

    feature_correlations = (
        correlation_matrix.where(upper_triangle)
        .stack()
        .sort_values(key=abs, ascending=False)
    )

    report.append(feature_correlations.head(10).round(3).to_string())

    # 7. Business Insights
    report.append("\n7. KEY BUSINESS INSIGHTS")

    report.append(
        "- The dataset contains an imbalanced attrition target."
    )
    report.append(
        "- Employees who left generally show lower median "
        "income and shorter company tenure."
    )
    report.append(
        "- Overtime is associated with a higher attrition rate."
    )
    report.append(
        "- Sales Representatives show a relatively high "
        "attrition rate."
    )
    report.append(
        "- Single employees show a higher attrition rate "
        "than married employees."
    )
    report.append(
        "- Several experience and income-related features "
        "are strongly correlated."
    )

    # 8. Modeling Considerations
    report.append("\n8. MODELING CONSIDERATIONS")

    report.append(
        "- Use stratified train-test splitting because "
        "the target classes are imbalanced."
    )
    report.append(
        "- Evaluate precision, recall, F1-score and ROC-AUC "
        "along with accuracy."
    )
    report.append(
        "- Fit preprocessing and resampling operations "
        "using training data only."
    )
    report.append(
        "- Investigate correlated features during model "
        "comparison and interpretation."
    )
    report.append(
        "- Treat EDA findings as associations, not causal "
        "relationships."
    )

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        "\n".join(line.rstrip() for line in report),
        encoding="utf-8"
    )

    print(f"EDA report generated successfully: {REPORT_PATH}")


if __name__ == "__main__":
    generate_eda_report()
