import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.data_loader import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

def analyze_correlations():
    df = load_dataset()

    # Exclude identifier and constant administrative columns
    excluded_columns = [
        "EmployeeNumber",
        "EmployeeCount",
        "StandardHours",
    ]

    numerical_df = df.select_dtypes(include=["number"]).drop(
        columns=excluded_columns,
        errors="ignore"
    )

    # Encode target variable: No = 0, Yes = 1
    numerical_df["Attrition"] = (
        df["Attrition"].map({"No": 0, "Yes": 1})
    )

    # Calculate Pearson correlation
    correlation_matrix = numerical_df.corr(method="pearson")

    print("--- Correlation with Attrition ---")

    attrition_correlation = (
        correlation_matrix["Attrition"]
        .drop("Attrition")
        .sort_values(ascending=False)
    )

    print(attrition_correlation.round(3))

    # Display correlation heatmap
    plt.figure(figsize=(16, 12))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / "correlation_heatmap.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"Plot saved successfully: {output_path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    analyze_correlations()
