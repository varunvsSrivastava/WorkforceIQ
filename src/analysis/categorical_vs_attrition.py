import matplotlib.pyplot as plt
from pathlib import Path
from src.data_loader import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

def analyze_categorical_features_by_attrition():
    df = load_dataset()

    categorical_features = [
        "BusinessTravel",
        "Department",
        "EducationField",
        "Gender",
        "JobRole",
        "MaritalStatus",
        "OverTime",
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(16, 20),
        constrained_layout=True
    )

    axes = axes.flatten()

    for index, feature in enumerate(categorical_features):

        # Calculate attrition rate for each category
        attrition_rates = (
            df.groupby(feature)["Attrition"]
            .apply(lambda values: (values == "Yes").mean() * 100)
            .sort_values(ascending=True)
        )

        print(f"\n--- Attrition Rate by {feature} ---")
        print(attrition_rates.round(2))

        # Plot horizontal bar chart
        axes[index].barh(
            attrition_rates.index.astype(str),
            attrition_rates.values
        )

        axes[index].set_title(f"Attrition Rate by {feature}")
        axes[index].set_xlabel("Attrition Rate (%)")
        axes[index].set_ylabel(feature)

    # Hide unused subplot
    fig.delaxes(axes[-1])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / "categorical_vs_attrition.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"Plot saved successfully: {output_path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    analyze_categorical_features_by_attrition()
