import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.data_loader import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

def analyze_categorical_distributions():
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

    # Display category counts and percentages
    for feature in categorical_features:
        print(f"\n--- {feature} Distribution ---")

        counts = df[feature].value_counts()
        percentages = (
            df[feature].value_counts(normalize=True) * 100
        ).round(2)

        distribution = pd.DataFrame({
            "Count": counts,
            "Percentage": percentages
        })

        print(distribution)

    # Create count plots
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(16, 20),
        constrained_layout=True
    )

    axes = axes.flatten()

    for index, feature in enumerate(categorical_features):
        sns.countplot(
            data=df,
            x=feature,
            order=df[feature].value_counts().index,
            ax=axes[index]
        )

        axes[index].set_title(f"Distribution of {feature}")
        axes[index].set_xlabel(feature)
        axes[index].set_ylabel("Employee Count")

        axes[index].tick_params(
            axis="x",
            rotation=35,
            labelsize=8
        )

    # Hide the unused subplot
    axes[-1].set_visible(False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / "categorical_distributions.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"Plot saved successfully: {output_path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    import pandas as pd

    analyze_categorical_distributions()
