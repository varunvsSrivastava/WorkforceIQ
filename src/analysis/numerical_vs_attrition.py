import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.data_loader import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

def analyze_numerical_features_by_attrition():
    df=load_dataset()

    numerical_features = [
        "Age",
        "MonthlyIncome",
        "DistanceFromHome",
        "TotalWorkingYears",
        "YearsAtCompany",
        "YearsInCurrentRole"
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(16,10),
        constrained_layout=True
    )

    axes = axes.flatten()

    for index, feature in enumerate(numerical_features):
        sns.boxplot(
            data=df,
            x="Attrition",
            y=feature,
            hue="Attrition",
            order=["No","Yes"],
            palette="Set2",
            legend=False,
            ax=axes[index]
        )

        axes[index].set_title(f"{feature} vs Attrition")
        axes[index].set_xlabel("Attrition")
        axes[index].set_ylabel(feature)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / "numerical_vs_attrition.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"Plot saved successfully: {output_path}")

    plt.show()
    plt.close()

if __name__=="__main__":
    analyze_numerical_features_by_attrition()
