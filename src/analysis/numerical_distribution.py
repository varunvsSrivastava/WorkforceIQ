import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.data_loader import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

def analyze_numerical_distribution():
    df =load_dataset()

    numerical_features = [
        "Age",
        "DailyRate",
        "DistanceFromHome",
        "HourlyRate",
        "MonthlyIncome",
        "MonthlyRate",
        "NumCompaniesWorked",
        "PercentSalaryHike",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
    ]

    print("--- Numerical Feature Summary ---")
    print(df[numerical_features].describe().round(2))

    fig, axes = plt.subplots(
        nrows=5,
        ncols=3,
        figsize=(18,24),
        constrained_layout=True
    )

    axes = axes.flatten()

    for index,feature in enumerate(numerical_features):
        sns.histplot(
            data=df,
            x=feature,
            bins=20,
            ax=axes[index]
        )

        axes[index].tick_params(
                        axis="x",
                        labelrotation=30,
                        labelsize=8
                    )

        axes[index].set_title(f"Distribution of {feature}")
        axes[index].set_xlabel(feature)
        axes[index].set_ylabel("Employee Count")

    fig.delaxes(axes[-1])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / "numerical_distributions.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"Plot saved successfully: {output_path}")

    plt.show()
    plt.close()

if __name__=="__main__":
    analyze_numerical_distribution()
