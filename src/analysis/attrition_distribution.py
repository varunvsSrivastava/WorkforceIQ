import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.data_loader import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

def analyze_attrition_distribution():
    df=load_dataset()
    attrition_counts = df["Attrition"].value_counts()
    attrition_percentages = (
        df["Attrition"].value_counts(normalize=True)*100
    ).round(2)

    print("--- Attrition Counts ---")
    print(attrition_counts)

    print("\n--- Attrition Percentages ---")
    print(attrition_percentages)

    plt.figure(figsize=(7,5))

    sns.countplot(
        data=df,
        x="Attrition",
        hue="Attrition",
        order=["No","Yes"],
        palette="Set2",
        legend=False
    )

    plt.title("Employee Attrition Distribution")
    plt.xlabel("Attrition")
    plt.ylabel("Number of Employees")

    plt.tight_layout()

    # Create the figures directory if it does not exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Save the visualization
    output_path = FIGURES_DIR / "attrition_distribution.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"Plot saved successfully: {output_path}")

    plt.show()
    plt.close()

if __name__=="__main__":
    analyze_attrition_distribution()
