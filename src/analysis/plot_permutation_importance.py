"""Plot permutation importance."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def plot_permutation_importance():
    """Plot and save top 15 permutation importance features."""

    # Load permutation importance results
    input_path = Path("reports/permutation_importance.csv")
    importance_df = pd.read_csv(input_path)

    # Select top 15 features
    top_features = importance_df.head(15).copy()

    # Reverse order for horizontal bar chart
    top_features = top_features.iloc[::-1]

    # Create figure
    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
        capsize=3,
    )

    plt.xlabel("Mean ROC-AUC Decrease")
    plt.ylabel("Feature")
    plt.title("Top 15 Permutation Feature Importance")
    plt.tight_layout()

    # Save visualization
    output_path = Path(
        "reports/figures/permutation_importance.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved permutation importance plot to: {output_path}")

    # Display visualization
    plt.show()

def main():
    """Run permutation importance visualization."""

    plot_permutation_importance()

if __name__ == "__main__":
    main()

