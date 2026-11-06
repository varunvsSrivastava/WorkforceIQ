"""Plot global feature importance ranking comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_global_feature_importance():
    """Compare feature rankings across MDI, permutation, and SHAP."""

    # Load consolidated importance comparison
    input_path = Path(
        "reports/global_feature_importance_comparison.csv"
    )
    comparison_df = pd.read_csv(input_path)

    # Select top 15 features according to SHAP ranking
    top_features = comparison_df.nsmallest(
        15,
        "shap_rank",
    ).copy()

    # Sort for horizontal chart
    top_features = top_features.sort_values(
        "shap_rank",
        ascending=False,
    )

    features = top_features["feature"]
    positions = np.arange(len(features))
    bar_width = 0.25

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.barh(
        positions - bar_width,
        top_features["mdi_rank"],
        height=bar_width,
        label="MDI Rank",
    )

    ax.barh(
        positions,
        top_features["permutation_rank"],
        height=bar_width,
        label="Permutation Rank",
    )

    ax.barh(
        positions + bar_width,
        top_features["shap_rank"],
        height=bar_width,
        label="SHAP Rank",
    )

    # Configure axes
    ax.set_yticks(positions)
    ax.set_yticklabels(features)

    ax.set_xlabel("Feature Rank (Lower = More Important)")
    ax.set_ylabel("Feature")
    ax.set_title("Global Feature Importance Ranking Comparison")

    # Rank 1 should appear at the right side
    ax.invert_xaxis()

    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()

    # Save visualization
    output_path = Path(
        "reports/figures/global_feature_importance_comparison.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved global importance comparison plot to: {output_path}")

    plt.show()

def main():
    """Run global feature importance visualization."""

    plot_global_feature_importance()

if __name__ == "__main__":
    main()
