"""Plot Random Forest feature importance."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance():
    """Plot and save the top 15 Random Forest features."""
    
    #Load feature importance results
    input_path = Path("reports/random_forest_feature_importance.csv")
    importance_df = pd.read_csv(input_path)
    
    #Select top 15 features
    top_features = importance_df.head(15).copy()
    
    #Remove preprocessing prefixes for readability
    top_features["feature"] = (
        top_features["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("nom__", "", regex=False)
        .str.replace("ord__", "", regex=False)
    )
    
    #Reverse order for horizontal bar chart
    top_features = top_features.iloc[::-1]
    
    #Create figure
    plt.figure(figsize=(10,8))
    
    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )
    
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Top 15 Random Forest Feature Importance")
    plt.tight_layout()
    
    #Save visualization
    output_path = Path("reports/figures/random_forest_feature_importance.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    
    print(f"Saved feature importance plot to: {output_path}")
    
    #Display visualization
    plt.show()
    

def main():
    """Run feature importance visualization."""
    
    plot_feature_importance()
    
if __name__=="__main__":
    main()
    