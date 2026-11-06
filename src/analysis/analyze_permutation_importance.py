"""Analyze permutation importance for Random Forest."""

from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from src.data_loader import load_dataset
from src.preprocessing.prepare_features import prepare_features
from src.preprocessing.split_data import split_dataset
from src.models.random_forest import build_random_forest
from src.pipeline.build_pipeline import build_pipeline

def build_tuned_random_forest():
    """Build Random Forest using phase 5 selected parameters."""
    
    model = build_random_forest()
    
    model.set_params(
        n_estimators=200,
        max_depth=None,
        max_features="log2",
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=None,
    )
    
    return model

def analyze_permutation_importance():
    """Calculate permutation importance on the test dataset."""
    
    #Load and prepare dataset
    df=load_dataset()
    X,y=prepare_features(df)
    
    #Recreate phase 5 train-test split
    X_train,X_test,y_train,y_test=split_dataset(X,y)
    
    #Build and fit tuned pipeline
    pipeline = build_pipeline(build_tuned_random_forest())
    pipeline.fit(X_train,y_train)
    
    #Calculate baseline test ROC-AUC
    baseline_auc = roc_auc_score(
        y_test,
        pipeline.predict_proba(X_test)[:,1],
    )
    
    #Calculate permutation importance
    result = permutation_importance(
        estimator=pipeline,
        X=X_test,
        y=y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    
    #Create importance DataFrame
    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    
    #Sort by mean importance
    importance_df = importance_df.sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(drop=True)
    
    #Save results
    output_path = Path("reports/permutation_importance.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_path,index=False)
    
    # Display results
    print("\n--- Permutation Importance ---")
    print("Baseline Test ROC-AUC:", round(baseline_auc, 4))
    print("Test samples:", len(X_test))
    print("Original features:", X_test.shape[1])
    print("Permutation repeats: 10")

    print("\n--- Top 15 Important Features ---")
    print(
        importance_df.head(15).to_string(
            index=False,
            formatters={
                "importance_mean": "{:.6f}".format,
                "importance_std": "{:.6f}".format,
            },
        )
    )

    print("\nSaved permutation importance to:", output_path)

    return importance_df

def main():
    """Run permutation importance analysis."""

    analyze_permutation_importance()

if __name__ == "__main__":
    main()