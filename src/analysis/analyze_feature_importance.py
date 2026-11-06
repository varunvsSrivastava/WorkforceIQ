"""Analyze Random Forest feature importance."""

import pandas as pd

from src.data_loader import load_dataset
from src.preprocessing.prepare_features import prepare_features
from src.preprocessing.split_data import split_dataset
from src.models.random_forest import build_random_forest
from src.pipeline.build_pipeline import build_pipeline

def build_tuned_random_forest():
    """Build Random Forest with phase 5 selected hyperparameter."""
    
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

def analyze_feature_importance():
    """Fit the tuned pipeline and extract feature importance."""
    
    #Load and prepare dataset
    df=load_dataset()
    X,y = prepare_features(df)
    
    #Recreate the Phase 5 train-test split
    X_train,X_test,y_train,y_test=split_dataset(X,y)
    
    #Build tuned pipeline
    pipeline = build_pipeline(build_tuned_random_forest())
    
    #Fit using training data only
    pipeline.fit(X_train,y_train)
    
    #Extract fitted preprocessing and classifier
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    #Retrieve transformed feature names
    feature_names = preprocessor.get_feature_names_out()
    
    #Retrieve impurity-based feature imortance
    importance_scores = classifier.feature_importances_
    
    #Validate feature alignment
    if len(feature_names) != len(importance_scores):
        raise ValueError(
            "Feature names and importance scores do not match."
        )
    
    #Create importance dataframe
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_scores,
        }
    )
    
    #Sort features by importance
    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)
    
    #Save complete importance table
    output_path = "reports/random_forest_feature_importance.csv"
    importance_df.to_csv(output_path,index=False)
    
    #Display results
    print("\n--- Random Forest Feature Importance ---")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Original features:", X.shape[1])
    print("Transformed features:", len(feature_names))
    
    print("\n--- Top 15 Important Features ---")
    
    print(
        importance_df.head(15).to_string(
            index=False,
            formatters={"importance": "{:.6f}".format},
        )
    )
    
    print("\nImportance score sum:", importance_scores.sum())
    print("Saved importance results to:", output_path)

    return importance_df

def main():
    """Run feature importance analysis."""

    analyze_feature_importance()


if __name__ == "__main__":
    main()
