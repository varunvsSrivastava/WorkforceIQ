"""Train Random Forest Classifier."""

from src.data_loader import load_dataset
from src.preprocessing.prepare_features import prepare_features
from src.preprocessing.split_data import split_dataset
from src.models.random_forest import build_random_forest
from src.evaluation.metrics import (
    evaluate_classification,
    print_classification_metrics,
)

def main():
    #Load and prepare dataset
    df=load_dataset()
    X,y=prepare_features(df)
    
    #Create train-test split
    X_train,X_test,y_train,y_test=split_dataset(X,y)
    
    #Build and train pipeline
    pipeline = build_random_forest()
    pipeline.fit(X_train,y_train)
    
    #Generate predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1]
    
    #Evaluate model
    metrics = evaluate_classification(y_test,y_pred,y_proba)
    print_classification_metrics(metrics)
    
    #Training and Testing accuracy
    train_accuracy = pipeline.score(X_train,y_train)
    test_accuracy = pipeline.score(X_test,y_test)
    
    print("\n--- Random Forest Results ---")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy: {test_accuracy:.4f}")
    
if __name__=="__main__":
    main()