"""Run cross-validation for baseline classifier."""

from src.data_loader import load_dataset
from src.preprocessing.prepare_features import prepare_features
from src.models.logistic_regression import build_logistic_regression
from src.models.decision_tree import build_decision_tree
from src.models.random_forest import build_random_forest
from src.models.gradient_boosting import build_gradient_boosting
from src.evaluation.cross_validation import (
    evaluate_cross_validation,
    print_cross_validation_results,
)

def main():
    #Load and prepare dataset
    df=load_dataset() 
    X,y = prepare_features(df)
    
    #Define baseline models
    models = {
        "Logistic Regression": build_logistic_regression(),
        "Decision Tree": build_decision_tree(),
        "Random Forest": build_random_forest(),
        "Gradient Boosting": build_gradient_boosting(),
    }
    
    #Evaluate each model
    for name,pipeline in models.items():
        print(f"\n=========== {name} ============")
        
        metrics = evaluate_cross_validation(pipeline,X,y)
        print_cross_validation_results(metrics)
        
if __name__=="__main__":
    main()