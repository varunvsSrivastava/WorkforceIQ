"""Cross-validation utilities."""

from sklearn.model_selection import StratifiedKFold, cross_validate

def evaluate_cross_validation(pipeline,X,y):
    """Evaluate a pipeline using stratified 5-fold cross-validation."""
    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
    }
    
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    
    metrics = {}
    
    for metric in scoring:
        scores = results[f"test_{metric}"]
        metrics[metric] = {
            "mean": scores.mean(),
            "std": scores.std(),
        }
        
    return metrics

def print_cross_validation_results(metrics):
    """Print cross-validation results."""
    
    print("\n--- 5-fold Cross-Validation Results. ---")
    
    for metric, values in metrics.items():
        print(
            f"{metric.upper()}: "
            f"{values['mean']:.4f} "
            f"+/- {values['std']:.4f}"
        )