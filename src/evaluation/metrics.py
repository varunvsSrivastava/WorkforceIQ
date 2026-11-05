"""Classification evaluation metrics."""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

def evaluate_classification(y_true,y_pred,y_proba):
    """Calculate classification metrics and confusion matrix."""
    
    metrics = {
        "accuracy": accuracy_score(y_true,y_pred),
        "precision": precision_score(y_true,y_pred,zero_division=0),
        "recall": recall_score(y_true,y_pred,zero_division=0),
        "f1_score": f1_score(y_true,y_pred,zero_division=0),
        "roc_auc": roc_auc_score(y_true,y_proba),
        "pr_auc": average_precision_score(y_true,y_proba),
        "confusion_matrix": confusion_matrix(y_true,y_pred),
    }
    
    return metrics

def print_classification_metrics(metrics):
    """Print classification evaluation results."""
    
    print("\n--- Classification Metrics ---")
    
    for metric,value in metrics.items():
        if metric=="confusion_matrix":
            print(f"\n{metric.replace('_',' ').title()}:")
            print(value)
            
        else:
            print(f"{metric.replace('_',' ').title()}: {value:.4f}")