"""Training and persist the final WorkforceIQ model."""

from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.data_loader import load_dataset
from src.preprocessing.prepare_features import prepare_features
from src.pipeline.build_pipeline import build_pipeline

#Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" /"workforceiq_pipeline.joblib"

#Select phase 5 classification threshold
CLASSIFICATION_THRESHOLD = 0.24

def build_final_model():
    """Build the tuned Random Forest with preprocessing."""
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        max_features="log2",
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )
    
    return build_pipeline(model)

def main():
    """Train the final model and save its deployment artifacts."""
    
    #Load complete dataset
    df = load_dataset()
    X,y = prepare_features(df)
    
    #Build and train pipeline
    pipeline = build_final_model()
    pipeline.fit(X,y)
    
    #Create deployment artifacts
    artifact = {
        "pipeline": pipeline,
        "threshold": CLASSIFICATION_THRESHOLD,
        "model_name": "Random Forest",
    }
    
    #Save artifact
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    
    print("\n--- WorkforceIQ Model Persistence ---")
    print(f"Training samples: {len(X)}")
    print(f"Model path {MODEL_PATH}")
    print(f"Classification threshold: {CLASSIFICATION_THRESHOLD}")
    print("Model saved successfully.")
    
if __name__ == "__main__":
    main()