from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    /"WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

def load_dataset() -> pd.DataFrame:
    """Load the employee attrition dataset from the raw data directory."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}"
        )

    return pd.read_csv(DATASET_PATH)
