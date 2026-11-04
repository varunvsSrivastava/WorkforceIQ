from sklearn.pipeline import Pipeline
from src.preprocessing.build_preprocessor import (
    build_preprocessor,
)

def build_pipeline(model):
    """Build a reusable preprocessing and model pipeline."""

    pipeline = Pipeline(
        steps = [
            ("preprocessor", build_preprocessor()),
            ("classifier", model)
        ]
    )

    return pipeline
