from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing.prepare_features import (
    NUMERICAL_FEATURES,
    NOMINAL_FEATURES,
    ORDINAL_FEATURES
)

def build_preprocessor():
    """Build a reusable preprocessing transformer."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES
            ),
            (
                "nom",
                OneHotEncoder(handle_unknown="ignore"),
                NOMINAL_FEATURES
            ),
            (
                "ord",
                "passthrough",
                ORDINAL_FEATURES
            )
        ]
    )

    return preprocessor
