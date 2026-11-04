
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.preprocessing.prepare_features import (
    NOMINAL_FEATURES,
)


def encode_nominal_features(X_train, X_test):
    """One-hot encode nominal features using training data only."""

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    encoder.fit(X_train[NOMINAL_FEATURES])

    train_encoded = pd.DataFrame(
        encoder.transform(X_train[NOMINAL_FEATURES]),
        columns=encoder.get_feature_names_out(NOMINAL_FEATURES),
        index=X_train.index
    )

    test_encoded = pd.DataFrame(
        encoder.transform(X_test[NOMINAL_FEATURES]),
        columns=encoder.get_feature_names_out(NOMINAL_FEATURES),
        index=X_test.index
    )

    X_train_transformed = (
        X_train.drop(columns=NOMINAL_FEATURES)
        .join(train_encoded)
    )

    X_test_transformed = (
        X_test.drop(columns=NOMINAL_FEATURES)
        .join(test_encoded)
    )

    return X_train_transformed, X_test_transformed, encoder
