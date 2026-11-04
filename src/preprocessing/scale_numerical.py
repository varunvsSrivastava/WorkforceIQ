import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.preprocessing.prepare_features import (NUMERICAL_FEATURES,)

def scale_numerical_features(X_train,X_test):
    """Scale numerical features using training data only."""

    scaler = StandardScaler()

    #Fit scaler only on training numerical features
    scaler.fit(X_train[NUMERICAL_FEATURES])

    #Transform numerical features
    train_scaled = pd.DataFrame(
        scaler.transform(X_train[NUMERICAL_FEATURES]),
        columns=NUMERICAL_FEATURES,
        index=X_train.index
    )

    test_scaled = pd.DataFrame(
        scaler.transform(X_test[NUMERICAL_FEATURES]),
        columns=NUMERICAL_FEATURES,
        index=X_test.index
    )

    #Replace original numerical features with scaled values
    X_train_transformed = X_train.copy()
    X_test_transformed = X_test.copy()

    X_train_transformed[NUMERICAL_FEATURES] = train_scaled
    X_test_transformed[NUMERICAL_FEATURES] = test_scaled

    return X_train_transformed, X_test_transformed, scaler
