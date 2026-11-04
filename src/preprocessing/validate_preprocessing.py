import numpy as np
from src.preprocessing.prepare_features import ORDINAL_FEATURES

def validate_preprocessing(
    X_train_transformed,
    X_test_transformed,
    y_train,
    y_test,
    feature_names,
    X_train_original
):
    """Validate transformed datasets before model training."""

    # Validate dimensions
    assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
    assert X_train_transformed.shape[0] == len(y_train)
    assert X_test_transformed.shape[0] == len(y_test)

    # Validate feature names
    assert len(feature_names) == X_train_transformed.shape[1]
    assert len(set(feature_names)) == len(feature_names)

    # Validate numerical values
    assert np.isfinite(X_train_transformed).all()
    assert np.isfinite(X_test_transformed).all()

    # Validate ordinal feature preservation
    for feature in ORDINAL_FEATURES:
        feature_index = list(feature_names).index(f"ord__{feature}")

        assert np.array_equal(
            X_train_transformed[:, feature_index],
            X_train_original[feature].to_numpy()
        )

    # Validate one-hot encoded features
    nominal_indices = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("nom__")
    ]

    nominal_values = X_train_transformed[:, nominal_indices]

    assert np.isin(nominal_values, [0, 1]).all()

    return True
