
import pandas as pd

from src.preprocessing.prepare_features import (
    DROP_COLUMNS,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    NOMINAL_FEATURES,
    TARGET_COLUMN,
    prepare_features,
)


def create_sample_dataset():
    """Create a synthetic dataset with all required features."""
    data = {}

    for feature in NUMERICAL_FEATURES:
        data[feature] = [25, 35]

    for feature in ORDINAL_FEATURES:
        data[feature] = [1, 2]

    for feature in NOMINAL_FEATURES:
        data[feature] = ["CategoryA", "CategoryB"]

    data.update({
        "EmployeeNumber": [1001, 1002],
        "EmployeeCount": [1, 1],
        "Over18": ["Y", "Y"],
        "StandardHours": [80, 80],
        "Attrition": ["No", "Yes"],
    })

    return pd.DataFrame(data)


def test_prepare_features_excludes_unnecessary_columns():
    df = create_sample_dataset()

    X, y = prepare_features(df)

    assert all(column not in X.columns for column in DROP_COLUMNS)
    assert TARGET_COLUMN not in X.columns
    assert X.shape == (2, 30)


def test_prepare_features_encodes_target_correctly():
    df = create_sample_dataset()

    X, y = prepare_features(df)

    assert y.tolist() == [0, 1]
    assert y.dtype == "int64"
    assert len(X) == len(y)


from src.preprocessing.split_data import split_dataset

def test_split_dataset_preserves_shapes():
    X = pd.DataFrame({"feature": range(100)})
    y = pd.Series([0] * 80 + [1] * 20)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    assert X_train.shape == (80, 1)
    assert X_test.shape == (20, 1)
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_split_dataset_preserves_class_distribution():
    X = pd.DataFrame({"feature": range(100)})
    y = pd.Series([0] * 80 + [1] * 20)

    _, _, y_train, y_test = split_dataset(X, y)

    assert y_train.value_counts(normalize=True)[1] == 0.2
    assert y_test.value_counts(normalize=True)[1] == 0.2



import numpy as np

from src.preprocessing.build_preprocessor import build_preprocessor


def test_preprocessor_transforms_features():
    df = create_sample_dataset()
    X, _ = prepare_features(df)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] > X.shape[1]
    assert transformed.shape[1] == len(preprocessor.get_feature_names_out())
    assert np.isfinite(transformed).all()


def test_preprocessor_handles_unseen_categories():
    df = create_sample_dataset()
    X, _ = prepare_features(df)

    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    X_unseen = X.copy()
    X_unseen.loc[X_unseen.index[0], "BusinessTravel"] = "UnseenCategory"

    transformed = preprocessor.transform(X_unseen)

    assert np.isfinite(transformed).all()



from sklearn.linear_model import LogisticRegression

from src.pipeline.build_pipeline import build_pipeline


def test_pipeline_fits_and_predicts():
    df = create_sample_dataset()
    X, y = prepare_features(df)

    pipeline = build_pipeline(
        LogisticRegression(max_iter=1000)
    )

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert predictions.shape == (2,)
    assert set(predictions).issubset({0, 1})
