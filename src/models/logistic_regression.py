"""Logistic Regression model builder."""

from sklearn.linear_model import LogisticRegression


def build_logistic_regression():
    """Build a Logistic Regression classifier."""

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    return model
