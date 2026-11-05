"""Gradient Boosting model builder."""

from sklearn.ensemble import GradientBoostingClassifier


def build_gradient_boosting():
    """Build a Gradient Boosting classifier."""

    model = GradientBoostingClassifier(
        random_state=42,
    )

    return model
