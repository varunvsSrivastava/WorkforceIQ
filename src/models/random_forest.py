"""Random Forest model builder."""

from sklearn.ensemble import RandomForestClassifier


def build_random_forest():
    """Build a Random Forest classifier."""

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    return model
