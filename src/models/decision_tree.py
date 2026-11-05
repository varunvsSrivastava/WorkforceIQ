"""
Decision Tree model configuration.
"""

from sklearn.tree import DecisionTreeClassifier


def build_decision_tree():
    """Build a Decision Tree classifier."""

    return DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    )
