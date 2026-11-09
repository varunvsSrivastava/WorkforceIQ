"""Tests for WorkforceIQ prediction classification."""

from src.api.prediction import classify_attrition


def test_prediction_below_threshold():
    assert classify_attrition(0.10, 0.24) == "No"


def test_prediction_equal_to_threshold():
    assert classify_attrition(0.24, 0.24) == "Yes"


def test_prediction_above_threshold():
    assert classify_attrition(0.40, 0.24) == "Yes"
