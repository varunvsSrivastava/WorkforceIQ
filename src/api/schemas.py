
"""Pydantic schemas for WorkforceIQ API."""

from typing import Literal

from pydantic import BaseModel, Field


class EmployeeInput(BaseModel):
    """Validated employee features for attrition prediction."""

    # Numerical features
    Age: int = Field(ge=18, le=60)
    DailyRate: int = Field(gt=0)
    DistanceFromHome: int = Field(ge=1)
    HourlyRate: int = Field(gt=0)
    MonthlyIncome: int = Field(gt=0)
    MonthlyRate: int = Field(gt=0)
    NumCompaniesWorked: int = Field(ge=0)
    PercentSalaryHike: int = Field(ge=0, le=100)
    TotalWorkingYears: int = Field(ge=0)
    TrainingTimesLastYear: int = Field(ge=0)
    YearsAtCompany: int = Field(ge=0)
    YearsInCurrentRole: int = Field(ge=0)
    YearsSinceLastPromotion: int = Field(ge=0)
    YearsWithCurrManager: int = Field(ge=0)

    # Ordinal features
    Education: int = Field(ge=1, le=5)
    EnvironmentSatisfaction: int = Field(ge=1, le=4)
    JobInvolvement: int = Field(ge=1, le=4)
    JobLevel: int = Field(ge=1, le=5)
    JobSatisfaction: int = Field(ge=1, le=4)
    PerformanceRating: Literal[3, 4]
    RelationshipSatisfaction: int = Field(ge=1, le=4)
    StockOptionLevel: int = Field(ge=0, le=3)
    WorkLifeBalance: int = Field(ge=1, le=4)

    # Nominal categorical features
    BusinessTravel: Literal[
        "Travel_Rarely",
        "Travel_Frequently",
        "Non-Travel",
    ]
    Department: Literal[
        "Sales",
        "Research & Development",
        "Human Resources",
    ]
    EducationField: Literal[
        "Life Sciences",
        "Medical",
        "Marketing",
        "Technical Degree",
        "Other",
        "Human Resources",
    ]
    Gender: Literal["Male", "Female"]
    JobRole: Literal[
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources",
    ]
    MaritalStatus: Literal["Single", "Married", "Divorced"]
    OverTime: Literal["Yes", "No"]



class FeatureContribution(BaseModel):
    """SHAP contribution for an individual feature."""

    feature: str
    impact: float


class PredictionResponse(BaseModel):
    """Prediction response returned by WorkforceIQ API."""

    attrition_prediction: Literal["Yes", "No"]
    attrition_probability: float = Field(ge=0, le=1)
    classification_threshold: float
    model_name: str
    feature_contributions: list[FeatureContribution]

