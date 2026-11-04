from src.data_loader import load_dataset

#column exclude from model training
DROP_COLUMNS = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

TARGET_COLUMN = "Attrition"

TARGET_MAPPING = {
    "No": 0,
    "Yes": 1
}

#numerical features
NUMERICAL_FEATURES = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]


#ordinal categorical features
ORDINAL_FEATURES = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance"
]

#nominal categorical features
NOMINAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime"
]

def prepare_features(df):
    """Seperate model features and encode the target variable."""

    required_columns = (
        DROP_COLUMNS + [TARGET_COLUMN] + NUMERICAL_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
    )

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    #seperate target and predictor features
    X = df.drop(
        columns = DROP_COLUMNS + [TARGET_COLUMN]
    ).copy()

    #validate feature classification
    classified_features = (
        NUMERICAL_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
    )

    if set(X.columns) != set(classified_features):
        raise ValueError(
            "Feature classification does not match "
            "the available predictor columns."
        )

    #encode target variable
    y = df[TARGET_COLUMN].map(TARGET_MAPPING)

    if y.isna().any():
        invalid_values = df.loc[y.isna(), TARGET_COLUMN].unique().tolist()

        raise ValueError(
            f"Unexpected target values: {invalid_values}"
        )

    y = y.astype("int64")

    return X, y

def main():
    df = load_dataset()
    X, y = prepare_features(df)

    print("--- Feature-Target Separation ---")

    print("\nOriginal dataset shape:", df.shape)

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    print("\n--- Feature Categories ---")

    print("Numerical features:", len(NUMERICAL_FEATURES))
    print("Ordinal features:", len(ORDINAL_FEATURES))
    print("Nominal features:", len(NOMINAL_FEATURES))

    print("\nTotal predictor features:", X.shape[1])

    print("\n--- Target Distribution ---")
    print(y.value_counts().sort_index())

    print("\n--- Target Mapping ---")
    print("No  -> 0")
    print("Yes -> 1")


if __name__ == "__main__":
    main()


