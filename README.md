# WorkforceIQ
## Employee Attrition Prediction and Analytics

WorkforceIQ is a machine learning project designed to predict employee attrition and analyze factors associated with employees leaving an organization.

### Objectives
- Build and compare multiple classification models.
- Identify factors associated with employee attrition.
- Evaluate model performance using appropriate classification metrics.
- Develop a reusable prediction pipeline and an interactive analytics dashboard.

### Planned Technology Stack
- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- MLflow
- FastAPI
- Streamlit
- pytest
- Docker


## Project Structure

    WorkforceIQ/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── notebooks/
    ├── src/
    │   ├── __init__.py
    │   ├── data_loader.py
    │   └── inspection/
    │       ├── __init__.py
    │       ├── inspect_dataset.py
    │       ├── inspect_target.py
    │       ├── inspect_data_quality.py
    │       ├── inspect_features.py
    │       ├── inspect_statistics.py
    │       ├── inspect_categories.py
    │       ├── inspect_variance.py
    │       ├── inspect_identifiers.py
    │       └── generate_dataset_report.py
    ├── models/
    ├── reports/
    │   └── dataset_profile.txt
    ├── tests/
    ├── .gitignore
    ├── requirements.txt
    └── README.md


### Planned Architecture

The following components will be introduced in upcoming development phases:

```text
src/
├── preprocessing/    # Phase 4
├── models/           # Phase 5
├── pipeline/         # Phase 8
└── api/              # Phase 8
```



## Current Progress

- Initial project setup completed.
- IBM HR Analytics dataset loaded and inspected.
- Dataset profile report generated.
- Inspection utilities organized into a reusable Python package.

## Running Dataset Inspection

Run individual inspection modules from the project root:

```bash
python -m src.inspection.inspect_dataset
python -m src.inspection.inspect_target
python -m src.inspection.inspect_data_quality
```

Generate the dataset profile report:

```bash
python -m src.inspection.generate_dataset_report
```


### Planned Models
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

### Project Status
Initial project setup completed. Dataset exploration, preprocessing, model development, evaluation, and application development are planned.

### Dataset
The project will use a suitable public IBM HR Analytics Employee Attrition dataset. The dataset source and applicable usage terms will be documented before use.

### Disclaimer
This project is intended for educational and analytical purposes. Predictions should not be used as the sole basis for employment decisions.