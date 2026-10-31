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
│
├── data/
├── notebooks/
│
├── src/
│   ├── data_loader.py
│   ├── inspection/
│   ├── preprocessing/    
│   ├── models/           
│   ├── pipeline/         
│   └── api/              
│
├── models/
├── reports/
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore


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