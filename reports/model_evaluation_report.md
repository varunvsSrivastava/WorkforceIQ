# WorkforceIQ – Model Evaluation Report

## 1. Objective

Evaluate multiple machine learning classifiers for employee attrition prediction, compare their predictive performance, optimize classification thresholds, and identify a suitable candidate model for attrition detection.

## 2. Dataset and Experimental Setup

- **Dataset:** IBM HR Analytics Employee Attrition
- **Total samples:** 1,470
- **Attrition cases:** 237 (16.12%)
- **Non-attrition cases:** 1,233 (83.88%)
- **Features retained:** 30 predictors
- **Target:** Attrition (No = 0, Yes = 1)
- **Train-test split:** 80:20
- **Training samples:** 1,176
- **Testing samples:** 294
- **Random state:** 42
- **Split strategy:** Stratified sampling

The dataset exhibits class imbalance, with attrition cases representing approximately 16.12% of observations.

## 3. Baseline Model Evaluation

Four classifiers were evaluated using the default classification threshold of 0.50.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 86.05% | 61.54% | 34.04% | 43.84% | 0.8115 | 0.5832 |
| Decision Tree | 84.01% | 50.00% | 19.15% | 27.69% | 0.6629 | 0.2796 |
| Random Forest | 83.33% | 41.67% | 10.64% | 16.95% | 0.7839 | 0.4345 |
| Gradient Boosting | 85.71% | 66.67% | 21.28% | 32.26% | 0.8060 | 0.4757 |

### Baseline Observations

- Logistic Regression demonstrated the strongest overall baseline performance.
- Gradient Boosting achieved the highest precision.
- Random Forest exhibited very low attrition recall at the default threshold.
- Decision Tree showed the weakest discrimination performance.

## 4. Cross-Validation Evaluation

Stratified 5-fold cross-validation was conducted using ROC-AUC and F1-score.

| Model | ROC-AUC Mean | ROC-AUC Std | F1 Mean | F1 Std |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8315 | 0.0235 | 0.5143 | 0.0516 |
| Decision Tree | 0.6957 | 0.0520 | 0.3624 | 0.0560 |
| Random Forest | 0.8041 | 0.0176 | 0.3222 | 0.0425 |
| Gradient Boosting | 0.8096 | 0.0152 | 0.4063 | 0.0540 |

### Cross-Validation Observations

- Logistic Regression achieved the highest mean ROC-AUC and F1-score.
- Gradient Boosting exhibited the lowest ROC-AUC variability.
- Decision Tree demonstrated the weakest discrimination and highest variability.
- Random Forest maintained competitive ROC-AUC but relatively low F1-score at the default threshold.

**Methodological note:** Baseline cross-validation was performed on the complete dataset and is treated as exploratory. Subsequent threshold optimization used training-only out-of-fold predictions, while the held-out test set was reserved for final evaluation.

## 5. Hyperparameter Tuning

GridSearchCV with Stratified 5-fold cross-validation was used to optimize model hyperparameters using ROC-AUC.

| Model | Baseline CV ROC-AUC | Tuned CV ROC-AUC | Tuned Test ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.8315 | 0.8397 | 0.8115 |
| Decision Tree | 0.6957 | 0.7096 | 0.6980 |
| Random Forest | 0.8041 | 0.8221 | 0.8046 |
| Gradient Boosting | 0.8096 | 0.8349 | 0.8073 |

### Best Hyperparameters

**Logistic Regression**
- C = 1
- class_weight = None
- solver = lbfgs

**Decision Tree**
- max_depth = 3
- min_samples_leaf = 4
- min_samples_split = 2
- class_weight = balanced

**Random Forest**
- n_estimators = 200
- max_depth = None
- max_features = log2
- min_samples_split = 5
- min_samples_leaf = 2
- class_weight = None

**Gradient Boosting**
- n_estimators = 200
- learning_rate = 0.1
- max_depth = 2
- min_samples_split = 2
- min_samples_leaf = 2
- subsample = 0.8

### Tuning Observations

- Logistic Regression remained a strong baseline with limited improvement after tuning.
- Random Forest achieved improved cross-validation and test ROC-AUC.
- Gradient Boosting achieved the highest tuned cross-validation ROC-AUC.
- Decision Tree showed modest improvement but remained the weakest model.

## 6. Out-of-Fold Threshold Optimization

Classification thresholds were optimized using training-only out-of-fold predicted probabilities.

- **Method:** Stratified 5-fold cross-validation
- **Samples:** 1,176 training observations
- **Optimization objective:** Maximum F1-score
- **Test data:** Not used for threshold selection

| Model | Optimized Threshold | OOF Precision | OOF Recall | OOF F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.3678 | 64.97% | 60.53% | 62.67% |
| Decision Tree | 0.2051 | 43.90% | 47.37% | 45.57% |
| Random Forest | 0.2400 | 43.80% | 63.16% | 51.72% |
| Gradient Boosting | 0.2257 | 50.23% | 57.89% | 53.79% |

Threshold optimization improved attrition detection by increasing recall compared with the default threshold.

## 7. Held-Out Test Evaluation

Each tuned model was fitted on the training set and evaluated on the untouched test set using both the default and optimized thresholds.

| Model | Threshold | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.50 | 86.05% | 61.54% | 34.04% | 43.84% |
| Logistic Regression | 0.3678 | 85.37% | 55.26% | 44.68% | 49.41% |
| Decision Tree | 0.50 | 72.79% | 32.26% | 63.83% | 42.86% |
| Decision Tree | 0.2051 | 15.99% | 15.75% | 97.87% | 27.14% |
| Random Forest | 0.50 | 84.35% | 57.14% | 8.51% | 14.81% |
| Random Forest | 0.2400 | 81.63% | 44.62% | 61.70% | 51.79% |
| Gradient Boosting | 0.50 | 87.07% | 80.00% | 25.53% | 38.71% |
| Gradient Boosting | 0.2257 | 81.29% | 43.33% | 55.32% | 48.60% |

ROC-AUC and PR-AUC remain unchanged when only the classification threshold is modified.

### Threshold Optimization Observations

- Logistic Regression improved recall and F1-score with a moderate reduction in precision.
- Decision Tree's optimized threshold produced excessive positive predictions on the test set.
- Random Forest demonstrated substantial improvement in recall and F1-score.
- Gradient Boosting achieved improved recall and F1-score after threshold adjustment.

## 8. Final Candidate Model

### Random Forest with Threshold 0.2400

The Random Forest classifier with an optimized threshold of 0.2400 is selected as the **provisional candidate model** for recall-oriented employee attrition detection.

| Metric | Default Threshold | Optimized Threshold |
|---|---:|---:|
| Accuracy | 84.35% | 81.63% |
| Precision | 57.14% | 44.62% |
| Recall | 8.51% | 61.70% |
| F1-score | 14.81% | 51.79% |

### Rationale

- Test recall increased from 8.51% to 61.70%.
- Test F1-score increased from 14.81% to 51.79%.
- The optimized threshold identified 29 attrition cases compared with 4 at the default threshold.
- Test F1-score (0.5179) closely matched training OOF F1-score (0.5172).
- Test recall (61.70%) was reasonably consistent with training OOF recall (63.16%).

Logistic Regression remains a useful alternative when precision and model simplicity are prioritized.

## 9. Limitations

- The dataset contains class imbalance.
- Evaluation is based on a single stratified train-test split.
- Baseline cross-validation was performed on the full dataset and should be treated as exploratory.
- Threshold selection maximized F1-score rather than an explicit HR business-cost objective.
- Threshold performance may vary across different samples and organizations.
- Predictive associations do not establish causality.
- The model is not a validated production HR decision system.

## 10. Conclusion

WorkforceIQ evaluated Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting for employee attrition prediction.

Hyperparameter tuning and threshold optimization improved the detection performance of selected models. Random Forest with threshold 0.2400 demonstrated the strongest balance of recall and F1-score on the held-out test set among the evaluated threshold-adjusted candidates.

The selected model is a provisional candidate for further validation, explainability analysis, and business-oriented threshold calibration before deployment.

---

**Project:** WorkforceIQ – Employee Attrition Prediction & Analytics  
**Phase:** Phase 5 – Model Development & Evaluation  
**Evaluation Strategy:** Baseline comparison, cross-validation, hyperparameter tuning, OOF threshold optimization, and held-out test evaluation