# loan-default-prediction
Credit risk modelling project using Python, data preprocessing, feature engineering, and machine learning classification.
The goal of the project was to predict whether a customer would successfully repay a loan using customer demographics, current loan characteristics, and historical loan behaviour.
## What I Did

- Explored customer demographic, loan performance, and previous-loan datasets
- Cleaned and transformed missing, categorical, numerical, and date-based data
- Engineered behavioural features from customers' historical loan activity
- Combined multiple datasets into a model-ready analytical dataset
- Compared several classification approaches including:
  - Logistic Regression
  - K-Nearest Neighbours
  - LightGBM
  - CatBoost
- Used cross-validation and hyperparameter tuning to compare model performance
- Analysed feature importance and tested additional feature-engineering approaches
- Selected CatBoost as the final model and generated predictions on unseen test data

## Model Performance

The strongest CatBoost configuration achieved approximately **80.1% cross-validation accuracy** after hyperparameter tuning.

## Technologies

Python, Pandas, NumPy, scikit-learn, CatBoost, LightGBM, Matplotlib

## Key Skills Demonstrated

Data Cleaning · Exploratory Data Analysis · Feature Engineering · Classification · Model Evaluation · Cross-Validation · Hyperparameter Tuning · Credit Risk Analytics

## Files

- `Team Danforth Code.ipynb` – exploratory analysis, preprocessing, modelling, model comparison and prediction pipeline

## Notes

This repository is a portfolio version of an academic team project. Course materials and source datasets are not included where redistribution rights are unclear.
