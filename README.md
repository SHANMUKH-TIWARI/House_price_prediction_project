# House_price_prediction_project
Predict residential house prices using machine learning on Ames, Iowa data. Cleaned and preprocessed features, handled missing values, applied one-hot encoding, and trained a Random Forest model to generate accurate, scalable price predictions for unseen properties.

Project Overview

Using a dataset with over 80 features, the project demonstrates end-to-end ML workflow:

Data Exploration: Checked data quality, visualized target distribution, and identified missing values.

Preprocessing: Imputed missing categorical and numerical values, applied one-hot encoding, and transformed skewed target variable.

Modeling: Trained a Random Forest Regressor with cross-validation to predict log-transformed sale prices.

Prediction: Converted predictions back to original scale and generated submission-ready output.

Key Features

Clean handling of missing data

Feature engineering based on domain knowledge (e.g., neighborhood medians for LotFrontage)

One-hot encoding for categorical variables

Random Forest model with robust cross-validation

Log transformation of target to reduce skewness and improve accuracy

Results

Cross-validated RMSE score provides a reliable estimate of model performance.

The pipeline is modular and reproducible, allowing easy extension with other models like XGBoost or LightGBM.

Dataset

Publicly available dataset from Kaggle
.

Contains 1,460 training and 1,459 test samples with 80+ features.

How to Run

Clone the repository.

Place the dataset in a Dataset folder (train.csv, test.csv, sample_submission.csv).

Run House_Price_Predictions.py to generate predictions.
