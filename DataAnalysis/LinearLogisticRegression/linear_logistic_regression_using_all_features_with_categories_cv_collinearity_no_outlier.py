import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from data_reader import store_data, fields
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from data_preprocessing import df_manhattan, encoded_category_columns
import joblib
import statsmodels.api as sm
import os

# Features to use
features = ["normalized_average_income_data", "has_subway_access", "normalized_complaints_within_radius", "normalized_precinct_distance",
    "normalized_park_distance", "normalized_subway_distance", "normalized_rat_sighting_distance", "normalized_closest_rat_sighting_count"] + encoded_category_columns.tolist()

y_success = df_manhattan["success"]  # Binary target for classification (1 = success, 0 = not)

# Target Variables
y_bayesian_score = df_manhattan["bayesian_score"]  # Continuous target for regression
y_imdb_score = df_manhattan["imdb_score"]  # Continuous target for regression

# Select features
X = df_manhattan[features]

# Remove high VIF features
# X = X.drop(columns=high_vif_features, errors='ignore')

print(f"Reduced feature set: {X.columns.tolist()}")
#  Outlier and Leverage Analysis (Logistic Regression - success) 
print("\n Outlier and Leverage Analysis (Logistic Regression - success) ")

X_sm_logistic = sm.add_constant(X)  # Add constant for statsmodels
model_sm_logistic = sm.Logit(y_success, X_sm_logistic).fit()

# Influence measures
influence_logistic = model_sm_logistic.get_influence()

# Leverage (Hat values)
leverage_logistic = influence_logistic.hat_matrix_diag

# Standardized residuals (Deviance residuals are commonly used for logistic regression)
residuals_logistic = influence_logistic.resid_dev

# Cook's Distance
cooks_distance_logistic = influence_logistic.cooks_distance[0]

# Identify Outliers and High Leverage Points
outlier_threshold_logistic = 3  # Standard deviations
high_leverage_threshold_logistic = 2 * (len(X.columns) + 1) / len(X)

outliers_logistic = np.abs(residuals_logistic) > outlier_threshold_logistic
high_leverage_logistic = leverage_logistic > high_leverage_threshold_logistic
cooks_outliers_logistic = cooks_distance_logistic > 4 / len(X)

print(f"Number of Logistic Outliers: {np.sum(outliers_logistic)}")
print(f"Number of Logistic High Leverage Points: {np.sum(high_leverage_logistic)}")
print(f"Number of Logistic Cook's Distance Outliers: {np.sum(cooks_outliers_logistic)}")

# Remove Outliers and High Leverage Points
outlier_indices_logistic = outliers_logistic | high_leverage_logistic | cooks_outliers_logistic
X_no_outliers_logistic = X[~outlier_indices_logistic]
y_success_no_outliers_logistic = y_success[~outlier_indices_logistic]
y_bayesian_score_no_outliers_logistic = y_bayesian_score[~outlier_indices_logistic]
y_imdb_score_no_outliers_logistic = y_imdb_score[~outlier_indices_logistic]


# Run Logistic Regression (Binary Target - Store Success)


# Logistic Regression with Cross-Validation (Reduced Features, No Outliers)
logistic_model_success = LogisticRegression(max_iter=500)

# Cross-validation for Logistic Regression
accuracy_scores = cross_val_score(logistic_model_success, X_no_outliers_logistic, y_success_no_outliers_logistic, cv=kf, scoring='accuracy')

print(f"\nLogistic Regression Cross-Validation Accuracy (Reduced Features, No Outliers): {accuracy_scores.mean():.4f} (± {accuracy_scores.std():.4f})")

# Logistic Regression with L1 Regularization and Cross Validation
logistic_model_l1 = LogisticRegression(max_iter=500, penalty="l1", solver="liblinear")
accuracy_l1_scores = cross_val_score(logistic_model_l1, X_no_outliers_logistic, y_success_no_outliers_logistic, cv=kf, scoring='accuracy')

print(f"\nLogistic Regression L1 Cross-Validation Accuracy (Reduced Features, No Outliers): {accuracy_l1_scores.mean():.4f} (± {accuracy_l1_scores.std():.4f})")

# Logistic Regression with L2 Regularization and Cross Validation
logistic_model_l2 = LogisticRegression(max_iter=500, penalty="l2")
accuracy_l2_scores = cross_val_score(logistic_model_l2, X_no_outliers_logistic, y_success_no_outliers_logistic, cv=kf, scoring='accuracy')

print(f"\nLogistic Regression L2 Cross-Validation Accuracy (Reduced Features, No Outliers): {accuracy_l2_scores.mean():.4f} (± {accuracy_l2_scores.std():.4f})")

# Save the logistic regression models
joblib.dump(logistic_model_success, 'logistic_model_success_reduced_no_outliers_logistic.pkl')
joblib.dump(logistic_model_l1, 'logistic_model_l1_reduced_no_outliers_logistic.pkl')
joblib.dump(logistic_model_l2, 'logistic_model_l2_reduced_no_outliers_logistic.pkl')

print("\nLogistic regression models (reduced features, no outliers) saved successfully.")

# Write model performance evaluation data to file
report_filename = os.path.splitext(__file__)[0] + '_report.txt'

with open(report_filename, 'w') as report_file:
    report_file.write("Logistic Regression Cross-Validation Accuracy (Reduced Features):\n")
    report_file.write(f"{accuracy_scores.mean():.4f} (± {accuracy_scores.std():.4f})\n\n")
    
    report_file.write("Logistic Regression L1 Cross-Validation Accuracy (Reduced Features):\n")
    report_file.write(f"{accuracy_l1_scores.mean():.4f} (± {accuracy_l1_scores.std():.4f})\n\n")
    
    report_file.write("Logistic Regression L2 Cross-Validation Accuracy (Reduced Features):\n")
    report_file.write(f"{accuracy_l2_scores.mean():.4f} (± {accuracy_l2_scores.std():.4f})\n\n")
