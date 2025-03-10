import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from data_reader import store_data, fields
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import time
import os

from data_preprocessing import df_manhattan, encoded_category_columns

"""Initial Modeling with entire Businesses data and 5 variables (Proximity to parks, proximity to Subway, ZIP income,
proximity to police, rat sighting), using Multiple Linear and Logistic Regression"""

print("Manhattan stores loaded:", len(df_manhattan))
print(df_manhattan.head())

#  Load & Prepare Data 

print("Columns in df_manhattan:", df_manhattan.columns.tolist())  # Debugging


# Run Multiple Regression (Continuous Target - Store Rating & Review Count)


# Features to use
features = ["normalized_average_income_data", "has_subway_access", "normalized_complaints_within_radius", "normalized_precinct_distance",
    "normalized_park_distance", "normalized_subway_distance", "normalized_rat_sighting_distance", "normalized_closest_rat_sighting_count"] + encoded_category_columns.tolist()

y_success = df_manhattan["success"]  # Binary target for classification (1 = success, 0 = not)

# Target Variables
y_bayesian_score = df_manhattan["bayesian_score"]  # Continuous target for regression
y_imdb_score = df_manhattan["imdb_score"]  # Continuous target for regression

# Select features
X = df_manhattan[features]

# Linear Regression with Cross-Validation
linear_model_rating = LinearRegression()
linear_model_reviews = LinearRegression()

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

# Rating Cross-Validation
start_rating = time.time()
rating_scores = cross_val_score(linear_model_rating, X, y_bayesian_score, cv=kf, scoring='neg_mean_squared_error')
end_rating = time.time()
elapsed_rating = end_rating - start_rating
rmse_rating_cv = np.sqrt(-rating_scores)  # Convert negative MSE to RMSE

print(f"\nLinear Regression Rating Cross-Validation RMSE: {rmse_rating_cv.mean():.4f} (± {rmse_rating_cv.std():.4f})")

# IMDB Score Cross-Validation
start_imdb = time.time()
imdb_scores = cross_val_score(linear_model_reviews, X, y_imdb_score, cv=kf, scoring='neg_mean_squared_error')
end_imdb = time.time()
elapsed_imdb = end_imdb - start_imdb
rmse_imdb_cv = np.sqrt(-imdb_scores)

print(f"\nLinear Regression IMDB Score Cross-Validation RMSE: {rmse_imdb_cv.mean():.4f} (± {rmse_imdb_cv.std():.4f})")

# Save the linear regression models
joblib.dump(linear_model_rating, 'linear_model_rating.pkl')
joblib.dump(linear_model_reviews, 'linear_model_reviews.pkl')

print("\nLinear regression models saved successfully.")


# Run Logistic Regression (Binary Target - Store Success)


# Logistic Regression with Cross-Validation
logistic_model_success = LogisticRegression(max_iter=500)

# Cross-validation for Logistic Regression
start_logistic = time.time()
accuracy_scores = cross_val_score(logistic_model_success, X, y_success, cv=kf, scoring='accuracy')
end_logistic = time.time()
elapsed_logistic = end_logistic - start_logistic

print(f"\nLogistic Regression Cross-Validation Accuracy: {accuracy_scores.mean():.4f} (± {accuracy_scores.std():.4f})")

# Logistic Regression with L1 Regularization and Cross Validation
start_l1 = time.time()
logistic_model_l1 = LogisticRegression(max_iter=500, penalty="l1", solver="liblinear")
accuracy_l1_scores = cross_val_score(logistic_model_l1, X, y_success, cv=kf, scoring='accuracy')
end_l1 = time.time()
elapsed_l1 = end_l1 - start_l1

print(f"\nLogistic Regression L1 Cross-Validation Accuracy: {accuracy_l1_scores.mean():.4f} (± {accuracy_l1_scores.std():.4f})")

# Logistic Regression with L2 Regularization and Cross Validation
start_l2 = time.time()
logistic_model_l2 = LogisticRegression(max_iter=500, penalty="l2")
accuracy_l2_scores = cross_val_score(logistic_model_l2, X, y_success, cv=kf, scoring='accuracy')
end_l2 = time.time()
elapsed_l2 = end_l2 - start_l2

print(f"\nLogistic Regression L2 Cross-Validation Accuracy: {accuracy_l2_scores.mean():.4f} (± {accuracy_l2_scores.std():.4f})")

# Save the logistic regression models
joblib.dump(logistic_model_success, 'logistic_model_success.pkl')
joblib.dump(logistic_model_l1, 'logistic_model_l1.pkl')
joblib.dump(logistic_model_l2, 'logistic_model_l2.pkl')

print("\nLogistic regression models saved successfully.")


# Final Summary and Save Results


print("\nSummary of Analysis:")

# Save results to file
script_name = os.path.splitext(os.path.basename(__file__))[0] + ".txt"
with open(script_name, "w") as f:
    f.write(f"Linear Regression Rating Cross-Validation RMSE: {rmse_rating_cv.mean():.4f} (± {rmse_rating_cv.std():.4f})\n")
    f.write(f"Linear Regression Rating Cross-Validation Time: {elapsed_rating:.4f} seconds\n")
    f.write(f"Linear Regression IMDB Score Cross-Validation RMSE: {rmse_imdb_cv.mean():.4f} (± {rmse_imdb_cv.std():.4f})\n")
    f.write(f"Linear Regression IMDB Score Cross-Validation Time: {elapsed_imdb:.4f} seconds\n")
    f.write(f"Logistic Regression Cross-Validation Accuracy: {accuracy_scores.mean():.4f} (± {accuracy_scores.std():.4f})\n")
    f.write(f"Logistic Regression Cross-Validation Time: {elapsed_logistic:.4f} seconds\n")
    f.write(f"Logistic Regression L1 Cross-Validation Accuracy: {accuracy_l1_scores.mean():.4f} (± {accuracy_l1_scores.std():.4f})\n")
    f.write(f"Logistic Regression L1 Cross-Validation Time: {elapsed_l1:.4f} seconds\n")
    f.write(f"Logistic Regression L2 Cross-Validation Accuracy: {accuracy_l2_scores.mean():.4f} (± {accuracy_l2_scores.std():.4f})\n")
    f.write(f"Logistic Regression L2 Cross-Validation Time: {elapsed_l2:.4f} seconds\n")

print(f"Results written to {script_name}")