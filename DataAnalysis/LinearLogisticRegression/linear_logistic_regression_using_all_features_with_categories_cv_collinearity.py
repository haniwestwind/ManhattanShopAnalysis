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
import os


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

#  Collinearity Analysis 
print("\n Collinearity Analysis ")

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)

# Visualize Correlation Matrix
correlation_matrix = X.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
plt.title("Correlation Matrix of Features")
plt.savefig('correlation_matrix.png')

#  Feature Selection based on VIF 
# Threshold for VIF (e.g., 10)
vif_threshold = 10

# Identify features with high VIF
high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]["feature"].tolist()

print(f"\nFeatures with high VIF (> {vif_threshold}): {high_vif_features}")

# Remove high VIF features
X_reduced = X.drop(columns=high_vif_features, errors='ignore')

print(f"Reduced feature set: {X_reduced.columns.tolist()}")

# Linear Regression with Cross-Validation (Reduced Features)
linear_model_rating = LinearRegression()
linear_model_reviews = LinearRegression()

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

# Rating Cross-Validation
rating_scores = cross_val_score(linear_model_rating, X_reduced, y_bayesian_score, cv=kf, scoring='neg_mean_squared_error')
rmse_rating_cv = np.sqrt(-rating_scores)  # Convert negative MSE to RMSE

print(f"\nLinear Regression Rating Cross-Validation RMSE (Reduced Features): {rmse_rating_cv.mean():.4f} (± {rmse_rating_cv.std():.4f})")

# IMDB Score Cross-Validation
imdb_scores = cross_val_score(linear_model_reviews, X_reduced, y_imdb_score, cv=kf, scoring='neg_mean_squared_error')
rmse_imdb_cv = np.sqrt(-imdb_scores)

print(f"\nLinear Regression IMDB Score Cross-Validation RMSE (Reduced Features): {rmse_imdb_cv.mean():.4f} (± {rmse_imdb_cv.std():.4f})")

# Save the linear regression models
joblib.dump(linear_model_rating, 'linear_model_rating_reduced.pkl')
joblib.dump(linear_model_reviews, 'linear_model_reviews_reduced.pkl')

print("\nLinear regression models (reduced features) saved successfully.")


# Run Logistic Regression (Binary Target - Store Success)


# Logistic Regression with Cross-Validation (Reduced Features)
logistic_model_success = LogisticRegression(max_iter=500)

# Cross-validation for Logistic Regression
accuracy_scores = cross_val_score(logistic_model_success, X_reduced, y_success, cv=kf, scoring='accuracy')

print(f"\nLogistic Regression Cross-Validation Accuracy (Reduced Features): {accuracy_scores.mean():.4f} (± {accuracy_scores.std():.4f})")

# Logistic Regression with L1 Regularization and Cross Validation
logistic_model_l1 = LogisticRegression(max_iter=500, penalty="l1", solver="liblinear")
accuracy_l1_scores = cross_val_score(logistic_model_l1, X_reduced, y_success, cv=kf, scoring='accuracy')

print(f"\nLogistic Regression L1 Cross-Validation Accuracy (Reduced Features): {accuracy_l1_scores.mean():.4f} (± {accuracy_l1_scores.std():.4f})")

# Logistic Regression with L2 Regularization and Cross Validation
logistic_model_l2 = LogisticRegression(max_iter=500, penalty="l2")
accuracy_l2_scores = cross_val_score(logistic_model_l2, X_reduced, y_success, cv=kf, scoring='accuracy')

print(f"\nLogistic Regression L2 Cross-Validation Accuracy (Reduced Features): {accuracy_l2_scores.mean():.4f} (± {accuracy_l2_scores.std():.4f})")

# Save the logistic regression models
joblib.dump(logistic_model_success, 'logistic_model_success_reduced.pkl')
joblib.dump(logistic_model_l1, 'logistic_model_l1_reduced.pkl')
joblib.dump(logistic_model_l2, 'logistic_model_l2_reduced.pkl')
# Write model performance evaluation data to file
report_filename = os.path.splitext(__file__)[0] + '_report.txt'

with open(report_filename, 'w') as report_file:
    report_file.write("Linear Regression Rating Cross-Validation RMSE (Reduced Features):\n")
    report_file.write(f"{rmse_rating_cv.mean():.4f} (± {rmse_rating_cv.std():.4f})\n\n")
    
    report_file.write("Linear Regression IMDB Score Cross-Validation RMSE (Reduced Features):\n")
    report_file.write(f"{rmse_imdb_cv.mean():.4f} (± {rmse_imdb_cv.std():.4f})\n\n")
    
    report_file.write("Logistic Regression Cross-Validation Accuracy (Reduced Features):\n")
    report_file.write(f"{accuracy_scores.mean():.4f} (± {accuracy_scores.std():.4f})\n\n")
    
    report_file.write("Logistic Regression L1 Cross-Validation Accuracy (Reduced Features):\n")
    report_file.write(f"{accuracy_l1_scores.mean():.4f} (± {accuracy_l1_scores.std():.4f})\n\n")
    
    report_file.write("Logistic Regression L2 Cross-Validation Accuracy (Reduced Features):\n")
    report_file.write(f"{accuracy_l2_scores.mean():.4f} (± {accuracy_l2_scores.std():.4f})\n\n")

print("\nModel performance evaluation data written to file successfully.")
print("\nLogistic regression models (reduced features) saved successfully.")
# Save the linear regression models
# joblib.dump(linear_model_rating, 'linear_model_rating.pkl')
# joblib.dump(linear_model_reviews, 'linear_model_reviews.pkl')

# print("\nLinear regression models saved successfully.")

# Final Summary

