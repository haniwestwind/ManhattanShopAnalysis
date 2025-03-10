import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from data_reader import store_data, fields
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from data_preprocessing import df_manhattan, encoded_category_columns
from sklearn.metrics import mean_squared_error

from linear_regression_util import train_and_evaluate_model

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
# y_rating = df_manhattan["rating"]
# y_reviews = df_manhattan["review_count"]

# Train-test split
X_train, X_test, y_train_rating, y_test_rating = train_test_split(X, y_bayesian_score, test_size=0.2, random_state=42)

# Train rating prediction model
linear_model_rating = LinearRegression()
# linear_model_rating.fit(X_train, y_train_rating)

X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb = train_test_split(X, y_imdb_score, test_size=0.2, random_state=42)

# Train review count prediction model
linear_model_reviews = LinearRegression()
# linear_model_reviews.fit(X_train_imdb, y_train_imdb)

# Print coefficients
# coefficients_rating = pd.DataFrame({"Feature": X.columns, "Coefficient": linear_model_rating.coef_})
# coefficients_imdb = pd.DataFrame({"Feature": X.columns, "Coefficient": linear_model_reviews.coef_})

# print("\nMultiple Linear Regression Coefficients (Predicting Rating):\n", coefficients_rating)
# print("\nMultiple Linear Regression Coefficients (Predicting Review Count):\n", coefficients_imdb)

# Evaluate models
# r_squared_rating = linear_model_rating.score(X_test, y_test_rating)
# r_squared_imdb = linear_model_reviews.score(X_test, y_test_imdb)
# # Calculate RMSE for rating prediction
# y_pred_rating = linear_model_rating.predict(X_test)
# rmse_rating = np.sqrt(mean_squared_error(y_test_rating, y_pred_rating))
# print(f"\nMultiple Linear Regression RMSE for Rating: {rmse_rating:.4f}")

# # Calculate RMSE for IMDB score prediction
# y_pred_imdb = linear_model_reviews.predict(X_test_imdb)
# rmse_imdb = np.sqrt(mean_squared_error(y_test_imdb, y_pred_imdb))
# print(f"\nMultiple Linear Regression RMSE for IMDB score: {rmse_imdb:.4f}")
# print(f"\nMultiple Linear Regression R-squared value for Rating: {r_squared_rating:.4f}")
# print(f"\nMultiple Linear Regression R-squared value for IMDB score: {r_squared_imdb:.4f}")


# Run Logistic Regression (Binary Target - Store Success)

# Train-test split
X_train_success, X_test_success, y_train_success, y_test_success = train_test_split(X, df_manhattan["success"], test_size=0.2, random_state=42)

# Train model
logistic_model_success = LogisticRegression(max_iter=500)
# logistic_model_success.fit(X_train_success, y_train_success)

# Print coefficients
# coefficients_success = pd.DataFrame({"Feature": X.columns, "Coefficient": logistic_model_success.coef_[0]})
# print("\nLogistic Regression Coefficients (Predicting Store Success):\n", coefficients_success)

# # Predict and evaluate
# y_pred_success = logistic_model_success.predict(X_test_success)
# accuracy_success = (y_pred_success == y_test_success).mean()
# print(f"\nLogistic Regression Model Accuracy: {accuracy_success:.4f}")


# Try regularization techniques on the logistic regression model
# L1 model
# Train model
logistic_model_success_l1 = LogisticRegression(max_iter=500, penalty="l1", solver="liblinear")
# logistic_model_success.fit(X_train_success, y_train_success)

# Print coefficients
# coefficients_success = pd.DataFrame({"Feature": X.columns, "Coefficient": logistic_model_success.coef_[0]})
# print("\nLogistic Regression with L1 Regularization Coefficients (Predicting Store Success):\n", coefficients_success)

# # Predict and evaluate
# y_pred_success = logistic_model_success.predict(X_test_success)
# accuracy_success = (y_pred_success == y_test_success).mean()
# print(f"\nLogistic Regression Model with L1 Regularization Accuracy: {accuracy_success:.4f}")

# Try regularization techniques on the logistic regression model
# L2 model
# Train model
logistic_model_success_l2 = LogisticRegression(max_iter=500, penalty="l2")
# logistic_model_success.fit(X_train_success, y_train_success)

# # Print coefficients
# coefficients_success = pd.DataFrame({"Feature": X.columns, "Coefficient": logistic_model_success.coef_[0]})
# print("\nLogistic Regression with L2 Regularization Coefficients (Predicting Store Success):\n", coefficients_success)

# # Predict and evaluate
# y_pred_success = logistic_model_success.predict(X_test_success)
# accuracy_success = (y_pred_success == y_test_success).mean()
# print(f"\nLogistic Regression Model with L2 Regularization Accuracy: {accuracy_success:.4f}")


# PCA - Feature Importance Analysis


train_and_evaluate_model(linear_model_rating, X_train, y_train_rating, X_test, y_test_rating, "linear_regression_rating")
train_and_evaluate_model(linear_model_reviews, X_train_imdb, y_train_imdb, X_test_imdb, y_test_imdb, "linear_regression_imdb")
train_and_evaluate_model(logistic_model_success, X_train_success, y_train_success, X_test_success, y_test_success, "logistic_regression_success")

train_and_evaluate_model(logistic_model_success_l1, X_train_success, y_train_success, X_test_success, y_test_success, "logistic_regression_success_l1")
train_and_evaluate_model(logistic_model_success_l2, X_train_success, y_train_success, X_test_success, y_test_success, "logistic_regression_success_l2")
