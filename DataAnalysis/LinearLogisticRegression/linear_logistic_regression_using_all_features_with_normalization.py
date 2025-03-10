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
    "normalized_park_distance", "normalized_subway_distance", "normalized_rat_sighting_distance", "normalized_closest_rat_sighting_count"]

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

X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb = train_test_split(X, y_imdb_score, test_size=0.2, random_state=42)

# Train review count prediction model
linear_model_reviews = LinearRegression()


# Run Logistic Regression (Binary Target - Store Success)

# Train-test split
X_train_success, X_test_success, y_train_success, y_test_success = train_test_split(X, df_manhattan["success"], test_size=0.2, random_state=42)

# Train model
logistic_model_success = LogisticRegression(max_iter=500)

train_and_evaluate_model(linear_model_rating, X_train, y_train_rating, X_test, y_test_rating, "linear_regression_rating")
train_and_evaluate_model(linear_model_reviews, X_train_imdb, y_train_imdb, X_test_imdb, y_test_imdb, "linear_regression_imdb")
train_and_evaluate_model(logistic_model_success, X_train_success, y_train_success, X_test_success, y_test_success, "logistic_regression_success")
