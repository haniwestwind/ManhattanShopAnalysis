import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pymongo
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")  # Update if using Docker or remote MongoDB
db = client["yelp_data"]  
collection = db["businesses"]  

# Load data into a DataFrame
store_data = list(collection.find({}, {
    "_id": 1, "rating": 1, "review_count": 1, "bayesian_score": 1, "imdb_score": 1,
    "coordinates": 1, 
    "average_income_data": 1, "closest_precincts": 1,
    "closest_parks": 1, "closest_subways": 1,
    "has_subway_access": 1, "closest_rat_sighting_count": 1,
    "closest_rat_sighting_distance": 1
}))

df_manhattan = pd.DataFrame(store_data)
print("Columns in df_stores:", df_manhattan.columns.tolist())  # Debugging

# Ensure location data exists
# df_stores["Zipcode"] = df_stores["location"].apply(lambda x: x.get("zip_code", None))

# df_manhattan = df_stores[df_stores["Zipcode"].astype(str).str.startswith("100")].copy()

# Convert necessary fields
df_manhattan["closest_precinct_distance"] = df_manhattan["closest_precincts"].apply(
    lambda x: x[0][1] if isinstance(x, list) and len(x) > 0 else None)
df_manhattan["closest_park_distance"] = df_manhattan["closest_parks"].apply(
    lambda x: x[0]["Distance"] if isinstance(x, list) and len(x) > 0 else None)
df_manhattan["closest_subway_distance"] = df_manhattan["closest_subways"].apply(
    lambda x: x[0]["subway_distance_miles"] if isinstance(x, list) and len(x) > 0 else None)

# df_manhattan["closest_rat_sighting_count"] = df_manhattan["closest_rat_sighting_count"]
# df_manhattan["closest_rat_sighting_distance"] = df_manhattan["closest_rat_sighting_distance"]
# Extract closest school distance
df_manhattan["closest_school_distance"] = df_manhattan["closest_schools"].apply(
    lambda x: x[0]["Distance"] if isinstance(x, list) and len(x) > 0 else None)

# Extract closest restroom distance
df_manhattan["closest_restroom_distance"] = df_manhattan["closest_restrooms"].apply(
    lambda x: x[0]["Distance"] if isinstance(x, list) and len(x) > 0 else None)

df_manhattan["normalized_average_income_data"] = df_manhattan["average_income_data"] / (df_manhattan["average_income_data"].max() + 1)

df_manhattan["has_subway_access"] = df_manhattan["has_subway_access"].fillna(0).astype(int)

# Normalize complaints_within_radius with the maximum value
df_manhattan["normalized_complaints_within_radius"] = df_manhattan["complaints_within_radius"] / (df_manhattan["complaints_within_radius"].max() + 1)

# Normalize distances with the maximum value in each column
df_manhattan["normalized_precinct_distance"] = df_manhattan["closest_precinct_distance"] / (df_manhattan["closest_precinct_distance"].max() + 1)
df_manhattan["normalized_park_distance"] = df_manhattan["closest_park_distance"] / (df_manhattan["closest_park_distance"].max() + 1)
df_manhattan["normalized_subway_distance"] = df_manhattan["closest_subway_distance"] / (df_manhattan["closest_subway_distance"].max() + 1)
df_manhattan["normalized_rat_sighting_distance"] = df_manhattan["closest_rat_sighting_distance"] / (df_manhattan["closest_rat_sighting_distance"].max() + 1)
df_manhattan["normalized_closest_rat_sighting_count"] = df_manhattan["closest_rat_sighting_count"] / (df_manhattan["closest_rat_sighting_count"].max() + 1)


# Drop missing values
df_manhattan.dropna(subset=[
    "normalized_average_income_data", "has_subway_access", "normalized_complaints_within_radius", "normalized_precinct_distance",
    "normalized_park_distance", "normalized_subway_distance", "normalized_rat_sighting_distance", "normalized_closest_rat_sighting_count"
], inplace=True)

print("Manhattan stores loaded:", len(df_manhattan))
print(df_manhattan.head())

#  Load & Prepare Data 

# Features to use
features = ["normalized_average_income_data", "has_subway_access", "normalized_complaints_within_radius", "normalized_precinct_distance",
    "normalized_park_distance", "normalized_subway_distance", "normalized_rat_sighting_distance", "normalized_closest_rat_sighting_count"]

# Target Variables
y_bayesian_score = df_manhattan["bayesian_score"]  # Continuous target for regression
y_imdb_score = df_manhattan["imdb_score"]  # Continuous target for regression

df_manhattan["success"] = (df_manhattan["bayesian_score"] >= df_manhattan["bayesian_score"].mean()).astype(int)
# y_rating = df_manhattan["rating"]  # Continuous target for regression
# df_manhattan["Success"] = (df_manhattan["rating"] >= 4.0).astype(int)
y_success = df_manhattan["success"]  # Binary target for classification (1 = success, 0 = not)

# Drop any missing values
# df_manhattan = df_manhattan.dropna(subset=features + ["rating", "Success"])
df_manhattan = df_manhattan.dropna(subset=features + ["bayesian_score", "success"])

X = df_manhattan[features]

# Split Data
X_train, X_test, y_train_rating, y_test_rating = train_test_split(X, y_bayesian_score, test_size=0.2, random_state=42)


# XGBoost Regression - Predicting Store Rating


xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
xgb_reg.fit(X_train, y_train_rating)
y_pred_rating = xgb_reg.predict(X_test)

# Evaluate Regression
mse = mean_squared_error(y_test_rating, y_pred_rating)
print(f"XGBoost Regression - Store Rating Prediction MSE: {mse:.4f}")
# Calculate R-squared for regression
r_squared = xgb_reg.score(X_test, y_test_rating)
print(f"XGBoost Regression - Store Rating Prediction R-squared: {r_squared:.4f}")

# Calculate additional metrics for classification


# XGBoost Classification - Predicting Store Success

X_train_cls, X_test_cls, y_train_success, y_test_success = train_test_split(X, y_success, test_size=0.2, random_state=42)

xgb_clf = xgb.XGBClassifier(objective="binary:logistic", n_estimators=100, learning_rate=0.1)
xgb_clf.fit(X_train_cls, y_train_success)
y_pred_success = xgb_clf.predict(X_test_cls)

# Evaluate Classification
accuracy = accuracy_score(y_test_success, y_pred_success)
print(f"XGBoost Classification - Store Success Accuracy: {accuracy:.4f}")


conf_matrix = confusion_matrix(y_test_success, y_pred_success)
class_report = classification_report(y_test_success, y_pred_success)

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Plot confusion matrix

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Store Success Prediction")
plt.show()

# Feature Importance


importance_df = pd.DataFrame({"Feature": X.columns, "Importance": xgb_clf.feature_importances_})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
print("\nFeature Importance (Most Important Factors for Store Success):")
print(importance_df)

# Visualizing Feature Importance
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance for Store Success")
plt.gca().invert_yaxis()
plt.show()
