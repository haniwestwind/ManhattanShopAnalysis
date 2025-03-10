import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

from data_preprocessing import df_manhattan, encoded_category_columns

"""SVM using all features and categories"""

print("Manhattan stores loaded:", len(df_manhattan))
print(df_manhattan.head())

#  Load & Prepare Data 

print("Columns in df_manhattan:", df_manhattan.columns.tolist())  


# Prepare Features and Target Variables


# Define features & target variables
features = ["normalized_average_income_data", "has_subway_access", "normalized_complaints_within_radius", 
            "normalized_precinct_distance", "normalized_park_distance", "normalized_subway_distance", 
            "normalized_rat_sighting_distance", "normalized_closest_rat_sighting_count"] + encoded_category_columns.tolist()

X = df_manhattan[features]

# Targets
y_success = df_manhattan["success"]  # Classification Target
y_bayesian_score = df_manhattan["bayesian_score"]  # Regression Target

# Standardize features (SVM works better with normalized data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train_success, y_test_success = train_test_split(X_scaled, y_success, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_bayesian, y_test_bayesian = train_test_split(X_scaled, y_bayesian_score, test_size=0.2, random_state=42)


# Train SVM Models


# **Classification: Predicting Store Success**
svm_classifier = SVC(kernel="rbf", C=1, gamma="scale")  # SVM for classification
svm_classifier.fit(X_train, y_train_success)

# Predictions & Accuracy
y_pred_success = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test_success, y_pred_success)
print(f"\n SVM Classification Accuracy (Predicting Success): {accuracy:.4f}")

# **Regression: Predicting Bayesian Score**
svm_regressor = SVR(kernel="rbf", C=1, gamma="scale")  # SVM for regression
svm_regressor.fit(X_train_reg, y_train_bayesian)

# Predictions & RMSE
y_pred_bayesian = svm_regressor.predict(X_test_reg)
mse_bayesian = mean_squared_error(y_test_bayesian, y_pred_bayesian)
rmse_bayesian = np.sqrt(mse_bayesian)
print(f"\n SVM Regression RMSE (Predicting Bayesian Score): {rmse_bayesian:.4f}")

# 
# # Optimize Hyperparameters 
# 

# # GridSearch for best parameters (can take time to run)
# param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", "auto", 0.01, 0.1, 1]}
# grid_search = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5, scoring="accuracy")
# grid_search.fit(X_train, y_train_success)

# print("\n Best Parameters for SVM Classification:", grid_search.best_params_)


# Visualize Feature Importance (Optional)


# # We can analyze feature importance using a linear SVM (approximation)
# svm_linear = SVC(kernel="linear", C=1)
# svm_linear.fit(X_train, y_train_success)

# # Get feature importance from linear model coefficients
# feature_importance = abs(svm_linear.coef_).mean(axis=0)
# feature_importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importance})
# feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance_df["Feature"][:10], feature_importance_df["Importance"][:10], color="skyblue")
# plt.xlabel("Feature Importance")
# plt.title("Top 10 Features Impacting Store Success (SVM)")
# plt.gca().invert_yaxis()
# plt.show()