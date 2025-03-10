import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import export_graphviz
import graphviz
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from data_preprocessing import df_manhattan, encoded_category_columns

"""Decision Tree using all features and categories"""

print("Manhattan stores loaded:", len(df_manhattan))
print(df_manhattan.head())

#  Load & Prepare Data 

print("Columns in df_manhattan:", len(df_manhattan.columns.tolist()))  


# Prepare Features and Target Variables


# Define features & target variables
features = ["normalized_average_income_data", "has_subway_access", "normalized_complaints_within_radius", 
            "normalized_precinct_distance", "normalized_park_distance", "normalized_subway_distance", 
            "normalized_rat_sighting_distance", "normalized_closest_rat_sighting_count"] + encoded_category_columns.tolist()
print("Size ", len(encoded_category_columns.tolist()))
print("Size ", len(features))
X = df_manhattan[features]
print("Shape of X ", X.shape)
y_success = df_manhattan["success"]  # Binary classification target
y_bayesian_score = df_manhattan["bayesian_score"]  # Continuous regression target

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train_success, y_test_success = train_test_split(X_scaled, y_success, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_bayesian, y_test_bayesian = train_test_split(X_scaled, y_bayesian_score, test_size=0.2, random_state=42)


# Decision Tree Classifier (Predicting Store Success)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train_success)

# Evaluate Accuracy
accuracy = clf.score(X_test, y_test_success)
print(f"\nDecision Tree Classification Accuracy: {accuracy:.4f}")


# Decision Tree Regressor (Predicting Bayesian Score)

reg = DecisionTreeRegressor(max_depth=5, random_state=42)
reg.fit(X_train_reg, y_train_bayesian)

# Evaluate RMSE
y_pred = reg.predict(X_test_reg)
rmse = np.sqrt(((y_pred - y_test_bayesian) ** 2).mean())
print(f"\nDecision Tree Regression RMSE: {rmse:.4f}")


# Feature Importance

# Ensure the number of features matches the length of feature importances
feature_importance = pd.DataFrame({
    "Feature": X.columns[:len(reg.feature_importances_)],  # Match the correct number of features
    "Importance": reg.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importances (Regression Model):")
print(feature_importance.head(10))  # Display top 10 features


# Visualize Tree

# Limit max depth for visualization speed (optional)
print("Length of the features ", len(features))
print(X.shape)
dot_data = export_graphviz(
    clf, out_file=None, feature_names=X.columns.tolist(), class_names=["Not Successful", "Successful"], 
    filled=True, rounded=True, special_characters=True, max_depth=4  # Limit depth to 4
)

graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Saves as decision_tree.pdf
graph