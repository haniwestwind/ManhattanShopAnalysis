import pymongo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import time
import os

from data_preprocessing import df_manhattan, encoded_category_columns

#  Load & Prepare Data 

print("Manhattan stores loaded:", len(df_manhattan))
print(df_manhattan.head())

# print("Columns in df_manhattan:", df_manhattan.columns.tolist())

# Features to use
features = ["normalized_average_income_data", "has_subway_access", "normalized_complaints_within_radius", "normalized_precinct_distance",
    "normalized_park_distance", "normalized_subway_distance", "normalized_rat_sighting_distance", "normalized_closest_rat_sighting_count"] + encoded_category_columns.tolist()

y_success = df_manhattan["success"]
y_bayesian_score = df_manhattan["bayesian_score"]
y_imdb_score = df_manhattan["imdb_score"]

X = df_manhattan[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Split data into training and testing sets
X_train, X_test, y_train_success, y_test_success, y_train_bayesian, y_test_bayesian, y_train_imdb, y_test_imdb = train_test_split(
    X_scaled, y_success, y_bayesian_score, y_imdb_score, test_size=0.2, random_state=42
)

#  Neural Network Models 

# 1. Neural Network for Bayesian Score (Regression)
def build_regression_model_wider(input_shape):
    start_compile = time.time()
    model = keras.Sequential([
        layers.Dense(1024, activation='relu', input_shape=(input_shape,), kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    end_compile = time.time()
    compile_time = end_compile - start_compile
    return model, compile_time

model_bayesian, bayesian_compile_time = build_regression_model_wider(X_train.shape[1])
model_bayesian.fit(X_train, y_train_bayesian, epochs=100, batch_size=32, verbose=1)

# Evaluate Bayesian Model
y_pred_bayesian = model_bayesian.predict(X_test).flatten()
mse_bayesian = mean_squared_error(y_test_bayesian, y_pred_bayesian)
rmse_bayesian = np.sqrt(mse_bayesian)
print(f"\nNeural Network Bayesian Score RMSE: {rmse_bayesian:.4f}")

# Save the Bayesian model
model_bayesian.save('neural_model_bayesian_wider.h5')

# 3. Neural Network for Success (Classification)
def build_classification_model_wider(input_shape):
    start_compile = time.time()
    model = keras.Sequential([
        layers.Dense(1024, activation='relu', input_shape=(input_shape,), kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    end_compile = time.time()
    compile_time = end_compile - start_compile
    return model, compile_time

model_success, success_compile_time = build_classification_model_wider(X_train.shape[1])
model_success.fit(X_train, y_train_success, epochs=100, batch_size=32, verbose=1)

# Evaluate Success Model
y_pred_success = (model_success.predict(X_test) > 0.5).astype("int32")
accuracy_success = accuracy_score(y_test_success, y_pred_success)
print(f"\nNeural Network Success Accuracy: {accuracy_success:.4f}")

# Save the Success model
model_success.save('neural_model_success_wider.h5')

print("\nNeural network models saved successfully.")

# Write results to file
script_name = os.path.splitext(os.path.basename(__file__))[0] + ".txt"
with open(script_name, "w") as f:
    f.write(f"Bayesian Model Compile Time: {bayesian_compile_time:.4f} seconds\n")
    f.write(f"Bayesian Model RMSE: {rmse_bayesian:.4f}\n")
    f.write(f"Success Model Compile Time: {success_compile_time:.4f} seconds\n")
    f.write(f"Success Model Accuracy: {accuracy_success:.4f}\n")

print(f"Results written to {script_name}")