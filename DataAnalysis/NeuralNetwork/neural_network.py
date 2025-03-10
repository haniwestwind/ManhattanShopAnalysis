import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import time
import os
from matplotlib import pyplot as plt

from data_preprocessing import df_manhattan, encoded_category_columns

#  Load & Prepare Data 

print("Manhattan stores loaded:", len(df_manhattan))
print(df_manhattan.head())

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
def build_regression_model(input_shape):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)  # Single output for regression
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

start_bayesian = time.time()
model_bayesian = build_regression_model(X_train.shape[1])
history_bayesian = model_bayesian.fit(X_train, y_train_bayesian, epochs=100, batch_size=32, verbose=1, validation_split=0.2)
end_bayesian = time.time()
elapsed_bayesian = end_bayesian - start_bayesian

# Evaluate Bayesian Model
y_pred_bayesian = model_bayesian.predict(X_test).flatten()
mse_bayesian = mean_squared_error(y_test_bayesian, y_pred_bayesian)
rmse_bayesian = np.sqrt(mse_bayesian)
print(f"\nNeural Network Bayesian Score RMSE: {rmse_bayesian:.4f}")

# Save the Bayesian model
model_bayesian.save('neural_model_bayesian.h5')

# Plot Bayesian Loss
plt.figure(figsize=(10, 5))
plt.plot(history_bayesian.history['loss'], label='Training Loss')
plt.plot(history_bayesian.history['val_loss'], label='Validation Loss')
plt.title('Bayesian Model Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('bayesian_model_loss.png')
plt.show()

# 2. Neural Network for IMDB Score (Regression)
start_imdb = time.time()
model_imdb = build_regression_model(X_train.shape[1])
history_imdb = model_imdb.fit(X_train, y_train_imdb, epochs=100, batch_size=32, verbose=1, validation_split=0.2)
end_imdb = time.time()
elapsed_imdb = end_imdb - start_imdb

# Evaluate IMDB Model
y_pred_imdb = model_imdb.predict(X_test).flatten()
mse_imdb = mean_squared_error(y_test_imdb, y_pred_imdb)
rmse_imdb = np.sqrt(mse_imdb)
print(f"\nNeural Network IMDB Score RMSE: {rmse_imdb:.4f}")

# Save the IMDB model
model_imdb.save('neural_model_imdb.h5')

#Plot Imdb Loss
plt.figure(figsize=(10, 5))
plt.plot(history_imdb.history['loss'], label='Training Loss')
plt.plot(history_imdb.history['val_loss'], label='Validation Loss')
plt.title('IMDB Model Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('imdb_model_loss.png')
plt.show()

# 3. Neural Network for Success (Classification)
def build_classification_model(input_shape):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

start_success = time.time()
model_success = build_classification_model(X_train.shape[1])
history_success = model_success.fit(X_train, y_train_success, epochs=100, batch_size=32, verbose=1, validation_split=0.2)
end_success = time.time()
elapsed_success = end_success - start_success

# Evaluate Success Model
y_pred_success = (model_success.predict(X_test) > 0.5).astype("int32")
accuracy_success = accuracy_score(y_test_success, y_pred_success)
print(f"\nNeural Network Success Accuracy: {accuracy_success:.4f}")

# Save the Success model
model_success.save('neural_model_success.h5')

#Plot Success Loss
plt.figure(figsize=(10, 5))
plt.plot(history_success.history['loss'], label='Training Loss')
plt.plot(history_success.history['val_loss'], label='Validation Loss')
plt.plot(history_success.history['accuracy'], label = "Training Accuracy")
plt.plot(history_success.history['val_accuracy'], label = "Validation Accuracy")
plt.title('Success Model Loss and Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.savefig('success_model_loss.png')
plt.show()

print("\nNeural network models saved successfully.")

# Write results to file
script_name = os.path.splitext(os.path.basename(__file__))[0] + ".txt"
with open(script_name, "w") as f:
    f.write(f"Bayesian Model Training Time: {elapsed_bayesian:.4f} seconds\n")
    f.write(f"Bayesian Model RMSE: {rmse_bayesian:.4f}\n")
    f.write(f"IMDB Model Training Time: {elapsed_imdb:.4f} seconds\n")
    f.write(f"IMDB Model RMSE: {rmse_imdb:.4f}\n")
    f.write(f"Success Model Training Time: {elapsed_success:.4f} seconds\n")
    f.write(f"Success Model Accuracy: {accuracy_success:.4f}\n")

print(f"Results written to {script_name}")