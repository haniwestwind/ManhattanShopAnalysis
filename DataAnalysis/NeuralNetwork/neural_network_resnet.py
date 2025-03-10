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

# Reshape input data for 1D convolution
X_train_reshaped = np.expand_dims(X_train, axis=-1)
X_test_reshaped = np.expand_dims(X_test, axis=-1)

#  Neural Network Models 

# 1. Neural Network for Bayesian Score (Regression)
def build_regression_model_resnet(input_shape, l2_reg=0.01, dropout_rate=0.3):
    start_compile = time.time()
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

    def residual_block(x, filters, stride=1):
        shortcut = x
        x = layers.Conv1D(filters, 3, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        if stride != 1 or x.shape[-1] != shortcut.shape[-1]:
            shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 512, stride=2)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    end_compile = time.time()
    compile_time = end_compile - start_compile
    return model, compile_time

model_bayesian, bayesian_compile_time = build_regression_model_resnet(X_train_reshaped.shape[1:])
start_fit_bayesian = time.time()
history_bayesian = model_bayesian.fit(X_train_reshaped, y_train_bayesian, epochs=50, batch_size=32, verbose=1, validation_split=0.2)
end_fit_bayesian = time.time()
fit_time_bayesian = end_fit_bayesian - start_fit_bayesian

plt.figure(figsize=(10, 5))
plt.plot(history_bayesian.history['loss'], label='Training Loss')
plt.plot(history_bayesian.history['val_loss'], label='Validation Loss')
plt.title('Bayesian ResNet Model Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('bayesian_resnet_model_loss.png')

# Evaluate Bayesian Model
y_pred_bayesian = model_bayesian.predict(X_test_reshaped).flatten()
mse_bayesian = mean_squared_error(y_test_bayesian, y_pred_bayesian)
rmse_bayesian = np.sqrt(mse_bayesian)
print(f"\nNeural Network Bayesian Score RMSE: {rmse_bayesian:.4f}")

# Save the Bayesian model
model_bayesian.save('neural_resnet_model_bayesian.h5')

# 3. Neural Network for Success (Classification)
def build_classification_model_resnet(input_shape, l2_reg=0.01, dropout_rate=0.3):
    start_compile = time.time()
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

    def residual_block(x, filters, stride=1):
        shortcut = x
        x = layers.Conv1D(filters, 3, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        if stride != 1 or x.shape[-1] != shortcut.shape[-1]:
            shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 512, stride=2)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    end_compile = time.time()
    compile_time = end_compile - start_compile
    return model, compile_time

model_success, success_compile_time = build_classification_model_resnet(X_train_reshaped.shape[1:])
start_fit_success = time.time()

history_success = model_success.fit(X_train_reshaped, y_train_success, epochs=50, batch_size=32, verbose=1, validation_split=0.2)
end_fit_success = time.time()
fit_time_success = end_fit_success - start_fit_success
plt.figure(figsize=(10, 5))
plt.plot(history_success.history['loss'], label='Training Loss')
plt.plot(history_success.history['val_loss'], label='Validation Loss')
plt.title('Success ResNet Model Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('success_resnet_model_loss.png')
plt.show()

# Evaluate Success Model
y_pred_success = (model_success.predict(X_test_reshaped) > 0.5).astype("int32")
accuracy_success = accuracy_score(y_test_success, y_pred_success)
print(f"\nNeural Network Success Accuracy: {accuracy_success:.4f}")

# Save the Success model
model_success.save('neural_resnet_model_success.h5')

# Write results to file
script_name = os.path.splitext(os.path.basename(__file__))[0] + ".txt"
with open(script_name, "w") as f:
    f.write(f"Bayesian Model Compile Time: {bayesian_compile_time:.4f} seconds\n")
    f.write(f"Bayesian Model Fit Time: {fit_time_bayesian:.4f} seconds\n")
    f.write(f"Bayesian Model RMSE: {rmse_bayesian:.4f}\n")
    f.write(f"Success Model Compile Time: {success_compile_time:.4f} seconds\n")
    f.write(f"Success Model Fit Time: {fit_time_success:.4f} seconds\n")
    f.write(f"Success Model Accuracy: {accuracy_success:.4f}\n")
   
print(f"Results written to {script_name}")