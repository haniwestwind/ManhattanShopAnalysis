from sklearn.metrics import mean_squared_error
import time
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Trains and evaluates a model, tracking metrics and saving results."""

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    elapsed_time = end_time - start_time

    if isinstance(model, LinearRegression):
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        performance_metric = rmse
        metric_name = "RMSE"
    elif isinstance(model, LogisticRegression):
        accuracy = model.score(X_test, y_test)
        performance_metric = accuracy
        metric_name = "Accuracy"
    else:
        raise ValueError("Unsupported model type")

    # Save results to file
    # Save results to file
    # script_name = os.path.splitext(os.path.basename(__file__))[0]
    # filename = f"{script_name}_{model_name}_results.txt"
    # Get the name of the calling script
    import inspect
    caller_frame = inspect.stack()[1]  # Get the caller's frame
    caller_filename = os.path.basename(caller_frame.filename)
    script_name = os.path.splitext(caller_filename)[0]

    # Save results to file
    filename = f"{script_name}_{model_name}_results.txt"
    with open(filename, "w") as f:
        f.write(f"Training Time: {elapsed_time:.4f} seconds\n")
        f.write(f"{metric_name}: {performance_metric:.4f}\n")

    print(f"Results for {model_name} saved to {filename}")