import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def calculate_metrics(
    model: LGBMRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict:
    """
    Evaluates the LightGBM regression model on the training and test datasets
    using RMSE, MAE, and MAPE metrics.

    Parameters:
    ----------
    X_train : pd.DataFrame
        Training features.

    X_test : pd.DataFrame
        Testing features.

    y_train : pd.Series
        Training labels.

    y_test : pd.Series
        Testing labels.

    Returns:
    -------
    dict
        Dictionary containing RMSE, MAE, and MAPE for both the training and test sets.
    """
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Train metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train) * 100

    # Test metrics
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100

    # Print results
    print(f"\nTrain Set Metrics ({model_name}):")
    print(f"RMSE: {rmse_train:.2f}")
    print(f"MAE: {mae_train:.2f}")
    print(f"MAPE: {mape_train:.2f}%")

    print(f"\nTest Set Metrics ({model_name}):")
    print(f"RMSE: {rmse_test:.2f}")
    print(f"MAE: {mae_test:.2f}")
    print(f"MAPE: {mape_test:.2f}%")

    # Return all metrics in a structured format
    return {
        "train": {"rmse": rmse_train, "mae": mae_train, "mape": mape_train},
        "test": {"rmse": rmse_test, "mae": mae_test, "mape": mape_test},
    }


def save_metrics(metrics: dict, model_name: str) -> None:
    """
    Exports the model metrics on the train test sets locally.

    Parameters:
    ----------
    metrics : dict
        Dictionary containing the RMSE, MAE, and MAPE scores on the train
        and test sets.

    model_name : str
        Name of the model.
    """
    save_path = f"reports/{model_name}.csv"

    # Convert to DataFrame
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index").reset_index()

    # Rename columns
    metrics_df.columns = ["Dataset", "RMSE", "MAE", "MAPE"]

    # Create directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)

    # Save as CSV (append if file exists, otherwise write header)
    metrics_df.to_csv(save_path, index=False)
