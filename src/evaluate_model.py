import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error
)

def get_metrics(
    model: RandomForestRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict:
    """
    Evaluates the Random Forest regression model on the training and test datasets
    using MAE, RMSE, and MAPE metrics.

    Returns:
    -------
    dict
        Dictionary containing RMSE, MAE, MAPE, and R2 for both the training and test sets.
    """
    logger.info(f"Calculating metrics for {model_name}")

    # Make predictions
    logger.info("Making predictions on training and test sets")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Plot the predictions
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    plt.plot(list(range(len(y_test))), y_test, color='black', label="Ground Truth")
    plt.plot(list(range(len(y_test))), y_pred_test, ls='--', color='blue', lw=0.7, label="Predicted")
    plt.title("Test Ground Truth vs Predicted")
    plt.legend()
    plt.savefig("reports/forecast.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Train metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train) * 100

    # Test metrics
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100

    # Print the results
    logger.info(f"Train Set Metrics ({model_name}):")
    logger.info(f"RMSE: {rmse_train:.2f}")
    logger.info(f"MAE: {mae_train:.2f}")
    logger.info(f"MAPE: {mape_train:.2f}%")

    logger.info(f"Test Set Metrics ({model_name}):")
    logger.info(f"RMSE: {rmse_test:.2f}")
    logger.info(f"MAE: {mae_test:.2f}")
    logger.info(f"MAPE: {mape_test:.2f}%")

    # Put the test metrics in a dict
    metrics_dict = {
        "rmse": rmse_test, 
        "mae": mae_test, 
        "mape": mape_test,
    }

    # Export evaluation report
    with open("reports/evaluation_results.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    return metrics_dict