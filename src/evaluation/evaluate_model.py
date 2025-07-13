from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import os

def calculate_metrics(
    model, 
    X_train, y_train,
    X_test, y_test, 
    model_name="Model"
):
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

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
        "test":  {"rmse": rmse_test,  "mae": mae_test,  "mape": mape_test}
    }


def save_metrics(metrics_dict, model_name):
    save_path = f"reports/{model_name}.csv"

    rows = []
    for split, metrics in metrics_dict.items():
        for metric_name, value in metrics.items():
            rows.append({
                "model_name": model_name,
                "split": split,
                "metric": metric_name,
                "value": round(value, 4)
            })
    
    df_metrics = pd.DataFrame(rows)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save as CSV (append if file exists, otherwise write header)
    write_header = not os.path.exists(save_path)
    df_metrics.to_csv(save_path, mode='a', index=False, header=write_header)

    print(f"Metrics saved to: {save_path}")