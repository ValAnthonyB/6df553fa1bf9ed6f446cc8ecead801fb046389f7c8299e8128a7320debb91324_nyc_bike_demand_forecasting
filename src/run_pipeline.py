from datetime import date

import optuna

from data_preprocessing import (
    get_features_labels,
    process_dataset,
    split_train_test_data,
)
from evaluate_model import calculate_metrics, save_metrics
from feature_engineering import export_feature_eng_data, feature_eng, get_nyc_holidays
from training import export_model, get_best_lightgbm_model

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    print("Reading and processing the dataset\n")
    df = process_dataset(raw_data_dir="../data/raw/")

    # Feature engineering step
    print("Engineering new features\n")

    # Get holidays in New York State from 2023 to 2025
    nyc_holidays = get_nyc_holidays()
    df = feature_eng(df, nyc_holidays)
    print(f"Feature engineered dataset: {df.shape}\n")

    # Export the feature-engineered data
    export_feature_eng_data(df, "../data/processed/")

    train_df, test_df = split_train_test_data(df, "2024-05-20")
    print(f"Train dataset: {train_df.shape}")
    print(f"Test dataset: {test_df.shape}\n")

    # Get the features and target variable
    X_train, X_test, y_train, y_test = get_features_labels(train_df, test_df)

    # Model training with optuna
    print("Model training using LightGBM with hyperparameter tuning Optuna.\n")
    today_str = date.today().strftime(
        "%Y-%m-%d"
    )  # Get today's date in YYYY-MM-DD format
    model_name = f"LightGBM_{today_str}"
    model = get_best_lightgbm_model(X_train, X_test, y_train, y_test, n_trials=50)

    # Export the model as a joblib file
    print("Exporting model\n")
    export_model(model, f"../models/{model_name}.joblib")

    # Evaluate the model
    metrics_df = calculate_metrics(
        model, X_train, y_train, X_test, y_test, model_name=model_name
    )

    # Save model metrics
    save_metrics(metrics_df, model_name=model_name)


if __name__ == "__main__":
    main()
