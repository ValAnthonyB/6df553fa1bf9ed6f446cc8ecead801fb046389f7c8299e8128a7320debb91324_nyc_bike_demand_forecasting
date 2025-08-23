import sys

import optuna
from loguru import logger

from data_preprocessing import (
    get_features_labels,
    process_dataset,
    split_train_test_data,
)
from evaluate_model import calculate_metrics, save_metrics
from feature_engineering import export_feature_eng_data, feature_eng, get_nyc_holidays
from training import export_model, get_best_rf_model

# Add file logging with timestamp
logger.add("logs/ml_pipeline_{time}.log")
logger.add(sys.stdout, colorize=True, enqueue=True, backtrace=True, diagnose=True)

# Suppress optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    logger.info("Starting ML pipeline execution")

    logger.info("Reading and processing the dataset")
    df, df_drifted = process_dataset(raw_data_dir="data/raw/")
    logger.success("Dataset processing completed")

    # Feature engineering step
    logger.info("Engineering new features")
    # Get holidays in New York State from 2023 to 2025
    nyc_holidays = get_nyc_holidays()
    df = feature_eng(df, nyc_holidays)
    df_drifted = feature_eng(df_drifted, nyc_holidays)
    logger.info(f"Feature engineered dataset shape: {df.shape}")

    # Export the feature-engineered data
    logger.info("Exporting feature-engineered data")
    export_feature_eng_data(df, "data/processed/")
    logger.success("Feature-engineered data exported successfully")

    # Train test split. Use last month as the test set.
    logger.info("Splitting data into train and test sets")
    train_df, test_df = split_train_test_data(df, "2025-07-01")
    train_df_drifted, test_df_drifted = split_train_test_data(
        df_drifted, "2025-07-01", is_drifted=True
    )
    logger.info(f"Train dataset shape: {train_df.shape}")
    logger.info(f"Test dataset shape: {test_df.shape}")

    # Get the features and target variable
    logger.info("Extracting features and labels")
    X_train, X_test, y_train, y_test = get_features_labels(train_df, test_df)
    X_train_drifted, X_test_drifted, y_train_drifted, y_test_drifted = (
        get_features_labels(train_df_drifted, test_df_drifted)
    )
    logger.success("Features and labels extracted successfully")

    # Model training with optuna
    logger.info("Starting RF model training with hyperparameter tuning (Optuna)")
    model_name = "RF_model"
    model = get_best_rf_model(X_train, X_test, y_train, y_test, n_trials=3)
    logger.success("Model training completed successfully")

    # Export the model as a joblib file
    model_path = f"models/{model_name}.joblib"
    logger.info("Exporting trained model")
    export_model(model, model_path)
    logger.success(f"Model exported to {model_path}")

    # Evaluate the model
    logger.info("Evaluating model performance")
    metrics_df = calculate_metrics(
        model, X_train, y_train, X_test, y_test, model_name=model_name
    )
    logger.success("Model evaluation completed")

    # Evaluate the model on the drifted test set
    metrics_df = calculate_metrics(
        model, X_train, y_train, X_test, y_test, model_name=model_name
    )

    # Evaluate on the drifted data
    logger.info("Evaluating model performance on the drifted dataset.")
    metrics_df_drifted = calculate_metrics(
        model,
        X_train_drifted,
        y_train_drifted,
        X_test_drifted,
        y_test_drifted,
        model_name=model_name,
    )

    # Save model metrics
    logger.info("Saving model metrics")
    save_metrics(metrics_df, model_name=model_name)
    save_metrics(metrics_df_drifted, model_name="RF_drifted")
    logger.success("Model metrics saved successfully")

    logger.success("ML pipeline execution completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.exception("Full traceback:")
        raise
