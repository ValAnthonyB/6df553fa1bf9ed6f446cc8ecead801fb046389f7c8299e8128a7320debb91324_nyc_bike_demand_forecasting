import sys
import mlflow
import optuna
from loguru import logger

from data_preprocessing import (
    get_features_labels,
    process_dataset,
    split_train_test_data,
)

from feature_engineering import feature_eng, get_nyc_holidays
from training import model_training
from evaluate_model import get_metrics
from drift_detection import detect_drift

# Add file logging
logger.remove()
logger.add("logs/ml_pipeline_{time}.log")
logger.add(sys.stdout, colorize=True, enqueue=True, backtrace=True, diagnose=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    logger.info("Starting ML pipeline execution")

    # Set up MLflow experiment
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("NYC Citi Bike Demand Forecasting")

    # Data processing
    logger.info("Reading and processing the dataset")
    df, df_drifted = process_dataset(raw_data_dir="data/raw/")
    logger.success("Dataset processing completed")

    # Feature engineering
    logger.info("Engineering new features")
    nyc_holidays = get_nyc_holidays()
    df, feature_names = feature_eng(df, nyc_holidays)
    df_drifted, _ = feature_eng(df_drifted, nyc_holidays)
    logger.info(f"Feature engineered dataset shape: {df.shape}")

    # Train test split
    logger.info("Splitting data into train and test sets")
    train_df, test_df = split_train_test_data(df, "2025-07-01")
    train_df_drifted, test_df_drifted = split_train_test_data(
        df_drifted, "2025-07-01", is_drifted=True
    )

    # Get features and labels
    X_train, X_test, y_train, y_test = get_features_labels(train_df, test_df)
    logger.success("Features and labels extracted successfully")

    # Model training (keeps run active)
    logger.info("Starting RF model training")
    model, run_id = model_training(
        X_train, X_test, y_train, y_test, feature_names, n_trials=20
    )
    logger.success("Model training completed successfully")

    # Model evaluation (continue with the active run from training)
    logger.info("Evaluating model performance")
    
    # The run is still active from training, so we don't need to start a new one
    # Just continue logging to the active run
    evaluation_metrics = get_metrics(model, X_train, y_train, X_test, y_test)
    mlflow.log_metric("MAPE", evaluation_metrics["mape"])
    mlflow.log_metric("MAE", evaluation_metrics["mae"])
    
    # Model registration (within the same active run)
    logger.info("Checking model performance for registration eligibility")
    MAPE_THRESHOLD = 11  # MAPE threshold in percentage
    
    if evaluation_metrics["mape"] <= MAPE_THRESHOLD:
        logger.info(f"Model meets threshold (MAPE: {evaluation_metrics['mape']:.2f}% < {MAPE_THRESHOLD}%)")
        
        # Register the model
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name="BikeRideDemand_RandomForest"
        )
        
        logger.info(f"Model registered successfully!")
        logger.info(f"Model Name: {registered_model.name}")
        logger.info(f"Model Version: {registered_model.version}")
        
    else:
        logger.warning(f"Model does not meet threshold (MAPE: {evaluation_metrics['mape']:.2f}% >= {MAPE_THRESHOLD}%)")

    # Run drift detection and log to the same active run
    logger.info("Running drift detection")
    test_drift_results = detect_drift('data/processed/test.csv', 
                                      'data/processed/drifted_test.csv', 
                                      threshold=0.1)
    
    top_5_drifted = sorted(test_drift_results["feature_drifts"].items(), key=lambda x: x[1])[:5]
    logger.info("Top 5 most drifted features:")
    [logger.info(f"{f}: p={p:.4f}") for f, p in top_5_drifted]

    # Log drift status to the same active run
    mlflow.log_param("test_drift_detected", test_drift_results["drift_detected"])
    mlflow.log_metric("test_overall_drift_score", test_drift_results["overall_drift_score"])

    # End the run after all logging is complete
    mlflow.end_run()
    logger.success("Model evaluation successful")

    # Raise error if drift detected (after the run is complete)
    if test_drift_results["drift_detected"]:
        raise ValueError("Data drift detected in test set! Model retraining required.")

    logger.success("ML pipeline execution completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.exception("Full traceback:")
        raise