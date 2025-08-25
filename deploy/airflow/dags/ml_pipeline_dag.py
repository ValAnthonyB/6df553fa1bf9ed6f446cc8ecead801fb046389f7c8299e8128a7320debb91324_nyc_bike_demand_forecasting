from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
import pickle
from airflow.operators.python import PythonOperator, BranchPythonOperator
import sys
import mlflow
import optuna
from loguru import logger
import pandas as pd
import json

sys.path.insert(0, '/opt/airflow')
from src.data_preprocessing import (
    get_features_labels,
    process_dataset,
    split_train_test_data,
)
from src.feature_engineering import feature_eng, get_nyc_holidays
from src.training import model_training
from src.evaluate_model import get_metrics
from src.drift_detection import detect_drift

# Configure logging
logger.remove()
logger.add("logs/ml_pipeline_{time}.log")
logger.add(sys.stdout, colorize=True, enqueue=True, backtrace=True, diagnose=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Default arguments for the DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Ensure temp directory exists
TEMP_DIR = Path("/tmp/ml_pipeline/")
Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("reports/")
Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

# MLflow experiment name
experiment_name = "NYC Citi Bike Demand Forecasting"

# Tasks
def setup_mlflow(**context):
    """Connect to MLflow and set up the experiment"""
    mlflow_uri = "http://mlflow:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow_uri

def preprocess_data(**context):
    """Data processing task (renamed from process_data)"""
    df, df_drifted = process_dataset(raw_data_dir="data/raw/")

    # Save processed data to temporary files
    df_path = TEMP_DIR / "processed_df.csv"
    df_drifted_path = TEMP_DIR / "processed_df_drifted.csv"
    df.to_csv(df_path, index=False)
    df_drifted.to_csv(df_drifted_path, index=False)

def feature_engineering(**context):
    # Feature engineering
    nyc_holidays = get_nyc_holidays()

    # Read original dataset
    df_path = TEMP_DIR / "processed_df.csv"
    df = pd.read_csv(df_path)

    # Read drifted dataset
    df_drifted_path = TEMP_DIR / "processed_df_drifted.csv"
    df_drifted = pd.read_csv(df_drifted_path)

    # Feature engineering
    df, feature_names = feature_eng(df, nyc_holidays)
    df_drifted, _ = feature_eng(df_drifted, nyc_holidays)

    # Export feature engineered datasets
    df.to_csv(df_path, index=False)
    df_drifted.to_csv(df_drifted_path, index=False)

    return feature_names # put the feature names in XCom

def train_model(**context):
    # Set up MLflow in this task
    mlflow_uri = context['task_instance'].xcom_pull(task_ids='setup_mlflow')
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    
    # Read feature engineered dataset
    df_path = TEMP_DIR / "processed_df.csv"
    df = pd.read_csv(df_path)

    # Read drifted dataset (not used in initial training)
    df_drifted_path = TEMP_DIR / "processed_df_drifted.csv"
    df_drifted = pd.read_csv(df_drifted_path)

    # Get feature names from XCom
    feature_names = context['task_instance'].xcom_pull(task_ids='feature_engineering')

    # Train and test split
    train_df, test_df = split_train_test_data(df, "2025-07-01")
    train_df_drifted, test_df_drifted = split_train_test_data(df_drifted, "2025-07-01", is_drifted=True)
    
    # Get features and labels (using original data for initial training)
    X_train, X_test, y_train, y_test = get_features_labels(train_df, test_df)

    # Train the random forest model with Optuna
    model, run_id = model_training(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        n_trials=20,
        mlflow_uri=mlflow_uri
    )

    # Export train and test sets as pickle files
    with open(TEMP_DIR / "X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open(TEMP_DIR / "X_test.pkl", "wb") as f:
        pickle.dump(X_test, f)
    with open(TEMP_DIR / "y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
    with open(TEMP_DIR / "y_test.pkl", "wb") as f:
        pickle.dump(y_test, f)
    with open(TEMP_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    return run_id

def evaluate_model(**context):
    # Set up MLflow in this task
    mlflow_uri = context['task_instance'].xcom_pull(task_ids='setup_mlflow')
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    
    run_id = context['task_instance'].xcom_pull(task_ids='train_model')
    
    # Load the pickled data
    with open(TEMP_DIR / "X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open(TEMP_DIR / "X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open(TEMP_DIR / "y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
    with open(TEMP_DIR / "y_test.pkl", "rb") as f:
        y_test = pickle.load(f)
    with open(TEMP_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    # Use nested=True to allow running within an already active run
    with mlflow.start_run(run_id=run_id, nested=True):
        evaluation_metrics = get_metrics(model, X_train, y_train, X_test, y_test)
        
        # Log evaluation metrics
        mlflow.log_metric("MAPE", evaluation_metrics["mape"])
        mlflow.log_metric("MAE", evaluation_metrics["mae"])

        # Register model if it meets performance threshold
        MAPE_THRESHOLD = 11  # MAPE threshold in percentage
    
        if evaluation_metrics["mape"] <= MAPE_THRESHOLD:
            logger.info(f"Model meets threshold (MAPE: {evaluation_metrics['mape']:.2f}% < {MAPE_THRESHOLD}%)")
            
            # Register the model
            registered_model = mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name="BikeRideDemand_RandomForest"
            )
            
        else:
            logger.warning(f"Model does not meet threshold (MAPE: {evaluation_metrics['mape']:.2f}% >= {MAPE_THRESHOLD}%)")

def drift_detection(**context):
    # Run drift detection and save to JSON file
    test_drift_results = detect_drift('data/processed/test.csv', 
                                      'data/processed/drifted_test.csv', 
                                      threshold=0.1)
    
    # Save drift results to JSON file for branching logic
    drift_report_path = REPORTS_DIR / "drift_report.json"
    with open(drift_report_path, 'w') as f:
        json.dump(test_drift_results, f, indent=2)
    
    # Setup MLflow for this task
    mlflow_uri = context['task_instance'].xcom_pull(task_ids='setup_mlflow')
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    
    run_id = context['task_instance'].xcom_pull(task_ids='train_model')
    
    # Log drift metrics to MLflow
    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_param("test_drift_detected", test_drift_results["drift_detected"])
        mlflow.log_metric("test_overall_drift_score", test_drift_results["overall_drift_score"])

def branch_on_drift(**context):
    """Branch based on drift detection results"""
    drift_report_path = REPORTS_DIR / "drift_report.json"
    
    with open(drift_report_path, 'r') as f:
        drift_results = json.load(f)
    
    if drift_results["drift_detected"]:
        logger.info("Data drift detected in test set! Model retraining required. Branching to retrain_model")
        return "retrain_model"
    
    else:
        logger.info("No data drift detected. Proceeding to pipeline completion")
        return "pipeline_complete"
    

def retrain_model(**context):
    """Retrain model using drifted dataset (simulated fresh data)"""
    # Set up MLflow in this task
    mlflow_uri = context['task_instance'].xcom_pull(task_ids='setup_mlflow')
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    # End previous MLflow run (we make a new one here)
    mlflow.end_run()
    
    # Read drifted dataset (treating it as fresh data for retraining)
    df_drifted_path = TEMP_DIR / "processed_df_drifted.csv"
    df_drifted = pd.read_csv(df_drifted_path)

    # Get feature names from XCom
    feature_names = context['task_instance'].xcom_pull(task_ids='feature_engineering')

    # Train and test split using drifted data
    train_df_drifted, test_df_drifted = split_train_test_data(df_drifted, "2025-07-01", is_drifted=True)
    
    # Get features and labels from drifted data
    X_train_drifted, X_test_drifted, y_train_drifted, y_test_drifted = get_features_labels(train_df_drifted, test_df_drifted)

    # Retrain the model on fresh data
    retrained_model, retrain_run_id = model_training(
        X_train_drifted,
        X_test_drifted,
        y_train_drifted,
        y_test_drifted,
        feature_names,
        n_trials=20,
        mlflow_uri=mlflow_uri
    )

    # Evaluate retrained model
    evaluation_metrics = get_metrics(retrained_model, X_train_drifted, y_train_drifted, X_test_drifted, y_test_drifted)
    
    # Log evaluation metrics for retrained model
    mlflow.log_metric("MAPE", evaluation_metrics["mape"])
    mlflow.log_metric("MAE", evaluation_metrics["mae"])
    mlflow.log_param("retrained_after_drift", True)

    # Register retrained model if it meets performance threshold
    MAPE_THRESHOLD = 11  # Same threshold as original model

    if evaluation_metrics["mape"] <= MAPE_THRESHOLD:
        logger.info(f"Retrained model meets threshold (MAPE: {evaluation_metrics['mape']:.2f}% < {MAPE_THRESHOLD}%)")
        
        # Register the retrained model
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name="BikeRideDemand_RandomForest_Retrained"
        )
    else:
        logger.warning(f"Retrained model does not meet threshold (MAPE: {evaluation_metrics['mape']:.2f}% >= {MAPE_THRESHOLD}%)")

    logger.info("Model retraining completed successfully!")

def pipeline_complete(**context):
    """Pipeline completion task"""
    logger.info("ML Pipeline completed successfully without retraining!")
    return "SUCCESS"

# Create the DAG
with DAG(
    dag_id='ml_pipeline_dag',
    default_args=default_args,
    description='ML pipeline for bike demand forecasting with drift detection',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'bike demand forecasting', 'drift detection']
) as dag:
    
    setup_mlflow_task = PythonOperator(
        task_id='setup_mlflow',
        python_callable=setup_mlflow
    )

    preprocess_data_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    feature_engineering_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model
    )

    drift_detection_task = PythonOperator(
        task_id='drift_detection',
        python_callable=drift_detection
    )

    branch_on_drift_task = BranchPythonOperator(
        task_id='branch_on_drift',
        python_callable=branch_on_drift
    )

    retrain_model_task = PythonOperator(
        task_id='retrain_model',
        python_callable=retrain_model
    )

    pipeline_complete_task = PythonOperator(
        task_id='pipeline_complete',
        python_callable=pipeline_complete
    )

    # Task dependencies
    (
        setup_mlflow_task >> 
        preprocess_data_task >> 
        feature_engineering_task >> 
        train_model_task >> 
        evaluate_model_task >>
        drift_detection_task >>
        branch_on_drift_task >> 
        [retrain_model_task, pipeline_complete_task]
    )

# Test command: docker-compose exec airflow-webserver airflow dags test ml_pipeline_dag 2025-08-02