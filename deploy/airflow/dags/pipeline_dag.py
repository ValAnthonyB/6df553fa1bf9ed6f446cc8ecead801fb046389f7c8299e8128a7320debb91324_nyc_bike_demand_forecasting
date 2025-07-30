import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add project root to Python path
sys.path.insert(0, "/opt/airflow")


def process_raw_data(raw_data_dir="data/raw/", output_dir="data/temp/", **kwargs):
    from src.data_preprocessing import process_dataset

    # Take all parquet files in the raw folder
    df = process_dataset(raw_data_dir=raw_data_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Export concatenated parquet file
    df.to_parquet(Path(output_dir) / "concatenated.parquet", compression="gzip")


def feature_engineering(**kwargs):
    import pandas as pd

    from src.feature_engineering import (
        export_feature_eng_data,
        feature_eng,
        get_nyc_holidays,
    )

    # Read the preprocessed file.
    df = pd.read_parquet("data/temp/concatenated.parquet")

    # Get the NYC holidays
    nyc_holidays = get_nyc_holidays()

    # Apply feature engineering
    df = feature_eng(df, nyc_holidays)

    # Export the feature engineered dataset to the processed/ dir
    export_feature_eng_data(df, "data/processed/")


def date_splitting(**kwargs):
    import pandas as pd

    from src.data_preprocessing import split_train_test_data

    # Get the feature engineered data
    df = pd.read_parquet("data/processed/feature_engineered_data.parquet")

    # Time series cross-validation
    train_df, test_df = split_train_test_data(df, "2024-05-20")

    # Export to parquet files
    train_df.to_parquet("data/temp/train_set.parquet", compression="gzip")
    test_df.to_parquet("data/temp/test_set.parquet", compression="gzip")


def model_training(**kwargs):
    """Trains a random forest regressor"""
    import pandas as pd

    from src.data_preprocessing import get_features_labels
    from src.training import export_model, get_best_rf_model

    # Get the features and labels
    train_df = pd.read_parquet("data/temp/train_set.parquet")
    test_df = pd.read_parquet("data/temp/test_set.parquet")
    X_train, X_test, y_train, y_test = get_features_labels(train_df, test_df)

    # Train the best lightgbm model
    model = get_best_rf_model(X_train, X_test, y_train, y_test, n_trials=50)

    # Dump model
    export_model(model, "models/RF_model.joblib")


def model_evaluation(**kwargs):
    """
    Measures model performance on train and test splits.
    """
    import pandas as pd
    from joblib import load

    from src.data_preprocessing import get_features_labels
    from src.evaluate_model import calculate_metrics, save_metrics

    # Load the model
    model = load("models/RF_model.joblib")

    # Get train and test
    train_df = pd.read_parquet("data/temp/train_set.parquet")
    test_df = pd.read_parquet("data/temp/test_set.parquet")
    X_train, X_test, y_train, y_test = get_features_labels(train_df, test_df)

    # Evaluate the model
    metrics_df = calculate_metrics(
        model, X_train, y_train, X_test, y_test, model_name="RF_model"
    )

    # Save model metrics into the reports directory
    save_metrics(metrics_df, model_name="RF_model")


def temp_cleanup(**kwargs):
    import shutil

    temp_dir = "data/temp"
    path = Path(temp_dir)

    path.mkdir(parents=True, exist_ok=True)

    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


# ------ Setup the DAGs ------
dag = DAG(
    "bike_forecasting_pipeline",
    description="Pipeline with data processing, training, and evaluation report.",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # require manual triggers
    catchup=False,  # no backfilling
)

process_raw_data_task = PythonOperator(
    task_id="process_raw_data",
    python_callable=process_raw_data,
    op_kwargs={"raw_data_dir": "data/raw/", "output_dir": "data/temp/"},
    dag=dag,
)

feature_engineering_task = PythonOperator(
    task_id="feature_engineering",
    python_callable=feature_engineering,
    provide_context=True,
    dag=dag,
)

date_splitting_task = PythonOperator(
    task_id="date_splitting",
    python_callable=date_splitting,
    provide_context=True,
    dag=dag,
)

model_training_task = PythonOperator(
    task_id="model_training",
    python_callable=model_training,
    provide_context=True,
    dag=dag,
)

model_evaluation_task = PythonOperator(
    task_id="model_evaluation",
    python_callable=model_evaluation,
    provide_context=True,
    dag=dag,
)

temp_cleanup_task = PythonOperator(
    task_id="temp_cleanup",
    python_callable=temp_cleanup,
    provide_context=True,
    dag=dag,
)

# Sequence
(
    process_raw_data_task
    >> feature_engineering_task
    >> date_splitting_task
    >> model_training_task
    >> model_evaluation_task
    >> temp_cleanup_task
)
