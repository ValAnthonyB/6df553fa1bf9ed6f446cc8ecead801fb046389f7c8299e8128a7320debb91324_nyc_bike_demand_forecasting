from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import os

MLFLOW_TRACKING_URI = "http://mlflow:5000"
MLFLOW_EXPERIMENT = "simple_test"

def log_to_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="hello_mlflow"):
        # log a simple param and metric
        mlflow.log_param("param1", "hello")
        mlflow.log_metric("metric1", 123)

        # log a very small artifact (text file)
        os.makedirs("/tmp/artifacts", exist_ok=True)
        filepath = "/tmp/artifacts/test.txt"
        with open(filepath, "w") as f:
            f.write("This is a simple test artifact!")

        mlflow.log_artifact(filepath)

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 0,
}

with DAG(
    dag_id="simple_mlflow_logger",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    log_task = PythonOperator(
        task_id="log_to_mlflow",
        python_callable=log_to_mlflow,
    )
