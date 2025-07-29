import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task

# We use this to access the scripts in the src/ dir
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# Data processing DAG
@dag(dag_id="data_processing", schedule=None, start_date=datetime(2024, 1, 1))
def preprocessing_pipeline():
    """
    Contains the tasks to run the data preprocessing step.
    """

    @task
    def process_raw_data():
        """Reads the raw datasets and concatenates them into a single file."""
        from src.data_preprocessing import process_dataset

        df = process_dataset(raw_data_dir="data/raw/")

        # Save intermediate result
        df.to_csv("data/processed/preprocessed_data.csv", index=False)

        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "output_path": "data/processed/preprocessed_data.csv",
        }
