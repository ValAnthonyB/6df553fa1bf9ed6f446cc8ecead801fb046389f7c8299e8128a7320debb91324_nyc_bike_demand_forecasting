import os

import pandas as pd
from dags.pipeline_dag import process_raw_data


def test_process_raw_data_creates_output_dir(tmp_path):
    """Test that process_raw_data creates output and processes data correctly."""
    # Setup directories
    raw_data_dir = tmp_path / "data" / "raw"
    temp_dir = tmp_path / "data" / "temp"
    raw_data_dir.mkdir(parents=True)
    temp_dir.mkdir(parents=True)

    # Create test data
    test_data = pd.DataFrame(
        {
            "ride_date": pd.date_range("2023-01-01", periods=2, freq="D"),
            "unique_rides": [10, 20],
        }
    )
    test_data.to_parquet(raw_data_dir / "test.parquet", engine="pyarrow")

    # Execute and assert
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        process_raw_data()

        output_file = temp_dir / "concatenated.parquet"
        assert output_file.exists(), "Output file not created"

        result_df = pd.read_parquet(output_file)
        assert len(result_df) == 2, "Expected 2 rows"
        assert "total_rides" in result_df.columns, "Missing total_rides column"

    finally:
        os.chdir(original_cwd)
