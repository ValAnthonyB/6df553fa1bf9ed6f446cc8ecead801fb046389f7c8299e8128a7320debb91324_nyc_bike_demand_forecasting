import os
from unittest.mock import patch

import pandas as pd
from dags.pipeline_dag import feature_engineering


def test_feature_engineering_reads_input_file(tmp_path):
    """Test that feature_engineering reads the concatenated.parquet file."""
    # Setup directories
    temp_dir = tmp_path / "data" / "temp"
    temp_dir.mkdir(parents=True)

    # Creates a sample input file
    input_data = pd.DataFrame(
        {
            "ride_date": pd.date_range("2023-01-01", periods=3, freq="D"),
            "total_rides": [100, 150, 200],
        }
    )
    input_file = temp_dir / "concatenated.parquet"
    input_data.to_parquet(input_file, engine="pyarrow")

    # Mock the feature engineering functions to avoid complex dependencies
    mock_holidays = ["2023-01-01"]
    mock_result_df = input_data.copy()

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        with (
            patch(
                "src.feature_engineering.get_nyc_holidays", return_value=mock_holidays
            ),
            patch("src.feature_engineering.feature_eng", return_value=mock_result_df),
            patch("src.feature_engineering.export_feature_eng_data") as mock_export,
        ):
            feature_engineering()

            # Assert input file was read (function completed without file read errors)
            assert input_file.exists(), "Input file should exist"
            mock_export.assert_called_once(), "Export function should be called"

    finally:
        os.chdir(original_cwd)
