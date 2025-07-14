from datetime import date
from pathlib import Path

import pandas as pd


def process_dataset(raw_data_dir: str) -> pd.DataFrame:
    """
    Loads Parquet files from the raw data folder, aggregates daily ride counts,
    filters dates from 2023 onwards, and ensures the date column is in datetime format.

    Parameters:
        raw_data_dir (str): Path to the raw data containing many parquet files.

    Returns:
        pd.DataFrame: DataFrame with 'ride_date' and 'total_rides' columns.
    """

    # Combine daily time series data from all raw parquet files
    data_path = Path(raw_data_dir)
    df = (
        pd.read_parquet(data_path, engine="pyarrow")
        .groupby("ride_date")
        .agg(total_rides=("unique_rides", "sum"))
        .reset_index()
        .pipe(lambda x: x[x["ride_date"] >= date(2023, 1, 1)])
        .sort_values("ride_date")
        .reset_index(drop=True)
    )

    # Enforce ride_date to be datetime variable
    df["ride_date"] = pd.to_datetime(df["ride_date"])

    return df


def split_train_test_data(
    df: pd.DataFrame, cutoff_dt: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    We use a cutoff date to split the training and test data using these rules:
    * Train set: ride_date <= cutoff date
    * Test set: ride_date > cutoff date

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the feature-engineered features and labels.

    cutoff_dt : str or pd.Timestamp
        Date used to split the data.

    Returns:
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        train_df: training set
        test_df: test set
    """

    df = df.copy()

    # Convert date string to pandas timestamp
    cutoff_dt = pd.Timestamp(cutoff_dt)

    # Time-based split
    train_df = df[df["ride_date"] <= cutoff_dt].copy()
    test_df = df[(df["ride_date"] > cutoff_dt)].copy()

    return train_df, test_df


def get_features_labels(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract features and labels from training and testing DataFrames.

    Parameters:
    ----------
    train_df : pd.DataFrame
        Training data with features and target column.

    test_df : pd.DataFrame
        Testing data with features and target column.

    Returns:
        X_train : pd.DataFrame
            Training features

        X_test : pd.DataFrame
            Testing features

        y_train : pd.Series
            Training target values

        y_test : pd.Series
            Testing target values
    -------
    tuple:
    """
    X_train = train_df.drop(["ride_date", "t+7d"], axis=1)
    y_train = train_df["t+7d"]

    X_test = test_df.drop(["ride_date", "t+7d"], axis=1)
    y_test = test_df["t+7d"]

    return X_train, X_test, y_train, y_test
