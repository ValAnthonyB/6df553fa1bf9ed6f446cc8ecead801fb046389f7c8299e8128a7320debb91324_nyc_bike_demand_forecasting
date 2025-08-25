from pathlib import Path

import numpy as np
import pandas as pd


def process_dataset(raw_data_dir: str) -> pd.DataFrame:
    """
    Loads Parquet files from the raw data folder, aggregates daily ride counts,
    filters dates from 2022 onwards, and ensures the date column is in datetime format.
    Parameters:
        raw_data_dir (str): Path to the raw data containing many parquet files.
    Returns:
        pd.DataFrame: DataFrame with 'ride_date' and 'total_rides' columns.
    """
    # Combine daily time series data from all raw parquet files
    data_path = Path(raw_data_dir)
    df = pd.read_parquet(data_path, engine="pyarrow")
    df["ride_date"] = pd.to_datetime(df["ride_date"])
    df = (
        df.groupby("ride_date")
        .agg(total_rides=("total_rides", "sum"))
        .reset_index()
        .pipe(lambda x: x[x["ride_date"] >= pd.Timestamp("2022-01-01")])
        .sort_values("ride_date")
        .reset_index(drop=True)
    )
    # Enforce ride_date to be datetime variable
    df["ride_date"] = pd.to_datetime(df["ride_date"])

    # Apply Gaussian noise to the total_rides
    df_drifted = df.copy()
    gaussian_noise = np.random.normal(0, 0.5 * df["total_rides"].std(), len(df_drifted))
    df_drifted["total_rides"] = df_drifted["total_rides"] + gaussian_noise

    return df, df_drifted


def split_train_test_data(
    df: pd.DataFrame, cutoff_dt: str, is_drifted: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    We use a cutoff date to split the training and test data using these rules:
    * Train set: ride_date <= cutoff date
    * Test set: ride_date > cutoff date

    Parameters:
    ----------
    df : pd.DataFrame
        Feature-engineered features and labels.

    cutoff_dt : str or pd.Timestamp
        Date used to split the data.

    Returns:
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        train_df: training set
        test_df: test set
    """

    df = df.copy()
    df["ride_date"] = pd.to_datetime(df["ride_date"])

    # Convert date string to pandas timestamp
    cutoff_dt = pd.Timestamp(cutoff_dt)

    # Time-based split
    train_df = df[df["ride_date"] <= cutoff_dt].copy()
    test_df = df[(df["ride_date"] > cutoff_dt)].copy()

    # Export to CSV
    if not is_drifted:
        train_df.to_csv("data/processed/train.csv", index=False)
        test_df.to_csv("data/processed/test.csv", index=False)

    else:
        train_df.to_csv("data/processed/drifted_train.csv", index=False)
        test_df.to_csv("data/processed/drifted_test.csv", index=False)

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
