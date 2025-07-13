from pathlib import Path
import pandas as pd
from datetime import date
from sklearn.preprocessing import SplineTransformer

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
    df['ride_date'] = pd.to_datetime(df['ride_date'])
    
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering on the time series dataframe.

    We use these feature engineering techniques
    1. Extract date-based features from ride_date (i.e. day of week, 
        day number of the month, month number, week number of the year)
    2. Extract lagged features across different time lags
    3. Compute moving-average features across different time lags
    4. Tagging of holidays
    5. Spline transformations on the day of month feature
    6. Creation of the target variable column

        Parameters:
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing the 'ride_date' (datetime) and 
        'total_rides' (integer) columns. 

    Returns:
    -------
    pd.DataFrame
        A transformed DataFrame with engineered features and target variable.
    """

    df = df.copy()

    # Date-based features
    df['day_of_week'] = df['ride_date'].dt.dayofweek
    df['month_day'] = df['ride_date'].dt.day
    df['month'] = df['ride_date'].dt.month
    df['week_of_year'] = df['ride_date'].dt.isocalendar().week
    df['day_of_year'] = df['ride_date'].dt.dayofyear

    # List of lags
    lags = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30, 54, 60]

    # Create lag features
    for lag in lags:
        df[f'lag-{lag}d'] = df['total_rides'].shift(lag)

    # Moving average features
    ma_windows = [3, 7, 14, 30]

    # Create rolling mean features including the current value
    for window in ma_windows:
        df[f'ma_{window}d'] = df['total_rides'].rolling(window=window).mean()

    # Holidays
    nyc_holidays = [
        '2023-01-01',  # New Year's Day
        '2023-01-02',  # New Year's Day (Observed)
        '2023-01-16',  # Martin Luther King Jr. Day
        '2023-02-12',  # Lincoln's Birthday
        '2023-02-20',  # Presidents' Day (Washington’s Birthday)
        '2023-05-29',  # Memorial Day
        '2023-06-19',  # Juneteenth
        '2023-07-04',  # Independence Day
        '2023-09-04',  # Labor Day
        '2023-10-09',  # Columbus Day
        '2023-11-07',  # Election Day
        '2023-11-10',  # Veterans Day (Observed)
        '2023-11-11',  # Veterans Day
        '2023-11-23',  # Thanksgiving
        '2023-12-25',  # Christmas Day

        '2024-01-01',  # New Year's Day
        '2024-01-15',  # Martin Luther King Jr. Day
        '2024-02-12',  # Lincoln's Birthday
        '2024-02-19',  # Presidents' Day
        '2024-05-27',  # Memorial Day
        '2024-06-19',  # Juneteenth
        '2024-07-04',  # Independence Day
        '2024-09-02',  # Labor Day
        '2024-10-14',  # Columbus Day
        '2024-11-05',  # Election Day
        '2024-11-11',  # Veterans Day
        '2024-11-28',  # Thanksgiving
        '2024-12-25',  # Christmas Day

        '2025-01-01',  # New Year's Day
        '2025-01-20',  # Martin Luther King Jr. Day
        '2025-02-12',  # Lincoln's Birthday
        '2025-02-17',  # Presidents' Day
        '2025-05-26',  # Memorial Day
        '2025-06-19',  # Juneteenth
        '2025-07-04',  # Independence Day
        '2025-09-01',  # Labor Day
        '2025-10-13',  # Columbus Day
        '2025-11-04',  # Election Day
        '2025-11-11',  # Veterans Day
        '2025-11-27',  # Thanksgiving
        '2025-12-25'   # Christmas Day
    ]

    # Convert holiday_dates to datetime
    holiday_dates = pd.to_datetime(nyc_holidays)
    df['is_holiday'] = df['ride_date'].isin(holiday_dates).astype(int)

    # Make splines on the time-based features
    dow_spline = SplineTransformer(n_knots=10, degree=3, include_bias=False)
    dow_spline.fit(df[['month_day']])  # Fit only on train
    X_md_spline = dow_spline.transform(df[['month_day']])
    spline_cols = [f'month_day_spline_{i}' for i in range(X_md_spline.shape[1])]
    df[spline_cols] = X_md_spline

    # Target variable
    df['t+7d'] = df['total_rides'].shift(-7)

    # Drop rows with NaNs
    df = (
        df
        #.drop("total_rides", axis=1)
        .dropna()
    )

    return df


def split_train_test_data(df: pd.DataFrame, cutoff_dt: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    train_df = df[df['ride_date'] <= cutoff_dt].copy()
    test_df  = df[(df['ride_date'] > cutoff_dt)].copy()

    return train_df, test_df


def get_features_labels(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame
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