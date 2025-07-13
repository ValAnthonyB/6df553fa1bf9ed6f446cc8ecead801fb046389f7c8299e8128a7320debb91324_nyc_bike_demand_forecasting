import pandas as pd
from datetime import datetime, date
from sklearn.preprocessing import SplineTransformer

def process_dataset(data_dir="../data/raw/"):
    df = (
        pd.read_parquet(data_dir, engine="pyarrow")
        .groupby("ride_date")
        .agg(total_rides=("unique_rides", "sum"))
        .reset_index()
        .pipe(lambda x: x[x["ride_date"] >= date(2023, 1, 1)])
        .sort_values("ride_date")
        .reset_index(drop=True)
    )

    df['ride_date'] = pd.to_datetime(df['ride_date'])
    
    return df


def feature_engineering(df):
    df = df.copy()

    # List of lags
    lags = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30, 54, 60]

    # Create lag features
    for lag in lags:
        df[f'lag-{lag}d'] = df['total_rides'].shift(lag)

    # Date-based features
    df['day_of_week'] = df['ride_date'].dt.dayofweek
    df['month_day'] = df['ride_date'].dt.day
    df['month'] = df['ride_date'].dt.month
    df['week_of_year'] = df['ride_date'].dt.isocalendar().week
    df['day_of_year'] = df['ride_date'].dt.dayofyear

    # Moving average features
    ma_windows = [3, 7, 14, 30]

    # Create rolling mean features including the current value
    for window in ma_windows:
        df[f'ma_{window}d'] = df['total_rides'].rolling(window=window).mean()

    # Make splines on the time-based features
    dow_spline = SplineTransformer(n_knots=10, degree=3, include_bias=False)
    dow_spline.fit(df[['month_day']])  # Fit only on train
    X_md_spline = dow_spline.transform(df[['month_day']])
    spline_cols = [f'month_day_spline_{i}' for i in range(X_md_spline.shape[1])]
    df[spline_cols] = X_md_spline

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

    # Target variable
    df['t+7d'] = df['total_rides'].shift(-7)

    # Drop rows with NaNs
    df = (
        df
        #.drop("total_rides", axis=1)
        .dropna()
    )

    return df

def split_train_test_data(df, cutoff_dt):

    df = df.copy()
    
    # Convert date string to pandas timestamp
    cutoff_dt = pd.Timestamp(cutoff_dt)

    # Time-based split
    train_df = df[df['ride_date'] <= cutoff_dt].copy()
    test_df  = df[(df['ride_date'] > cutoff_dt)].copy()

    return train_df, test_df


def get_features_labels(train_df, test_df):
    X_train = train_df.drop(["ride_date", "t+7d"], axis=1)
    y_train = train_df["t+7d"]

    X_test = test_df.drop(["ride_date", "t+7d"], axis=1)
    y_test = test_df["t+7d"]

    return X_train, X_test, y_train, y_test
