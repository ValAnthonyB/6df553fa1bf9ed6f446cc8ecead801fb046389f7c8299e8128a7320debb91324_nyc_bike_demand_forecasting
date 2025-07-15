import holidays
import pandas as pd
from sklearn.preprocessing import SplineTransformer


def get_nyc_holidays(yr_min: int = 2023, yr_max: int = 2026):
    """
    Automatically get the list of holidays in New York State.

    Parameters:
    ----------
    yr_min : int
        Starting year for scanning holidays.

    yr_min : int
        End year for scanning holidays (exclusive).

    Returns:
    -------
    list
        A list of dates in string format (YYYY-MM-DD).
    """

    nyc_holidays = holidays.US(state="NY", years=range(yr_min, yr_max))

    # Convert to list of date strings in YYYY-MM-DD format
    nyc_holidays = [date.strftime("%Y-%m-%d") for date in nyc_holidays.keys()]

    return sorted(nyc_holidays)


def feature_eng(
    df: pd.DataFrame, nyc_holidays: list[str], export_dataset: bool = True
) -> pd.DataFrame:
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

    nyc_holidays : list of str
        List of holidays in New York City.

    Returns:
    -------
    pd.DataFrame
        A transformed DataFrame with engineered features and target variable.
    """

    df = df.copy()

    # Date-based features
    df["day_of_week"] = df["ride_date"].dt.dayofweek
    df["month_day"] = df["ride_date"].dt.day
    df["month"] = df["ride_date"].dt.month
    df["week_of_year"] = df["ride_date"].dt.isocalendar().week
    df["day_of_year"] = df["ride_date"].dt.dayofyear

    # List of lags
    lags = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30, 54, 60]

    # Create lag features
    for lag in lags:
        df[f"lag-{lag}d"] = df["total_rides"].shift(lag)

    # Moving average features
    ma_windows = [3, 7, 14, 30]

    # Create rolling mean features including the current value
    for window in ma_windows:
        df[f"ma_{window}d"] = df["total_rides"].rolling(window=window).mean()

    # Convert holiday_dates to datetime
    holiday_dates = pd.to_datetime(nyc_holidays)
    df["is_holiday"] = df["ride_date"].isin(holiday_dates).astype(int)

    # Make splines on the time-based features
    dow_spline = SplineTransformer(n_knots=10, degree=3, include_bias=False)
    dow_spline.fit(df[["month_day"]])  # Fit only on train
    X_md_spline = dow_spline.transform(df[["month_day"]])
    spline_cols = [f"month_day_spline_{i}" for i in range(X_md_spline.shape[1])]
    df[spline_cols] = X_md_spline

    # Target variable
    df["t+7d"] = df["total_rides"].shift(-7)

    # Drop rows with NaNs
    df = (
        df
        # .drop("total_rides", axis=1)
        .dropna()
    )

    return df


def export_feature_eng_data(df: pd.DataFrame, export_dir: str) -> None:
    """
    Exports the feature engineered dataset to the processed directory in parquet format.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the feature-engineered dataset.

    export_dir : str
        Path to export the feature-engineered dataset.
    """

    # Export dataset
    filename = f"{export_dir}/feature_engineered_data.parquet"
    df.to_parquet(filename, compression="gzip", index=False)

    print(f"Exported feature-engineered data to {export_dir}")
