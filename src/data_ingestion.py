import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests


def processing_time_series_data(csv_filename, csv_file_base, parquet_file_base):
    parquet_filename = csv_filename.split(".")[0] + ".parquet"

    # We only need these when we read the CSV file
    columns = ["ride_id", "started_at"]

    try:
        # Read CSV
        df = pd.read_csv(
            csv_file_base / csv_filename, engine="pyarrow", usecols=columns
        )

        # Convert datetime and add date column
        df["started_at"] = pd.to_datetime(df["started_at"])
        df["ride_date"] = df["started_at"].dt.strftime("%Y-%m-%d")

        # Aggregate to get the daily ridership
        df_agg = (
            df.groupby("ride_date")
            .agg(total_rides=("ride_id", "nunique"))
            .reset_index()
            .sort_values("ride_date")
        )

        # Save to parquet
        df_agg.to_parquet(
            parquet_file_base / parquet_filename, compression="gzip", index=False
        )
        print(f"{parquet_filename} processed")

    except Exception as e:
        print(f"Error processing {csv_filename}: {e}")


def download_process_data():
    # Setup directories. Make sure they exist
    temp_dir = Path("data/temp")
    unzipped_dir = temp_dir / "unzipped"
    processed_dir = Path("data/raw")
    for d in [temp_dir, unzipped_dir, processed_dir]:
        d.mkdir(exist_ok=True)

    # Files and URL
    files = [
        #'2022-citibike-tripdata.zip',
        "2023-citibike-tripdata.zip",
        # "202401-citibike-tripdata.zip",
        # "202402-citibike-tripdata.zip",
        # "202403-citibike-tripdata.zip",
        # "202404-citibike-tripdata.zip",
        # "202405-citibike-tripdata.zip",
        # "202406-citibike-tripdata.zip",
        # "202407-citibike-tripdata.zip",
        # "202408-citibike-tripdata.zip",
        # "202409-citibike-tripdata.zip",
        # "202410-citibike-tripdata.zip",
        # "202411-citibike-tripdata.zip",
        # "202412-citibike-tripdata.zip",
        # "202501-citibike-tripdata.zip",
        # "202502-citibike-tripdata.zip",
        # "202503-citibike-tripdata.zip",
        # "202504-citibike-tripdata.zip",
        # "202505-citibike-tripdata.zip",
        # "202506-citibike-tripdata.zip",
        # "202507-citibike-tripdata.zip"
    ]
    base_url = "https://s3.amazonaws.com/tripdata/"

    for filename in files:
        print(f"Downloading {filename}...")

        # Download the zip file
        success = False
        for attempt in range(10):
            response = requests.get(base_url + filename)
            if response.status_code == 200:
                success = True
                break
            print(f"Attempt {attempt + 1} failed for {filename}")

        if not success:
            print(f"Failed to download {filename} after 10 attempts")
            continue

        # Save the zip file
        zip_path = temp_dir / filename
        with open(zip_path, "wb") as f:
            f.write(response.content)

        # Extract the zip file to the temp folder
        print(f"Extracting main zip file: {filename}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(unzipped_dir)

        # Handle nested zip files (like in 2022/2023 data)
        for nested_zip in unzipped_dir.rglob("*.zip"):
            print(f"Found nested zip file: {nested_zip.name}")
            try:
                print(f"Attempting to extract: {nested_zip.name}")
                with zipfile.ZipFile(nested_zip, "r") as nested_zip_ref:
                    nested_zip_ref.extractall(nested_zip.parent)
                print(f"Successfully extracted: {nested_zip.name}")
                nested_zip.unlink()

            except zipfile.BadZipFile:
                print(f"ERROR: {nested_zip.name} is not a valid zip file - skipping")
                continue

            except Exception as e:
                print(f"ERROR extracting {nested_zip.name}: {e}")
                continue

        # Process CSV files
        print("Processing CSV files...")
        for csv_file in unzipped_dir.rglob("*.csv"):
            print(f"Processing CSV: {csv_file.name}")
            processing_time_series_data(csv_file.name, csv_file.parent, processed_dir)

        # Clean up all extracted content after processing
        for item in unzipped_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

        # Delete the zip file
        zip_path.unlink()


if __name__ == "__main__":
    download_process_data()

    # Delete the contents of the data/temp folder
    temp_dir = Path("data/temp")
    [f.unlink() for f in temp_dir.glob("*") if f.is_file()]
