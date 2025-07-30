# NYC Daily Bike Demand Forecasting
## Project Overview
[Citi Bike](https://citibikenyc.com/homepage) is a bike-sharing program in New York City, providing both classic pedal bikes and e-bikes for convenient, affordable, and fun transportation around the city. You can rent a bike using the Citi Bike app or the Lyft ride-hailing app. A rider can pick up a bike at one station and return it to any other station.

The [Citi Bike NYC System Dataset](https://citibikenyc.com/system-data) contains trip records from the Citi Bike bicycle-sharing system in New York City. Each data point includes a unique trip identifier, trip duration, start and end times, station locations, the bike used, membership type, and many other fields. The dataset can be used for analyzing commuting patterns, bike usage, and urban mobility in New York City. 

I originally used this dataset late last year to learn new techniques in time series forecasting using gradient boosting models, while also studying MLOps best practices on the side. It is a good dataset for learning time series forecasting since it is a real-world dataset and Citi Bike regularly uploads new data on a monthly basis, so there is an opportunity to incorporate data drift methods and deploy it in the cloud. 

## How to Get the Data
Citi Bike stores their historical trip data in an [S3 bucket](https://s3.amazonaws.com/tripdata/index.html). The data in `data/raw/` was previously processed from a past project of mine by aggregating the daily total rides from January 2023 to June 2024, which are stored in individual parquet files in the `data/raw/` directory. The aggregation code to do the aggregation is in `notebooks/1 Processing time series data.ipynb`.

## Folder Structure
The project uses a standard project folder structure:

```
.
├── data/                       # Data files organized by processing stage
│   ├── raw/                    # Original time series data files
│   └── processed/              # Cleaned and feature-engineered dataset
├── notebooks/                  # Jupyter notebooks for exploration and prototyping
├── src/                        # Project scripts
│   ├── data_processing.py      # Data pre-processing
│   ├── feature_engineering.py  # Applies feature engineering
│   ├── training.py             # Automatically train the best lightgbm model
│   ├── evaluate_model.py       # Evaluates model performance
│   └── run_pipeline.py         # Executes the entire data pipeline
├── models/                     # Trained model artifacts
├── reports/                    # Model performance results in .CSV files
├── deploy/                     # Data files organized by processing stage
│   ├── airflow/                # Directory for storing airflow-related objects
│       ├── dags/               # Airflow dags
│       ├── logs/               # Stores logs of dag runs
│       └── config/             # Configuration files.
│   └── docker/                 # Files related to docker
├── pyproject.toml              # Project dependencies from uv environment
├── requirements.txt            # Project dependencies
└── README.md                   # Project overview and how to use
```

## Setup Instructions
I used these to setup the uv environment, project directory and project files:
1. Installed Python from the [Python website](https://www.python.org/) on my Windows 11 machine.
2. Installed `uv` with `pipx install uv`.
3. Used `uv init` to automatically generate the README.md, pyproject.toml, and other configuration files.
4. Switched the environment's Python version from 3.12 to 3.10.18 in the pyproject.toml and .python-version files. Ran `uv sync` to ensure that we are using Python 3.10 in the python environment.
5. Ran `uv add numpy pandas pyarrow scikit-learn lightgbm optuna joblib` to install essential packages. I initially used other packages like polars, matplotlib, XGBoost, and Jupyter Lab for exploration but these are not needed in `run_pipeline.py`.
6. Used `uv pip freeze > requirements.txt` to export the requirements.txt file.

In order to run the pipeline, please use the instructions below:
1. Clone or fork the repository.
2. Use `uv sync` to automatically install the dependencies from the pyproject.toml file.
3. Finally, to run the data pipeline, first go to the src folder using `cd src` and then run the pipeline using `uv run ./run_pipeline.py`.


ride_date	total_rides

## Pre-Commit Configuration
To enhance code quality, I implemented pre-commit hooks to my environment. I used `uv add pre-commit` to install the pre-commit package, created a `.pre-commit-config.yaml` file, and ran `pre-commit install` to implement my pre-commit hooks. I used these two pre-commit hooks:
* **Ruff** - I used Ruff to automatically check for PEP8 violations, remove unused imports, flagging undefined variables, automatically sort import statements, and enforcing PEP8 naming standards. I specifically prevented Ruff from enforcing certain naming rules in function parameter names to accomodate for common machine learning naming conventions (e.g., `X_train`, `X_test`).

* **uv.lock** - Whenever the pyproject.toml file has changes, this pre-commit hook automatically synchronizes the dependencies in the uv.lock file. 

# Containerization
1. Installed Docker desktop from the official website. Docker version 28.3.2, build 578ccf6
2. I opened Docker desktop then created the `Dockerfile` at the root directory. 
3. I built the Docker image using
```Bash
docker build -t 6df553fa1bf9ed6f446cc8ecead801fb046389f7c8299e8128a7320debb91324-ml-pipeline .
```
4. Lastly, I ran the containerized pipeline in Git Bash using 
```Bash
docker run --rm \
  -v "/$(pwd)/data:/app/data" \
  -v "/$(pwd)/models:/app/models" \
  -v "/$(pwd)/reports:/app/reports" \
6df553fa1bf9ed6f446cc8ecead801fb046389f7c8299e8128a7320debb91324-ml-pipeline
```

# Airflow DAGs for Containerization
1. Used docker-compose.yml to install Airflow with the most basic functionalities.
2. Had trouble with mounting data/, /models, reports/, and src/ directories to Airflow's docker container.
3. Had to install libraries using requirements.txt.
4. Had to setup the credentials manually
5. Docker compose is limited so cannot use parallelization in training RF.
6. Installation difficulty with lightgbm due to OS-specific dependencies had to switch to Random Forest.

# PyTests
Testing: pytest tests/test_process_raw_data.py -v