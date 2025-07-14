# NYC Daily Bike Demand Forecasting
## Project Overview
[Citi Bike](https://citibikenyc.com/homepage) is bike-sharing program in New York City, providing both classic pedal bikes and e-bikes for convenient and affordable transportation around the city. You can rent a bike using the Citi Bike app or the Lyft app. Pick up a bike at one station and return it to any other station.

The [Citi Bike NYC System Dataset](https://citibikenyc.com/system-data) contains detailed trip records from the Citi Bike bicycle-sharing system in New York City. Each entry includes data such as a unique trip identifier, trip duration, start and end times, station locations, the bike used, and membership type. It is used for analyzing commuting patterns, bike usage, and urban mobility in New York City. The project uses a standard project folder structure:

```
.
├── data/               # Data files organized by processing stage
│   ├── raw/            # Original time series data files
│   └── processed/      # Cleaned and feature-engineered dataset
├── notebooks/          # Jupyter notebooks for EDA and modeling
├── models/             # Trained model artifacts
├── reports/            # Model performance results in .CSV files
├── src/                # Project scripts
│   ├── preprocessing/  # Data processing, engineering, and time-series splitting
│   ├── model_training/ # Automatically train the best lightgbm model
│   └── evaluation/     # Evaluates model performance
├── pyproject.toml      # Project dependencies
├── README.md           # Project overview and how to use
└── run_pipeline.py     # For running the pipeline
```

## How to Get the Data
Citi Bike stores their historical trip data in an [S3 bucket](https://s3.amazonaws.com/tripdata/index.html). The data was processed from a previous project by aggregating the daily total rides from January 2023 to June 2024, which are stored in individual parquet files in the `data/raw/` directory. The aggregation code to do the aggregation is in `notebooks/1 Processing time series data.ipynb`.

## Setup and Instructions
1. Installed Python from the [Python website](https://www.python.org/).
2. Installed uv using `pipx install uv`.
3. Setup Git Bash terminal in VS Code.
4. Used `uv init` in project directory.
5. Changed Python version from 3.12 to 3.10.18 in the pyproject.toml and .python-version files. Ran `uv sync` to ensure that we are using Python 3.10 in the python environment.
6. Ran `uv add numpy pandas pyarrow polars matplotlib scikit-learn xgboost lightgbm optuna jupyterlab joblib` to install essential packages.

To run the application use `uv run ./run_pipeline.py`.

## Pre-Commit Configuration
To enhance code quality, I used these two pre-commit hooks:
* **Ruff** - I used Ruff to automatically check for PEP8 violations, remove unused imports, flagging undefined variables, automatically sort import statements, and enforcing PEP8 naming standards. I specifically prevented Ruff from enforcing certain naming rules in function parameter names to accomodate for common machine learning naming conventions (e.g., `X_train`, `X_test`).

* **uv.lock** - Whenever the pyproject.toml file has changes, this pre-commit hook automatically synchronizes the dependencies in the uv.lock file. 


## Reflection
I found this assignment very fulfilling as a data scientist who is interested in transitioning to a machine learning engineer or AI engineer. To preface, I have zero experience in software engineering. Some of my models at work are stuck in Jupyter notebooks and they lack a simple .py script to operationalize the entire data pipeline. I enrolled in this subject to learn the best practices and use tools commonly used in operationalizing machine learning models. Since the first lecture, I have transitioned from using `poetry` to `uv`, incorporated pre-commit hooks, and I have been using VS code a lot more in my projects. I look forward to eventually incorporating new features like scheduling, docker containerization, Flask, and model monitoring to this project in the future!