# NYC Daily Bike Demand Forecasting
The [Citi Bike NYC System Dataset](https://citibikenyc.com/system-data)
 contains detailed trip records from the Citi Bike bicycle-sharing system in New York City. Each entry includes data like trip duration, start and end times, station locations, bike ID, and rider demographics (if available). It is used for analyzing commuting patterns, bike usage, and urban mobility. The data is already processed from a previous project by aggregating the daily total rides from January 2023 to June 2024 stored in individual parquet files and is stored in the `data/raw/` directory. The project uses a standard project folder structure:

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

## Project Overview

## How to Get the Data

## Setup and Instructions
1. Installed Python from the [Python website](https://www.python.org/) 
2. Installed uv using `pipx install uv`
3. Setup Git Bash terminal in VS Code
4. Used `uv init` in project directory
5. Changed Python version from 3.12 to 3.10.18 in the pyproject.toml and .python-version files. Ran `uv sync` to ensure that we are using Python 3.10 in the python environment.
6. Ran `uv add numpy pandas pyarrow polars matplotlib scikit-learn xgboost lightgbm optuna jupyterlab joblib` to install essential packages.

To run the application use `uv run ./run_pipeline.py`.


## Reflection

## Pre-Commit Configuration