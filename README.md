# NYC Bike_Demand_Forecasting
The [Citi Bike NYC System Dataset](https://citibikenyc.com/system-data)
 contains detailed trip records from the Citi Bike bicycle-sharing system in New York City. Each entry includes data like trip duration, start and end times, station locations, bike ID, and rider demographics (if available). It is used for analyzing commuting patterns, bike usage, and urban mobility. The data is already processed by aggregating the daily total rides from January 2023 to June 2024 stored in individual parquet files.

```
.
├── data/               # Raw and processed data files
├── notebooks/          # Jupyter notebooks for EDA and modeling
├── models/             # Trained model artifacts
├── src/                # Source code for data processing and training
├── pyproject.toml      # Project dependencies
└── README.md           # Project overview and usage guide
```

## uv Installation and Setup
* Installed Python from Microsoft Store
* Installed uv using `pipx install uv`
* Setup Git Bash terminal in VS Code
* Used `uv init` in project directory
* Changed Python version from 3.12 to 3.10 in the pyproject.toml and .python-version files. Ran `uv sync` to ensure that we are using Python 3.10 in the python environment.
* Ran `uv add numpy pandas polars matplotlib scikit-learn xgboost jupyterlab` to install essential packages.