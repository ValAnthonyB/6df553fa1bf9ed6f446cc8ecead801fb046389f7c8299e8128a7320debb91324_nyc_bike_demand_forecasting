# NYC Daily Bike Demand Forecasting

## Project Overview 
[Citi Bike](https://citibikenyc.com/homepage) is a bike-sharing program in New York City, providing both classic pedal bikes and e-bikes for convenient, affordable, and fun transportation around the city. You can rent a bike using the Citi Bike app or the Lyft ride-hailing app. A rider can pick up a bike at one station and return it to any other station. The [Citi Bike NYC System Dataset](https://citibikenyc.com/system-data) contains trip records from the Citi Bike bicycle-sharing system in New York City. Each data point includes a unique trip identifier, trip duration, start and end times, station locations, the bike used, membership type, and many other fields. The dataset can be used for analyzing commuting patterns, bike usage, and urban mobility in New York City. The goal of this project is to train a model that predicts bike demand one week in advance. The data processing pipeline aggregates daily ride data to create a dataset containing each `ride_date` and `total_rides` (the total number of rides for that date).

The project uses a standard project folder structure but with added directories for Docker-related files, Airflow DAGs, and MLflow runs.

```
.
├── data/                       
│   ├── raw/                    # Original time series data files
│   └── processed/              # Cleaned and feature-engineered dataset
├── src/                        
│   ├── data_ingestion.py       # Downloads and initial processing of the raw data
│   ├── data_processing.py      # Data pre-processing
│   ├── feature_engineering.py  # Applies feature engineering
│   ├── training.py             # Automatically train the best random forest model
│   ├── evaluate_model.py       # Evaluates model performance
│   ├── drift_detection.py      # Detects data drift in the features using evidently
│   └── run_pipeline.py         # Executes the entire data pipeline
├── models/                     # Trained model artifacts
├── reports/                    # Store model performance and drift detection results
├── deploy/                     # Data files organized by processing stage
│   ├── airflow/                # Directory for storing Airflow-related objects
│       ├── dags/               # Airflow dags
│       ├── logs/               # Stores logs of DAG runs
│       └── config/
├── mlflow/                     # Store MLflow-related files
├── pyproject.toml
├── requirements.txt
├── docker-compose.yml          # Launches Airflow and MLflow
├── Dockerfile.mlflow           # Dockerfile to create the MLflow image
├── Dockerfile                  # Dockerized ml pipeline
└── README.md
```


## Setup Instructions
To run the pipeline, please follow these steps:
1. Clone the repository.
2. Navigate to the project directory.
3. Ensure Docker Desktop is installed and running.
4. Run `uv sync` to install the project packages specified in the pyproject.toml file.
5. Before running the pipeline, prepare the datasets in the `data/raw` directory.
6. Use `docker-compose up --build` to set up and launch Apache Airflow and MLflow.
7. Open the following URLs in your browser:
  * Airflow UI: `http://127.0.0.1:8080`
  * MLflow UI: `http://127.0.0.1:5000`
Log into Airflow using the default credentials:
  * Username: admin
  * Password: admin
8. To run the pipeline in Airflow, click on the ml_pipeline_dag DAG and then click the Trigger DAG button.

## MLFlow Integration
MLflow is a tool for managing the end-to-end machine learning lifecycle, including experiment tracking, model registry, versioning, and deployment. It offers an organized way to log models and performance metrics, enabling data scientists to compare different approaches, reproduce successful experiments, and maintain a clear history of their work. MLflow also helps in model deployment by providing a standardized format for packaging models and offering multiple deployment options for batch and online inference.


## Model Drift Detection
To simulate model drift, the original test set was modified by adding Gaussian noise, creating a new dataset that represents drifted data. All of the features in the dataset are numerical and Evidently automatically compares the reference dataset and the drifted dataset to create a report. In this project, if at least one feature has exhibited statistically significant drift (p-value < .10) then retraining on fresh new data (original training data with the same Gaussian noise) is required. The retrained model is logged to MLflow afterwards.


## Folder Structure
To simulate model drift, the original test set was modified by adding Gaussian noise, creating a new dataset that represents drifted data. Since all features in the dataset are numerical, Evidently automatically compares the reference dataset against the drifted dataset and generates a comprehensive drift report. In this project, if at least one feature exhibits statistically significant drift (p-value < 0.10), the model is retrained using fresh data consisting of the original training set with the same Gaussian noise applied.

## Testing Instructions
* Use `uv run python src/run_pipeline.py` to run the pipeline script. Before running this command, make sure that the environment is activated and MLflow is launched from the docker-compose file.

* To run the model pipeline in Airflow, use `docker-compose exec airflow-webserver airflow dags test ml_pipeline_dag 2025-08-02` in the terminal. Docker Desktop must be running for this to work.

* To verify that MLflow is working, run `curl http://localhost:5000` in the terminal. The output should look like this:

```
StatusCode        : 200
StatusDescription : OK
Content           : <!doctype html><html lang="en"><head><meta charset="utf-8"/><meta name="viewport"
                    content="width=device-width,initial-scale=1,shrink-to-fit=no"/><link rel="shortcut icon"
                    href="./static-files/favicon....
RawContent        : HTTP/1.1 200 OK
                    Connection: close
                    Content-Disposition: inline; filename=index.html
                    Content-Length: 645
                    Cache-Control: no-cache
                    Content-Type: text/html; charset=utf-8
                    Date: Mon, 25 Aug 2025 13:23...
Forms             : {}
Headers           : {[Connection, close], [Content-Disposition, inline; filename=index.html], [Content-Length, 645],
                    [Cache-Control, no-cache]...}
Images            : {}
InputFields       : {}
Links             : {}
ParsedHtml        : mshtml.HTMLDocumentClass
RawContentLength  : 645
```

## Reflection
This homework was challenging for me mainly due to the docker-compose setup. I encountered weird issues with Airflow 3.0.3 not being able to interact with MLflow to log model artifacts. To remedy this issue, I downgraded to Airflow 2.10, and the problem stopped occurring. I was then able to create DAGs in Airflow and simultaneously log model experiments and metrics to MLflow smoothly. This assignment is very important for understanding model deployment in industry, since ML models are expected to decline in performance over time. With this repo, I am positive that I could implement model drift detection at work easily.