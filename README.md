# NYC Daily Bike Demand Forecasting

## Project Overview 
[Citi Bike](https://citibikenyc.com/homepage) is a bike-sharing program in New York City, providing both classic pedal bikes and e-bikes for convenient, affordable, and fun transportation around the city. You can rent a bike using the Citi Bike app or the Lyft ride-hailing app. A rider can pick up a bike at one station and return it to any other station. The [Citi Bike NYC System Dataset](https://citibikenyc.com/system-data) contains trip records from the Citi Bike bicycle-sharing system in New York City. Each data point includes a unique trip identifier, trip duration, start and end times, station locations, the bike used, membership type, and many other fields. The dataset can be used for analyzing commuting patterns, bike usage, and urban mobility in New York City. The goal of this project is to train a model that predicts bike demand one week in advance. The data processing pipeline aggregates daily ride data to create a dataset containing each `ride_date` and `unique_rides` (the total number of rides for that date).

The pipeline was dockerized to prevent dependency conflicts by packaging all necessary components into a single container. Running the image through Docker ensures the ML pipeline is fully portable across different systems. Another important component of this project is Airflow orchestration. Unlike cron scheduling which takes a lot of manual work to manage pipeline dependencies, Airflow automates relationships between pipelines through DAGs. Airflow also offers features such as automatic retries and failure alerts to ensure pipeline reliability, distributed execution through executors like Celery for scaling tasks across worker nodes, and event-based triggers that can initiate workflows when files arrive in an Amazon S3 bucket.

The project uses a standard project folder structure but with added directories for Docker-related files and Airflow DAGs and logs.

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
├── Dockerfile                  # Dockerfile to run the pipeline
├── docker-compose.yml          # Launches airflow to run DAGs
└── README.md                   # Project overview and how to use
```


## Setup Instructions
To run the pipeline, please follow these steps:
1. Clone the repository.
2. CD to the directory.
3. Make sure Docker desktop is installed and opened.
4. Before running the pipeline, the datasets must pre prepared in the `data/raw` directory. 
4. Use `docker-compose up -d` to setup and launch Airflow.
5. Open `http://127.0.0.1:8080` in the browser to open the Airflow UI. Log into Airflow with the default username `admin` and the password `admin`.
6. To run the pipeline, click the `bike_forecasting_pipeline` and click the `Trigger DAG` button.


## Docker Integration
* The Python 3.10.10-slim environment was used for memory efficiency. 

* This run command 
```Bash
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
```
to install necessary extensions and tools for the pipeline to run. 

* The `WORKDIR /app` command sets up the container's working directory. 

* To install the project's dependencies, I used 
```Bash
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```
where the requirements.txt contains the libraries (taken from uv with specified library versions). I manually specified Tsinghua University's PyPI mirror for faster downloads since I experienced slow downloads during testing. 

* The `COPY src/ ./src/` was then used to copy the Python scripts in the `src/` directory to the container's `src/` directory. 

* The `RUN mkdir /app/models /app/reports` command was used to create the `models/` and `reports/` for storing the model artifact and the model metrics and results, respectively. 

* `VOLUME ["/app/data", "/app/models", "/app/reports"]` was used to create mount points for reading the data, exporting the Random Forest model, and the model performance results. 

* This was done so that the files exported by the pipeline is persisted. Finally, the `CMD ["python", "src/run_pipeline.py"]` command is used to run the pipeline. 

* I built the Docker image using:
```Bash
docker build -t 6df553fa1bf9ed6f446cc8ecead801fb046389f7c8299e8128a7320debb91324-ml-pipeline .
```

* To run the container I used this command:
```Bash
docker run --rm \
  -v "/$(pwd)/data:/app/data" \
  -v "/$(pwd)/models:/app/models" \
  -v "/$(pwd)/reports:/app/reports" \
6df553fa1bf9ed6f446cc8ecead801fb046389f7c8299e8128a7320debb91324-ml-pipeline
```
Ensure that Docker desktop is intalled. In my setup, I downloaded Docker desktop from the official website with version 28.3.2 and build number 578ccf6.


## Airflow DAG
The individual components of the pipeline was separated into Tasks in Airflow. Each task in the DAG use the scripts from the `src/`. The tasks of `pipeline_dag.py` are as follows:
1. `process_raw_data()` - Reads the raw datasets, concatenates them into a single dataframe, and exported as a parquet file in the `data/temp` directory.
2. `feature_engineering()` - Implements the feature engineering pipeline.
3. `date_splitting()` - Splits the feature-engineered dataset according to a date cut-off.
4. `model_training()` - Applies hyperparameter finetuning to a Random Forest model. Model artifact is exported to the `models/` directory.
5. `model_evaluation()` - Gets performance metrics of the trained model.
6. `temp_cleanup()` - Removes files stored in the temp folder. 

The tasks are run sequentially. No scheduling was implemented since the raw data arrives at unpredictable times in S3. Hence, we use Airflow's `S3KeySensor()` if this model is to be deployed into production.


## Reflection
I found this assignment extremely difficult to work on but it was worth it in the end. I became more familiar with creating Dockerfiles and docker-compose.yml. A major challenge I faced was that the original model was supposed to be a LightGBM model but due to incompatibility issues with the containerized environment of the docker-compose it was not possible despite trying multiple workarounds. Also, mounting the relevant directories proved to be a challenge since the pipeline in the Docker container and in the Airflow DAG kept failing unless I specifically mount directories needed by the workflow. All in all, this was an exciting yet difficult assignment and I am excited to incorporate GitHub actions and model monitoring in the next steps. 