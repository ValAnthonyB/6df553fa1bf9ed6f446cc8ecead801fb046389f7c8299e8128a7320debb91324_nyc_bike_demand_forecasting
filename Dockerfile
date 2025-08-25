FROM python:3.10.18-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# We use a specific mirror since the default caused timeouts.
RUN pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
COPY src/ ./src/
RUN mkdir /app/models /app/reports /app/logs
VOLUME ["/app/data", "/app/models", "/app/reports", "/app/logs"]
RUN mkdir -p /mlflow/artifacts && \
    chmod 777 /mlflow/artifacts
CMD ["python", "src/run_pipeline.py"]
# docker build -t nyc_bike_demand_forecasting .
# docker run --rm \
#   -v "/$(pwd)/data:/app/data" \
#   -v "/$(pwd)/models:/app/models" \
#   -v "/$(pwd)/reports:/app/reports" \
#   -v "/$(pwd)/logs:/app/logs" \
# nyc_bike_demand_forecasting:latest
# docker exec -it 7f63c835fc72 bash