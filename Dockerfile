FROM python:3.10.18-slim

RUN apt-get update && apt-get install -y --no-install-recommends \ 
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# We use a specific mirror since the default caused timeouts.
RUN pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY src/ ./src/

RUN mkdir /app/models /app/reports

VOLUME ["/app/data", "/app/models", "/app/reports"]

CMD ["python", "src/run_pipeline.py"]