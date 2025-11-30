
# ===== FILE 2: kafka-pipeline/Dockerfile =====
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts
COPY producer.py processor.py ./

# Default command (will be overridden by docker-compose)
CMD ["python", "producer.py"]
