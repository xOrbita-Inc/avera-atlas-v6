FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .

# Create data directory
RUN mkdir -p /data/planner_artifacts

# Environment
ENV PORT=8000
ENV DATA_DIR=/data/planner_artifacts
ENV LOG_LEVEL=INFO

EXPOSE 8000

CMD ["python", "main.py"]
