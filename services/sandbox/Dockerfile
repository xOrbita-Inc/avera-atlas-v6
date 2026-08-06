FROM python:3.11-slim

WORKDIR /app

# httpx added SCRUM-373 AC2: sensor_knowledge.py's push_to_tracker() calls
# the live tracker /v1/observations endpoint.
RUN pip install --no-cache-dir pytest numpy httpx

COPY services ./services

ENV PYTHONPATH=/app

CMD ["python", "-m", "pytest", "services/sandbox/tests"]
