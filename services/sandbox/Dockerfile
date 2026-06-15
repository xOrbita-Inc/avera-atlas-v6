FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir pytest numpy

COPY services ./services

ENV PYTHONPATH=/app

CMD ["python", "-m", "pytest", "services/sandbox/tests"]