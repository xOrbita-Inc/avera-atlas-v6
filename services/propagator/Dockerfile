FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY pc_utils.py .
ENV DATA_DIR=/data/planner_artifacts
CMD ["python","-u","main.py"]
