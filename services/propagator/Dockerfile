FROM python:3.11-slim
WORKDIR /app
COPY services/propagator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# SCRUM-388: build context is the repository root so libs/ is reachable.
# A Docker build cannot COPY from outside its context, which is why the
# context moved from ./services/propagator to . in docker-compose.
COPY libs/aps_math/ /app/aps_math/
COPY services/propagator/main.py .
ENV DATA_DIR=/data/planner_artifacts
CMD ["python","-u","main.py"]
