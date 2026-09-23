# services/ui/Dockerfile
# SCRUM-447: build context is the repository root so libs/ is reachable, matching
# services/planner/Dockerfile and services/propagator/Dockerfile (SCRUM-388). The
# UI needs libs/aps_math for the shared two-body propagator the globe draws with.
FROM python:3.11-slim
WORKDIR /app
COPY services/ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY libs/aps_math/ /app/aps_math/
COPY services/ui/ /app/
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
