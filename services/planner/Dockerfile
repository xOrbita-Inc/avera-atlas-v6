# services/planner/Dockerfile
# SCRUM-341: container hardening ported from SCRUM-213 APS planner.
# Changes from previous:
#   - Added COPY common/ and COPY config/ (latent gap, becomes real failure when 9.6 lands)
#   - Non-root user (planner)
#   - HEALTHCHECK polling /health
#   - curl added for HEALTHCHECK
#   - PYTHONUNBUFFERED and PYTHONDONTWRITEBYTECODE set
#   - ARTIFACTS_DIR env removed (platform planner is API-only, no shared volume write)

FROM python:3.11-slim

# Install curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY services/planner/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all runtime paths.
# common/ is dead code at runtime until 9.6 lands -- added now so the
# container does not fail the moment atlas_artifact integration merges.
# config/ is baked in so the default operator_policy_leo.yaml ships in
# the image and /ready can verify it loads without a volume mount.
# SCRUM-388: build context is the repository root so libs/ is reachable.
COPY libs/aps_math/ /app/aps_math/
# SCRUM-379: the decision state machine's validity seam calls the SCRUM-378
# assessor, so the planner image needs it. Only the two runtime modules and the
# package marker, not the service's tests or conftest.
COPY services/validity/__init__.py services/validity/wrapper.py services/validity/routing.py /app/validity/
COPY services/planner/avoid/   /app/avoid/
COPY services/planner/common/  /app/common/
COPY services/planner/config/  /app/config/
COPY services/planner/server.py .

# Non-root user -- principle of least privilege.
RUN useradd --no-create-home --shell /bin/false planner \
    && chown -R planner:planner /app
USER planner

EXPOSE 8060

# HEALTHCHECK: start period covers uvicorn startup and policy file load.
# interval/timeout/retries are conservative for a lightweight inference service.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8060/health || exit 1

CMD ["uvicorn", "server:svc", "--host", "0.0.0.0", "--port", "8060"]
