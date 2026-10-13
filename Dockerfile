FROM python:3.11-slim

WORKDIR /app

# Install build deps and project requirements
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt google-cloud-storage>=2.10

# Copy source + config + prompts + the GCS wrapper script
COPY src/ ./src/
COPY config/ ./config/
COPY prompts/ ./prompts/
COPY scripts/ ./scripts/

# Install the llm-eval-forge package itself
RUN pip install --no-cache-dir -e .

# The container entrypoint runs the eval and uploads the report to GCS.
# RESULTS_BUCKET env var must be set on the Cloud Run Job.
ENTRYPOINT ["python", "scripts/run_and_upload.py"]
