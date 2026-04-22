FROM python:3.11-slim

# Build metadata (required on tag builds)
ARG VERSION
ARG BUILD_TS

# Validate build args early (fail fast in CI)
RUN test -n "${VERSION}" || (echo "VERSION build arg is required" && exit 1)
RUN test -n "${BUILD_TS}" || (echo "BUILD_TS build arg is required" && exit 1)

LABEL org.opencontainers.image.title="PostgresAI Monitoring Flask Backend"
LABEL org.opencontainers.image.vendor="PostgresAI"
LABEL org.opencontainers.image.source="https://gitlab.com/postgres-ai/postgres_ai"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_TS}"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py promql_utils.py ./

# Embed build metadata into the image filesystem
RUN printf '%s' "${VERSION}" > /VERSION \
  && printf '%s' "${BUILD_TS}" > /BUILD_TS

# Expose port
EXPOSE 8000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "app:app"]
