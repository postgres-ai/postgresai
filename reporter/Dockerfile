FROM python:3.11-slim

ARG VERSION
RUN test -n "${VERSION}" || (echo "VERSION build arg is required" && exit 1)
ARG BUILD_TS
RUN test -n "${BUILD_TS}" || (echo "BUILD_TS build arg is required" && exit 1)

LABEL org.opencontainers.image.title="PostgresAI Reporter"
LABEL org.opencontainers.image.description="Automated Postgres health check and monitoring reports"
LABEL org.opencontainers.image.vendor="PostgresAI"
LABEL org.opencontainers.image.source="https://github.com/PostgresAI/postgres-ai-monitoring"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_TS}"

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the full reporter package (postgres_reports.py imports reporter.*)
RUN mkdir -p /app/reporter
COPY . /app/reporter

# The AWS RDS CA bundle at the path the RDS collector's conn_str names
# (sslrootcert=/etc/ssl/certs/rds-global-bundle.pem): the reporter connects to the
# node itself for catalog facts (A005, D001, D004, G003). A host bind mount still
# overrides it. Source: https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem
# (refresh when AWS adds a region or rotates its roots).
COPY certs/rds-global-bundle.pem /etc/ssl/certs/rds-global-bundle.pem

# Make script executable
RUN chmod +x /app/reporter/postgres_reports.py

# Embed build metadata into the image filesystem
RUN printf '%s' "${VERSION}" > /VERSION \
  && printf '%s' "${BUILD_TS}" > /BUILD_TS

# Create reports directory
RUN mkdir -p /app/reports

# Default command
CMD ["python", "-m", "reporter.postgres_reports"]
