FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir ".[orchestration,versioning]"
COPY scripts ./scripts
ENV MLOPS_PROJECT_ROOT=/app MLOPS_HOME=/runtime
CMD ["uvicorn","mlops_platform.serving.main:app","--host","0.0.0.0","--port","8200"]
