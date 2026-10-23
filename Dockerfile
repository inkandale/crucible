FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH HF_HOME=/cache

WORKDIR /app

# Deps first so a source change does not re-resolve the environment.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY crucible ./crucible

# The bundle downloads here; the chart mounts an emptyDir over it.
RUN mkdir /cache && chown 1000:1000 /cache
USER 1000
EXPOSE 8080
ENTRYPOINT ["python", "-m", "crucible"]
