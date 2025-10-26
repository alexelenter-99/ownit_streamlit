FROM python:3.13-slim

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root

# 7. Copy the application code into the container
# This copies your `app` directory into the container's working directory
COPY ./app .

ENV PYTHONPATH="/app/src"

# 8. Command to run the application when the container launches
# Render sets the PORT environment variable automatically.
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}