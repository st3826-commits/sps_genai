FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /usr/local/bin/uv

WORKDIR /code

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

RUN uv pip install --python /code/.venv/bin/python \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl


COPY app ./app
COPY models ./models
COPY helper_lib ./helper_lib

EXPOSE 8000

CMD ["/code/.venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]