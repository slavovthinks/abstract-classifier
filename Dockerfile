# syntax=docker/dockerfile:1

# Slim CPU image built with uv. NOTE: when the real model lands (Stage 2+), torch
# is added pinned to the CPU-only wheel index (extra-index-url
# https://download.pytorch.org/whl/cpu) so this image never pulls multi-GB CUDA
# wheels — see implementation-plan §8, §15.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install dependencies first (cached) using only the lockfiles.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Then install the project itself.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.14-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    DJANGO_SETTINGS_MODULE=config.settings \
    MODEL_BACKEND=stub

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app /app

USER appuser
EXPOSE 8000

CMD ["gunicorn", "--config", "gunicorn.conf.py", "config.wsgi:application"]
