"""Django settings — a thin host for the DRF adapter over ``arxiv_ml``.

Environment variables are parsed once here into a framework-agnostic
:class:`arxiv_ml.config.InferenceConfig`, which is the only thing the ML core
sees. This keeps ``arxiv_ml`` free of any Django coupling.
"""

import os
from pathlib import Path

from arxiv_ml.config import InferenceConfig
from arxiv_ml.enums import InferenceDevice, ModelBackend, ModelSourceKind

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- Core Django -----------------------------------------------------------

SECRET_KEY = _env("DJANGO_SECRET_KEY", "insecure-dev-key-change-in-production")
DEBUG = _env_bool("DJANGO_DEBUG", False)
ALLOWED_HOSTS = [h for h in _env("DJANGO_ALLOWED_HOSTS", "*").split(",") if h]

INSTALLED_APPS = [
    "rest_framework",
    "classifier",
]

MIDDLEWARE = [
    "classifier.middleware.RequestIDMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# No relational data is used; a throwaway in-memory SQLite keeps Django happy.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    }
]

USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- DRF -------------------------------------------------------------------

REST_FRAMEWORK = {
    "EXCEPTION_HANDLER": "classifier.exceptions.exception_handler",
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
    "UNAUTHENTICATED_USER": None,
}

# --- ML inference config (the seam handed to arxiv_ml) ---------------------

# Each backend resolves its own artifact path/version so that selecting it is a
# single env var (MODEL_BACKEND). MODEL_PATH / MODEL_VERSION still override when
# pointing at a non-standard artifact (e.g. a freshly retrained `artifacts/tfidf`).
# TFIDF defaults to the committed `pretrained/tfidf` so a fresh clone (or built
# image) runs the real model with no training step and no mounted volume.
_BACKEND_DEFAULTS = {
    ModelBackend.STUB: (BASE_DIR / "artifacts", "stub-v1"),
    ModelBackend.TFIDF: (BASE_DIR / "pretrained" / "tfidf", "tfidf-arxiv-v1"),
    ModelBackend.DISTILBERT: (BASE_DIR / "artifacts" / "distilbert", "distilbert-arxiv-v1"),
}

_model_backend = ModelBackend(_env("MODEL_BACKEND", ModelBackend.STUB))
_default_path, _default_version = _BACKEND_DEFAULTS[_model_backend]

INFERENCE_CONFIG = InferenceConfig(
    model_backend=_model_backend,
    model_source=ModelSourceKind(_env("MODEL_SOURCE", ModelSourceKind.LOCAL)),
    model_path=Path(_env("MODEL_PATH", str(_default_path))),
    model_version=_env("MODEL_VERSION", _default_version),
    inference_device=InferenceDevice(_env("INFERENCE_DEVICE", InferenceDevice.CPU)),
    max_abstract_chars=int(_env("MAX_ABSTRACT_CHARS", "20000")),
)

# --- Logging (structured JSON from the start) ------------------------------

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "request_id": {"()": "classifier.observability.RequestIDFilter"},
    },
    "formatters": {
        "json": {"()": "classifier.observability.JSONFormatter"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "filters": ["request_id"],
        },
    },
    "root": {"handlers": ["console"], "level": _env("LOG_LEVEL", "INFO")},
    "loggers": {
        "django": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "classifier": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}
