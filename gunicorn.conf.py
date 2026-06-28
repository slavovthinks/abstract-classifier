"""Gunicorn configuration for serving the classifier API.

Inference is CPU-bound and PyTorch only partially releases the GIL, so a single
worker serializes requests. We therefore scale with a *small* number of sync
workers (see docs/implementation-plan.md §12).

Worker count vs. model memory is the key tradeoff: EACH worker loads its OWN copy
of the model (DistilBERT ≈ 250 MB resident). So total RAM ≈ workers × model size
plus overhead. Pick workers to fit the box, not the usual (2·CPU)+1 web heuristic
— that would multiply model memory needlessly. Override with WEB_CONCURRENCY.
"""

import os

bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:8000")
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
worker_class = "sync"

# Long enough for a cold first inference; short enough to recycle a stuck worker.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "60"))
graceful_timeout = 30

# Recycle workers periodically to bound any memory growth.
max_requests = int(os.environ.get("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = 100

# Each worker loads its own model in its own process (no shared CoW model state).
preload_app = False

accesslog = "-"
errorlog = "-"
