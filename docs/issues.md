1. Pointless "request handled" log message from middleware django.server logger should be enough
2. django.server log missing request_id it has "-" instead
3. Log messages seem oversaturated clean some of the contents
4. Inference latency logic inside inference view. Might not be the cleanest approach. Prefer to keep views for bussines logic only.
5. "X-Request-ID" header is not getting validated
6. Not fully convinced that inference latency must be part of the API response
7. Pick a better name for contract.py
8. There is no .env file + loaddotenv or similar alternative
9. A lot of magic string env vars, prefer settings.py or similar approach over magic strings
10. Gunicorn conf needs further review
11. Feature suggestion: Add make command that downloads the dataset
12. DRFSpectacular vs Default DRF openapi docs? Maybe simply use the built in one

Discovered by an AI Agent:
### P1 - Docker image can package private/local repo files

`Dockerfile:21` copies the entire build context into the image, and `.dockerignore:1-13` only excludes a small set of common build artifacts. It does not exclude the private issue-tracking document covered by `AGENTS.md`, nor local agent configuration directories such as `.codex/` and `.claude/`. Because `docker compose up --build` sends that context and the final image copies `/app` wholesale at `Dockerfile:38`, those files can end up in the built runtime image even though they are not application inputs.

Fix by tightening `.dockerignore` and/or replacing `COPY . .` with explicit copies for `arxiv_ml/`, `classifier/`, `config/`, `manage.py`, docs needed at runtime, and server config. At minimum, ignore the private document and local agent/tooling directories.

### P2 - Production defaults are unsafe for a production-oriented API

`config/settings.py:27-29` defaults to a known insecure `SECRET_KEY` and `ALLOWED_HOSTS="*"` while `DEBUG` defaults to false. `docker-compose.yml` also does not set `DJANGO_SECRET_KEY` or `DJANGO_ALLOWED_HOSTS`, so the provided production-ish path runs with those defaults. `manage.py check --deploy` also reports security warnings, including the insecure secret and missing security middleware.

For this Stage 1 deliverable, the image should fail fast when `DJANGO_SECRET_KEY` is missing in non-debug mode, and production/container docs should require explicit allowed hosts. Consider adding `SecurityMiddleware` and the low-cost response hardening settings as well; even for an API, this is expected baseline Django production posture.

### P2 - The required error-path test is missing

The plan requires pytest coverage for validation, happy path, error path, and health/ready. Current tests cover validation, happy path, deterministic stub behavior, request ID, and readiness. They do not force a classifier runtime failure or verify that an unexpected predictor error returns the standard envelope without leaking a traceback.

The autouse fixture in `classifier/tests/conftest.py:13-17` always installs a working `StubPredictor`, and `classifier/tests/test_predict.py:6-36` only exercises successful predictions. Add a test with a predictor double whose `predict()` raises, or reset the registry and post to `/api/v1/classify`, then assert the error envelope and no stack trace. This protects `classifier/exceptions.py:24-45`, which is part of the Stage 1 contract.

### P3 - Model startup runs during every Django management command

`classifier/apps.py:17-33` builds and warms the predictor in `AppConfig.ready()` for any Django process except the autoreloader parent. I confirmed `manage.py check` and `manage.py check --deploy` both load and warm the model. That is tolerable for the stub, but once Stage 2/3 artifacts land it makes routine commands and tests slow or dependent on model artifacts being present.

Keep the startup warm-up for actual serving processes, but add a guard or explicit startup path so non-serving management commands do not load the real model unnecessarily.
