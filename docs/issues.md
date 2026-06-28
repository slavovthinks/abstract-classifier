1. Pointless "request handled" log message from middleware, django.server logger should be enough
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
13. There a lot of comments and docstrings that are unneeded like explaining why a file sits in a specific place
Scan the repo for such and clean them. Keep only such that add real value. (AGENTS.md was instructed but not concrete enough for that)
14. Claude has the tendency to import from __future__ annotations which is unnecessary with 3.14 add that to the AGENTS.md
15. Im not sure that text.py place is in predictors/ maybe move it to the parent folder and keep predictors/ for ... predictors

Discovered by an AI Agent - Stage 1:
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

Discovered by an AI Agent stage 2:
### 1. `.dockerignore` blocks the documented "bake into image" path
`.dockerignore` excludes `artifacts` (and now `data`), and the `Dockerfile` only does
`COPY . .`. So the production "bake the artifact into the image at build" path described in
the compose comment and README/§13 is **not actually achievable today** — a build would
silently exclude `artifacts/tfidf/`. Stage 2 works because compose mounts `./artifacts:ro`
for dev, which is fine. Recommend either (a) explicitly note that baking requires
un-ignoring `artifacts/tfidf` (and add a scoped `COPY`/`!artifacts/tfidf` exception), or
(b) state outright that baking is deferred and dev-mount is the only wired path for now. As
written, the docs imply a capability that isn't there.

### 2. EDA length stats accumulate ~3M-element Python lists
`eda.py` appends `len(abstract)` and `len(abstract.split())` for every mapped record into
two growing Python lists (~3.08M elements each) before converting to NumPy. That's a few
hundred MB of transient memory — not a violation of "never load 4 GB," and acceptable for a
one-off EDA, but it does undercut the "streaming, never in memory" framing for the *length*
stats specifically. If you want it strictly bounded, preallocate a NumPy array, reservoir-
sample the lengths, or use a streaming percentile (e.g. `tdigest`). Low priority.

### 3. No automation copying generated plots → `docs/assets/`
EDA/metrics PNGs are written to the gitignored `artifacts/eda` and `artifacts/metrics`, but
the committed copies under `docs/assets/` (referenced by the README) appear to be placed
manually. `AGENTS.md` (which this change itself amends) says to *automate repeatable steps*;
a tiny `make publish-figures` (or a `--publish` flag) that copies the canonical PNGs into
`docs/assets/` would close the loop and prevent stale committed plots. Minor.

### 4. `TfidfPredictor` does not enforce `MAX_ABSTRACT_CHARS`
Truncation/rejection of oversized input lives only in the DRF serializer, so in serving the
predictor never sees an over-length abstract — fine in practice. Worth being aware that the
core predictor, used directly (e.g. in a script), has no length guard of its own. This is an
implicit coupling to the web layer for a protection the core could arguably own. No action
required for Stage 2.

### 5. Module-level imports after `matplotlib.use("Agg")`
`eda.py` and `evaluate.py` call `matplotlib.use("Agg")` and then `import matplotlib.pyplot`
mid-module. It's the correct way to force the non-interactive backend, but it will trip
`E402` (import-not-at-top) under flake8/ruff defaults. Cosmetic — suppress with a noqa or a
ruff config exception if you lint these.

### Low: TF-IDF artifact class order is trusted without validation

`TfidfPredictor` loads `labels.json` and zips it with `predict_proba()` output (`arxiv_ml/predictors/tfidf.py:22-31`). That is correct for the artifact currently produced by `train_baseline.py`, but the loader never checks that `labels.json` matches `pipeline.named_steps["clf"].classes_`.

If `labels.json` is stale, manually edited, or produced by a future artifact pipeline with a different order, the API will silently return wrong category labels with plausible confidence scores. Since this is the artifact boundary for serving, it should fail fast at startup when the label list differs from the classifier classes.

### Low: README dataset size is stale relative to the generated EDA counts

`README.md:93-94` says the Kaggle snapshot is `~2.5M records`, but the checked-in distribution table sums to 3,082,781 mapped records (`README.md:108-117`) before any dropped/unmapped records. This is minor, but it makes the model section look partially stale. Prefer deriving the total from EDA output or using a less exact phrase.
