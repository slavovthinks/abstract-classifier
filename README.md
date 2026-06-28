# abstract-classifier

A Django REST Framework API that classifies arXiv research-paper abstracts into
top-level category groups (`cs`, `math`, `physics`, `q-bio`, `q-fin`, `stat`,
`eess`, `econ`), backed by a swappable ML core.

The API is the deliverable; the model is loaded behind a clean contract so it can
be swapped (stub → TF-IDF → DistilBERT) by changing a single env var, with no
change to the web layer. See `docs/implementation-plan.md` for the full design and
`AGENTS.md` for the working agreement.

> **Status: Stage 1 (working vertical slice).** The service runs end-to-end with a
> dependency-free `StubPredictor` that returns a deterministic, defensibly-shaped
> response over the real label set. Real models land in Stages 2–3.

## Quick start

### Docker (recommended)

```bash
docker compose up --build
# wait for /ready to return 200, then:
curl -s -X POST localhost:8000/api/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{"abstract":"We present a transformer-based approach to graph learning."}'
```

### Local (uv)

```bash
make install          # uv sync
make run              # dev server (autoreload) on :8000
# or: make serve      # gunicorn (production server)
make test             # pytest
```

## API

| Method | Path                 | Description                                         |
|--------|----------------------|-----------------------------------------------------|
| POST   | `/api/v1/classify`   | Classify an abstract into a category group          |
| GET    | `/health`            | Liveness — always `200` while the process is up     |
| GET    | `/ready`             | Readiness — `200` once the model is loaded + warmed; `503` otherwise |

**`POST /api/v1/classify`**

```json
{ "abstract": "We present a transformer-based approach to..." }
```

```json
{
  "predicted_category": "cs",
  "confidence": 0.87,
  "top_k": [
    { "category": "cs",   "confidence": 0.87 },
    { "category": "stat", "confidence": 0.09 },
    { "category": "math", "confidence": 0.03 }
  ],
  "model_version": "stub-v1"
}
```

Validation errors return `400` with a consistent envelope (no stack traces leak):

```json
{ "error": { "code": "validation_error", "message": "Request validation failed.", "details": { "abstract": ["..."] } } }
```

## Configuration

Environment variables, parsed once into a framework-agnostic `InferenceConfig`:

| Variable             | Default     | Meaning                                            |
|----------------------|-------------|----------------------------------------------------|
| `MODEL_BACKEND`      | `stub`      | `stub` \| `tfidf` \| `distilbert`                  |
| `MODEL_SOURCE`       | `local`     | Artifact source (`local` now; `hf`/`s3` later)     |
| `MODEL_PATH`         | `artifacts` | Local artifact directory                           |
| `MODEL_VERSION`      | `stub-v1`   | Reported in responses for traceability             |
| `INFERENCE_DEVICE`   | `cpu`       | `cpu` \| `cuda` \| `auto` (CPU is the prod default)|
| `MAX_ABSTRACT_CHARS` | `20000`     | Reject inputs longer than this                     |
| `WEB_CONCURRENCY`    | `2`         | Gunicorn sync workers (each loads its own model)   |

## Design notes

- **Architecture.** `arxiv_ml/` is a standalone, framework-agnostic ML core (no
  Django/DRF imports) defining the `Predictor` contract; `classifier/` is a thin
  DRF adapter over it; `training/` (later stages) produces artifacts. Dependency
  direction is one-way — both depend on `arxiv_ml`, which depends on neither — so
  the web framework or the ML core can each be swapped/extracted independently.
- **Model lifecycle.** The predictor is built once at startup in `AppConfig.ready()`,
  warmed with one inference, and held as a process-wide singleton. `/ready` only
  flips to `200` after that warm-up, so traffic is never routed to a cold model.
- **Observability.** Structured JSON logging from the start: a request-ID filter
  (honoring an inbound `X-Request-ID`) plus per-request inference latency; full
  exceptions are logged server-side only.
- **Concurrency.** Inference is CPU-bound and PyTorch only partially releases the
  GIL, so we scale with a small number of sync gunicorn workers — see the tradeoff
  comment in `gunicorn.conf.py`.

## Roadmap

Stages 2–3 add the TF-IDF baseline and the fine-tuned DistilBERT model (drop-in via
the `MODEL_BACKEND` env var). Phase 2 polish (OpenAPI/Swagger, `/api/v1` versioning,
throttling) and the full model/metrics documentation follow. Phases 3–4 (Prometheus,
auth, CI, a dedicated inference server / model registry / GPU serving image) are
scoped in `docs/implementation-plan.md`.
