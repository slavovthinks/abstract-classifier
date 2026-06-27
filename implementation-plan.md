# Implementation Plan: arXiv Abstract Classifier — Production-Oriented API

## How to read this document

This is a decision-complete spec, concrete enough that **independent agents can each
take a workstream (§11) and build it without further clarification.** The technical
choices below are **already made** — do not re-litigate them, just execute. Where
rationale is given, it's so you make consistent downstream choices. If you hit a
genuine fork not covered here, prefer the option that makes the **API more
production-ready** (the graded deliverable) and is consistent with `AGENTS.md`.

Current state: **greenfield, nothing built yet.**

---

## 1. Context & objective

This is a take-home interview task. The nominal task is "classify research articles
into categories." The reviewer explicitly said: **do not focus on the model — make the
API as production-ready as possible.** Time budget is a single weekend.

Therefore the priorities are inverted from a typical ML task:

- **The model is a checkbox.** It must exist, load, and return a defensible label. It
  does not need to be tuned or state-of-the-art.
- **The API is the deliverable.** This is where engineering quality is judged.

The owner is a senior backend engineer. Treat the API code to that standard. Avoid
toy patterns; avoid over-building past what a weekend allows (see phases in §8).

**Working philosophy:** ship a working end-to-end slice first, then improve in
independently-shippable stages (§10). The contract between model and API (§5) is defined
first so both can be built in parallel.

---

## 2. The dataset & label scheme

- Source: Kaggle `Cornell-University/arxiv` — a metadata snapshot of arxiv.org (NOT
  full papers). Single JSON-lines file `arxiv-metadata-oai-snapshot.json`, ~4 GB,
  ~2.5M records.
- Each record is a JSON object. **Only two fields matter:** `abstract` (input X) and
  `categories` (label y).
- **Load it streaming, line by line.** Never read 4 GB into memory.

### Label scheme (decision made)

- arXiv's taxonomy does NOT map cleanly to "biology, chemistry, physics, CS, social
  sciences." Do not try to force that mapping.
- Use arXiv's **top-level category groups** as labels: `cs`, `math`, `physics`
  (collapse astro-ph, cond-mat, hep-*, quant-ph, etc. into one `physics` bucket),
  `q-bio`, `q-fin`, `stat`, `eess`, `econ`. These are the `Category` StrEnum (§5).
- Papers are **multi-label** (multiple categories; first listed = primary). For v1,
  do **single-label** classification using the top-level group of the *primary*
  category. Note the multi-label nature in the README — do not implement multi-label.
- The distribution is heavily imbalanced. Handle via **capped stratified subsampling**
  (cap per-class counts so majority classes are truncated, minority classes kept whole),
  plus class weights in the loss for residual imbalance (see §7).

---

## 3. Tech stack (locked)

- **API:** Django + Django REST Framework (a **thin adapter** over the ML core, §4).
- **ML core:** standalone, framework-agnostic package `arxiv_ml/` with **no Django/DRF
  imports**, so it can be extracted to its own package/repo and the web layer can be
  swapped (Flask/FastAPI) by replacing only the adapter.
- **Model staging:** TF-IDF + LogisticRegression baseline first (fast, also satisfies
  the task's classical clean/lowercase/stopword/stem/vectorize instruction), then
  `distilbert-base-uncased` fine-tuned for sequence classification — both behind one
  `Predictor` interface (§5). DistilBERT over BERT-base: faster training, smaller
  artifact, lower latency, better prod story.
- **ML libs:** PyTorch + HuggingFace `transformers` (+ scikit-learn for the baseline).
  **Note:** the task brief asks for "TensorFlow or Keras based" models; we deliberately
  deviate to PyTorch (better CPU-serving story, smaller artifact, team familiarity) and
  **must justify this in the README**. HF `transformers` exposes the same interface for
  both backends, so the choice stays defensible.
- **Serving:** gunicorn. Inference device (CPU/GPU) is **switchable via settings/env**
  (`InferenceDevice` StrEnum), **CPU is the default** and the documented production
  default. One CPU-only-wheel image; a GPU image is documented, not built (§13).
- **Container:** Docker (CPU-only torch wheel, built with `uv`), docker-compose.
- **Docs:** `drf-spectacular` for OpenAPI/Swagger.
- **Packaging:** `uv` (`pyproject.toml` + `uv.lock`). **Tests:** pytest (`pytest-django`).
- **Code conventions:** contract DTOs = frozen stdlib dataclasses; interfaces = ABC base
  classes; closed value sets = `enum.StrEnum`. See `AGENTS.md`.

---

## 4. Architecture: package layout & dependency direction (replaces old §4)

```
abstract-classifier/
├── README.md  AGENTS.md  implementation-plan.md
├── Dockerfile  docker-compose.yml  .dockerignore  .gitignore
├── pyproject.toml  uv.lock  Makefile  gunicorn.conf.py  manage.py
├── arxiv_ml/                    # STANDALONE, framework-agnostic ML core (no Django/DRF) — extractable
│   ├── __init__.py
│   ├── enums.py                 # Category, ModelBackend, ModelSourceKind, InferenceDevice (StrEnum)
│   ├── contract.py              # Prediction/CategoryScore frozen dataclasses + Predictor ABC (THE CONTRACT)
│   ├── config.py                # InferenceConfig: plain frozen dataclass (no Django)
│   ├── predictors/
│   │   ├── stub.py              # StubPredictor (fixed/random, real label set)
│   │   ├── tfidf.py             # TfidfPredictor (sklearn artifact)
│   │   └── distilbert.py        # DistilBertPredictor (HF artifact, device-aware)
│   └── loader.py                # ArtifactSource ABC (LocalArtifactSource now) + build_predictor(config) factory
├── config/                      # Django project: settings.py urls.py wsgi.py asgi.py
├── classifier/                  # DRF app — THIN ADAPTER over arxiv_ml, serving only; never imports training/
│   ├── apps.py                  # AppConfig.ready(): build_predictor(...) + warm-up inference
│   ├── registry.py              # process-wide predictor singleton + ready flag
│   ├── serializers.py  views.py  urls.py  exceptions.py
│   └── tests/                   # conftest.py test_validation.py test_predict.py test_health.py
├── training/                    # NOT served; produces artifacts; may import arxiv_ml.enums
│   ├── config.py                # arxiv→group map, caps, seed, paths (imports Category)
│   ├── eda.py                   # stream → class distribution + abstract-length percentiles
│   ├── prepare_data.py          # stream → map→group → capped stratified subsample → split
│   ├── train_baseline.py        # TF-IDF + LogisticRegression → artifact + label map
│   ├── train_distilbert.py      # HF Trainer, 1–2 epochs → artifact + label map
│   └── evaluate.py              # shared metrics: macro P/R/F1, accuracy, confusion matrix
└── artifacts/                   # GITIGNORED — baked into image at build / mounted in dev (MODEL_PATH)
```

**Dependency direction (one-way):** `classifier` and `training` both depend on
`arxiv_ml`; `arxiv_ml` depends on neither. Consequences:
- Switching web frameworks = replace `classifier/` + `config/` only.
- Extracting the ML = move `arxiv_ml/` to its own repo/package.
- Serving never imports `training/`. The trained artifact carries its own `label↔id`
  map; serving reads labels from the artifact and never needs the arxiv→group map
  (that lives only in `training/config.py`).

---

## 5. The model ↔ API contract (Stage 0 — defined before any code)

`arxiv_ml/enums.py` + `arxiv_ml/contract.py` are the linchpin both workstreams build
against. **Write and commit these first.**

```python
# arxiv_ml/enums.py
class Category(StrEnum):
    CS = "cs"; MATH = "math"; PHYSICS = "physics"; Q_BIO = "q-bio"
    Q_FIN = "q-fin"; STAT = "stat"; EESS = "eess"; ECON = "econ"

class ModelBackend(StrEnum):   STUB = "stub"; TFIDF = "tfidf"; DISTILBERT = "distilbert"
class ModelSourceKind(StrEnum): LOCAL = "local"; HF = "hf"; S3 = "s3"
class InferenceDevice(StrEnum): CPU = "cpu"; CUDA = "cuda"; AUTO = "auto"
```

```python
# arxiv_ml/contract.py
@dataclass(frozen=True, slots=True)
class CategoryScore:
    category: Category
    confidence: float

@dataclass(frozen=True, slots=True)
class Prediction:
    predicted_category: Category
    confidence: float
    top_k: list[CategoryScore]
    model_version: str

class Predictor(ABC):
    model_version: str
    labels: list[Category]

    @abstractmethod
    def predict(self, abstract: str, top_k: int = 3) -> Prediction: ...

    # shared concrete helper, e.g. _build_top_k(scores) -> (Prediction fields)
```

`StubPredictor`, `TfidfPredictor`, `DistilBertPredictor` subclass `Predictor`. The DRF
view maps `Prediction` → the §9 JSON response (StrEnum members serialize as their string
value). **Swapping models never touches the view.**

---

## 6. Model-loader seam & config

The model is loaded behind one clean seam — **start simple (baked-in local path), design
for an external store later.** Do not build a plugin framework.

```python
# arxiv_ml/loader.py
class ArtifactSource(ABC):
    @abstractmethod
    def ensure_local(self) -> Path: ...        # returns local artifact dir

class LocalArtifactSource(ArtifactSource): ...  # implemented now — returns MODEL_PATH
# HFHubArtifactSource / S3ArtifactSource — documented subclasses for LATER, NOT built now

def build_predictor(cfg: InferenceConfig) -> Predictor:
    # resolve artifact dir via the configured ArtifactSource, then instantiate the
    # Predictor for cfg.backend. Returns the process singleton (held in classifier/registry.py).
```

`arxiv_ml/config.py` exposes `InferenceConfig` — a plain frozen dataclass the Django
settings layer parses from env and hands in (keeps the core Django-free):

| key | enum/type | default |
|---|---|---|
| `MODEL_BACKEND` | `ModelBackend` | `STUB` |
| `MODEL_SOURCE` | `ModelSourceKind` | `LOCAL` |
| `MODEL_PATH` | path to baked artifacts dir | `artifacts/` |
| `MODEL_VERSION` | str | e.g. `tfidf-arxiv-v1` |
| `INFERENCE_DEVICE` | `InferenceDevice` | `CPU` |
| `MAX_ABSTRACT_CHARS` | int | e.g. `20000` |

`DistilBertPredictor` resolves device: `CUDA` if requested (or `AUTO`) and
`torch.cuda.is_available()`, else `CPU`. Inference under `model.eval()` + `torch.no_grad()`.

---

## 7. The training pipeline (do this minimally — time-box it hard)

`training/` scripts must be **repeatable and parameterized** (shared knobs in
`training/config.py`) so retraining and replacing a poor model is easy. Train on a
**small subset** — large enough for a valid split, small enough to stay fast/cheap.

1. **`eda.py`** — stream the file, compute class distribution and abstract length
   percentiles. Output a couple of plots/tables for the README. That's all.
2. **`prepare_data.py`** — stream, map primary category → top-level `Category`, capped
   stratified subsample (target ~100k–150k rows total is plenty), stratified
   train/val/test split. Persist as parquet/csv.
3. **`train_baseline.py`** — classical preprocessing (clean, lowercase, stopwords,
   stemming/lemmatization) + TF-IDF vectorizer + LogisticRegression (class-weighted).
   Save vectorizer + model + `label↔id` map. **This is the first shippable model.**
4. **`train_distilbert.py`** — tokenize (truncate to 256 tokens),
   `DistilBertForSequenceClassification`, HuggingFace `Trainer`, **1–2 epochs only**,
   no hyperparameter search. Save model, tokenizer, and the `label↔id` map.
5. **`evaluate.py`** — shared metrics on the val/test set: accuracy + **macro**
   precision/recall/F1 + confusion matrix (macro matters because of imbalance; micro
   flatters and misleads). Used by both training scripts.

### Training environment — RunPod (GPU)

- **`train_distilbert.py` runs on a RunPod GPU pod** (the TF-IDF baseline in
  `train_baseline.py` trains fine on CPU/laptop). Document the RunPod workflow in the
  README: spin up a GPU pod, sync `training/` + the prepared dataset, run the script,
  pull the saved artifact back, and tear the pod down. Keep training device-agnostic —
  the script picks `cuda` when available, else `cpu` — so it runs locally too, just
  slower.
- **Serving is decoupled from RunPod.** Artifacts must load and run on **CPU by
  default** (the production default); GPU serving is optional and env-switchable
  (`INFERENCE_DEVICE`), and could itself run on a RunPod GPU pod — but a GPU serving
  image is Phase 4 (§8), not built now.
- **Do not commit artifacts (~250 MB) or the dataset.** Gitignore `artifacts/`. Bake
  the artifact into the Docker image at build time (or mount it). The README notes the
  real-world pattern: pulling a versioned artifact from an external store at startup
  (the loader seam, §6, §13).

---

## 8. The API — build in priority phases

Build Phase 1 fully before touching Phase 2. Mention Phase 3 in the README. Explicitly
name Phase 4 as out of scope in the README.

### Phase 1 — required; this *is* "production-ready"

- **Model loaded once at startup** via `AppConfig.ready()` (or ASGI lifespan), never
  per request. Held as a singleton in `classifier/registry.py`. Inference under
  `model.eval()` + `torch.no_grad()`, on the configured device (CPU default).
- **DRF serializers both directions.** Request: `abstract` required, non-empty,
  length-capped (`MAX_ABSTRACT_CHARS`, reject absurdly long input). Response: structured
  (see §9).
- **Consistent error envelope**, correct HTTP status codes, no stack traces leaked to
  clients. Validation errors → 400 with the standard envelope (`classifier/exceptions.py`).
- **Structured logging from the start.** Configure Django `LOGGING` with a structured
  (JSON) formatter and a request-ID filter; log model load + warm-up, each request with
  its inference latency, and full exceptions **server-side only** (never leaked to the
  client). Proper observability is a Phase 1 foundation, not a later add-on.
- **`/health`** (liveness) and **`/ready`** (returns 200 only once the model is loaded
  and can serve; 503 otherwise) — two distinct endpoints.
- **Warm-up inference at startup** (cheap, kept in Phase 1): run one inference after
  load so `/ready` only passes once the model can actually serve — the first inference is
  always slow.
- **Dockerfile** using `uv` and the **CPU-only torch wheel** (do NOT pull CUDA torch
  into a CPU image — it's gigabytes). Slim base image, locked deps. **docker-compose.yml**.
- **README** with run instructions, example `curl`, and a design-notes section.
- **pytest tests:** validation failure, happy path, error path, health/ready (against
  `StubPredictor`).

### Phase 2 — do if Phase 1 lands early (high signal, cheap)

- OpenAPI/Swagger via `drf-spectacular`.
- API versioning (`/api/v1/`), DRF throttling, max request body size.
- `gunicorn.conf.py` with sensible worker count **and a comment explaining the
  workers × model-memory tradeoff**. (Startup warm-up is already a Phase 1 requirement.)

### Phase 3 — mention in README, do NOT build

Prometheus `/metrics`, token auth, GitHub Actions CI, batch-classify endpoint.

### Phase 4 — explicitly name as out of scope in README

Dedicated inference server (TorchServe/Triton) or a queue (Celery) for real throughput;
model registry + S3 artifact versioning (the loader seam is built to accept it);
blue-green/canary model rollout; a GPU serving image. Include a short "what I'd do beyond
a weekend" subsection listing these — it signals you know exactly where this stops being
a toy. **This subsection is high-leverage; do not skip it.**

---

## 9. API contract (locked)

**`POST /api/v1/classify`**

Request:
```json
{ "abstract": "We present a transformer-based approach to..." }
```

Success `200`:
```json
{
  "predicted_category": "cs",
  "confidence": 0.87,
  "top_k": [
    { "category": "cs",   "confidence": 0.87 },
    { "category": "stat", "confidence": 0.09 },
    { "category": "math", "confidence": 0.03 }
  ],
  "model_version": "distilbert-arxiv-v1"
}
```

Validation error `400`:
```json
{
  "error": {
    "code": "validation_error",
    "message": "The 'abstract' field is required and must be non-empty.",
    "details": {}
  }
}
```

**`GET /health`** → `200` always (process is up).
**`GET /ready`** → `200` if model loaded and warmed up; `503` otherwise.

---

## 10. Build order — MVP-first, each stage independently shippable (replaces old §8)

The API must NOT be blocked on the trained model. Build interface-first:

- **Stage 0 — Contract.** Write `arxiv_ml/enums.py` + `contract.py` + `config.py`;
  confirm §9 API contract. Commit first; both workstreams depend on it.
- **Stage 1 — Working slice (FIRST SHIPPABLE).** Scaffold Django/DRF + `StubPredictor`
  + full Phase 1 API (incl. startup warm-up gating `/ready` and structured logging) +
  pytest against the stub + Dockerfile (`uv`, CPU wheel) + compose.
  **`docker-compose up` returns stub predictions end-to-end.**
- **Stage 2 — TF-IDF baseline (first real model).** `prepare_data.py` (small capped
  subset) + `train_baseline.py` + `TfidfPredictor` + bake artifact; `MODEL_BACKEND=tfidf`.
  `eda.py` + `evaluate.py` produce README tables.
- **Stage 3 — DistilBERT upgrade.** `train_distilbert.py` on Runpod GPU →
  `DistilBertPredictor`, device-switchable; `MODEL_BACKEND=distilbert`; re-evaluate.
  Drop-in via the contract.
- **Stage 4 — Phase 2 polish.** `drf-spectacular`; versioning/throttling/body-size;
  `gunicorn.conf.py` worker tuning. (Structured logging is already in place from Stage 1.)
- **Stage 5 — README + Phase 3/4 sections** (incl. PyTorch-vs-TF justification; GHCR
  image distribution; S3/HF-Hub + GPU image as "beyond a weekend").

Each stage leaves the system runnable. Swapping stub → baseline → DistilBERT is a config
change (`MODEL_BACKEND`), not a rewrite.

---

## 11. Parallel workstreams (for independent agents)

- **WS-A — API adapter** (Stages 0,1,4): `config/` Django project, `classifier/` app
  (serializers, views, error envelope, health/ready, registry, structured logging),
  settings/env → `InferenceConfig` wiring, tests vs `StubPredictor`.
- **WS-B — ML core + training** (Stages 0,2,3): `arxiv_ml/*` (enums, contract,
  predictors, loader, config) + `training/*`; produces artifacts and the predictor
  implementations that satisfy the contract.
- **WS-C — Infra/serving** (Stages 1,4,5): Dockerfile (`uv` + CPU wheel), compose,
  `.dockerignore`, `gunicorn.conf.py`, `Makefile`, artifact baking, startup warm-up.

**Dependency edges:** WS-A & WS-B both depend only on `arxiv_ml/contract.py` + `enums.py`
(Stage 0). WS-C bakes WS-B's artifacts and runs WS-A. No other cross-coupling — the three
can proceed in parallel after Stage 0 lands.

---

## 12. Inference concurrency note (get this right)

PyTorch inference partially holds the GIL (it releases during tensor ops but holds it
for Python overhead), so a single sync worker serializes requests. Correct weekend-scale
answer: **gunicorn with a few sync workers**, each loading its own model copy
(DistilBERT ~250 MB → 2–4 workers is fine), OR async Django running inference in a
threadpool executor. Inference is CPU-bound, so **sync workers are the default** — async
only helps I/O-bound concurrency. Do **not** build a queue/dedicated inference server —
that belongs in Phase 4 of the README, not in the code.

---

## 13. Artifact & image distribution

Bake the artifact into the image now (`LocalArtifactSource` reads `MODEL_PATH`). The README
documents the evolution the loader seam is built for:
- **Image distribution:** GitHub Container Registry (GHCR, `ghcr.io`) — free for public
  images.
- **Versioned artifact store (later):** HuggingFace Hub (free public repos) or a GitHub
  Release (≤2 GB/file) for the weekend; S3 / an MLflow registry as the production
  end-state — a drop-in `ArtifactSource` subclass. **Avoid Git LFS** (1 GB free tier is too
  small). The dataset stays on Kaggle, never in the repo.

---

## 14. Definition of done — acceptance criteria per phase

A phase is "done" only when its box list passes. Phase 1 is the graded minimum; Phases
2–4 are scoped by §8 (2 = build if time, 3 = mention only, 4 = name as out of scope).

### Phase 1 — required (the graded minimum)

- [ ] `docker-compose up` brings the service up cleanly.
- [ ] Model loads exactly once at startup; `/ready` flips to 200 only after load + warm-up.
- [ ] `POST /api/v1/classify` returns the §9 schema for valid input.
- [ ] Empty / missing / oversized `abstract` → 400 with the standard error envelope.
- [ ] No stack traces leak to the client on any error path.
- [ ] Structured (JSON) logging is configured: each request logs a request ID + inference
      latency, model load/warm-up is logged, and exceptions are logged server-side only.
- [ ] Docker image uses the CPU-only torch wheel; image is not bloated with CUDA.
- [ ] `pytest` passes (validation, happy path, error path, health/ready).
- [ ] `arxiv_ml/` imports no Django/DRF; `classifier/` imports nothing from `training/`.
- [ ] README (minimal): setup/run instructions, an example `curl`, and a short
      design-notes section. (The fuller model/metrics documentation is a Phase 2
      criterion below — it must not gate the MVP.)
- [ ] The 4 GB dataset and the model artifacts are gitignored.

### Phase 2 — done if built (high signal, cheap)

- [ ] OpenAPI schema is served and the Swagger UI is reachable; `/api/v1/classify`,
      `/health`, `/ready` appear with request/response schemas (`drf-spectacular`).
- [ ] Endpoints are under `/api/v1/`; throttling returns `429` past the limit; an
      oversized request body is rejected before reaching the model.
- [ ] `gunicorn.conf.py` exists with an explicit worker count and a comment explaining
      the workers × model-memory tradeoff.
- [ ] README (full) adds, now that a real model exists: the metrics table (macro
      P/R/F1 + confusion matrix), the label-scheme rationale (incl. why chemistry /
      social-sciences are not arXiv top-level outputs), the multi-label caveat, the
      preprocessing-vs-BERT note, the RunPod training workflow (once DistilBERT lands),
      and the justification for PyTorch over the task's "TensorFlow/Keras" ask.

### Phase 3 — done = documented, not built

- [ ] README names each deferred item (Prometheus `/metrics`, token auth, GitHub
      Actions CI, batch-classify endpoint) with one line on what it would add and why
      it's out of weekend scope.
- [ ] No Phase 3 code exists in the repo.

### Phase 4 — done = explicitly scoped out

- [ ] README "beyond a weekend" subsection lists, with a sentence each: a dedicated
      inference server (TorchServe/Triton) or a Celery queue for throughput; a model
      registry + S3/MLflow artifact versioning (the loader seam is built to accept it);
      blue-green/canary model rollout; and a GPU serving image (optionally on RunPod).
- [ ] No Phase 4 code exists in the repo.

---

## 15. Things NOT to do

- Don't fine-tune for hours or do hyperparameter search — the model is a checkbox.
- Don't do heavy classical preprocessing (stopword removal, stemming) on text fed to
  DistilBERT — it has its own subword tokenizer and wants natural text. (Do that
  preprocessing only for the TF-IDF baseline.)
- Don't load the model per request.
- Don't commit the dataset or the artifacts.
- Don't put CUDA torch in the serving image.
- Don't import Django/DRF anywhere in `arxiv_ml/`, and don't import `training/` from
  `classifier/` — these break the extractability/framework-swap guarantees.
- Don't build Phase 3/Phase 4 features — reference them in the README instead.
