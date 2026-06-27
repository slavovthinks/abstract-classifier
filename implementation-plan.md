# Handoff: arXiv Abstract Classifier — Production-Oriented API

## How to read this document

This is a decision-complete spec. The technical choices below are **already made** —
do not re-litigate them, just execute. Where rationale is given, it's so you make
consistent downstream choices, not an invitation to reconsider. If you hit a genuine
fork not covered here, prefer the option that makes the **API more production-ready**,
since that is the graded deliverable.

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
toy patterns; avoid over-building past what a weekend allows (see tiers in §6).

---

## 2. The dataset

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
  `q-bio`, `q-fin`, `stat`, `eess`, `econ`.
- Papers are **multi-label** (multiple categories; first listed = primary). For v1,
  do **single-label** classification using the top-level group of the *primary*
  category. Note the multi-label nature in the README — do not implement multi-label.
- The distribution is heavily imbalanced. Handle via stratified subsampling with
  capped per-class counts (see §5).

---

## 3. Tech stack (locked)

- **API:** Django + Django REST Framework
- **Model:** `distilbert-base-uncased` fine-tuned for sequence classification
  (DistilBERT, not BERT-base — faster training, smaller artifact, lower inference
  latency, better prod story)
- **ML libs:** PyTorch + HuggingFace `transformers`
- **Serving:** CPU inference, gunicorn
- **Container:** Docker (CPU-only torch wheel), docker-compose
- **Docs:** `drf-spectacular` for OpenAPI/Swagger
- **Tests:** pytest

---

## 4. Project structure

```
arxiv-classifier/
├── README.md
├── handoff.md                 # this file
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── gunicorn.conf.py
├── manage.py
├── config/                    # Django project
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── classifier/                # DRF app (serving only)
│   ├── apps.py                # model load on AppConfig.ready()
│   ├── ml/
│   │   ├── predictor.py       # singleton: model + tokenizer + label map
│   │   └── artifacts/         # saved model — GITIGNORED, mounted/baked at build
│   ├── serializers.py
│   ├── views.py
│   ├── urls.py
│   └── tests/
│       ├── test_validation.py
│       ├── test_predict.py
│       └── test_health.py
└── training/                  # NOT part of the served app
    ├── prepare_data.py        # stream + subsample + split
    ├── train.py               # fine-tune + save artifact
    └── eda.py                 # quick EDA, distribution + length stats
```

**Critical separation:** `training/` produces the artifact; `classifier/` only loads
it. The served app must never import training code or pull the 4 GB dataset.

---

## 5. The model (do this minimally — ~3 hrs of effort, time-box it hard)

`training/` scripts:

1. **`eda.py`** — stream the file, compute class distribution and abstract length
   percentiles. Output a couple of plots/tables for the README. That's all.
2. **`prepare_data.py`** — stream, map primary category → top-level group, stratified
   subsample with capped per-class counts (target ~100k–150k rows total is plenty),
   train/val/test split (stratified). Persist as parquet/csv.
3. **`train.py`** — tokenize (truncate to 256 tokens), `DistilBertForSequenceClassification`,
   HuggingFace `Trainer`, **1–2 epochs only**, no hyperparameter search. Evaluate once
   on the val set: accuracy + **macro** precision/recall/F1 + confusion matrix (macro
   matters because of imbalance; micro will flatter and mislead). Save model, tokenizer,
   and the label↔id map into `classifier/ml/artifacts/`.

Gotchas:
- **Train on GPU, serve on CPU.** Training can run on a Runpod GPU; the predictor must
  detect device and default to CPU. The artifact must load and run on CPU.
- **Do not commit the artifact (~250 MB) to git.** Gitignore it. Bake it into the
  Docker image at build time, or mount it. In the README, note that the real-world
  pattern is pulling a versioned artifact from object storage (S3) at startup.

---

## 6. The API — build in priority tiers

Build Tier 1 fully before touching Tier 2. Mention Tier 3 in the README. Explicitly
name Tier 4 as out of scope in the README.

### Tier 1 — required; this *is* "production-ready"

- **Model loaded once at startup** via `AppConfig.ready()` (or ASGI lifespan), never
  per request. Inference under `model.eval()` + `torch.no_grad()`, pinned to CPU.
- **DRF serializers both directions.** Request: `abstract` required, non-empty,
  length-capped (reject absurdly long input). Response: structured (see §7).
- **Consistent error envelope**, correct HTTP status codes, no stack traces leaked to
  clients. Validation errors → 400 with the standard envelope.
- **`/health`** (liveness) and **`/ready`** (returns 200 only once the model is
  loaded and can serve; 503 otherwise) — two distinct endpoints.
- **Dockerfile** using the **CPU-only torch wheel** (do NOT pull CUDA torch into a CPU
  image — it's gigabytes). Slim base image, pinned deps. **docker-compose.yml**.
- **README** with run instructions, example `curl`, and a design-notes section.
- **pytest tests:** validation failure, happy path, error path, health/ready.

### Tier 2 — do if Tier 1 lands early (high signal, cheap)

- OpenAPI/Swagger via `drf-spectacular`.
- Structured logging with request IDs; **log inference latency** per request.
- API versioning (`/api/v1/`), DRF throttling, max request body size.
- `gunicorn.conf.py` with sensible worker count **and a comment explaining the
  workers × model-memory tradeoff**. Add a **warm-up inference at startup** so `/ready`
  only passes once the model can actually serve (first inference is always slow).

### Tier 3 — mention in README, do NOT build

Prometheus `/metrics`, token auth, GitHub Actions CI, batch-classify endpoint.

### Tier 4 — explicitly name as out of scope in README

Dedicated inference server (TorchServe/Triton) or a queue (Celery) for real
throughput; model registry + S3 artifact versioning; blue-green/canary model rollout.
Include a short "what I'd do beyond a weekend" subsection listing these — it signals
you know exactly where this stops being a toy. **This subsection is high-leverage; do
not skip it.**

---

## 7. API contract (locked)

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
**`GET /ready`** → `200` if model loaded; `503` otherwise.

---

## 8. Build order (important: decouple the two workstreams)

The API must NOT be blocked on the trained model. Build interface-first:

1. **Scaffold** the project structure (§4) and Django/DRF skeleton.
2. **Stub predictor first.** Implement `predictor.py` behind a clean interface that, in
   stub mode, returns a fixed/random label with plausible confidences and the real
   label set. Build and fully test the entire API (Tier 1) against this stub.
3. **Write the `training/` scripts** in parallel (§5).
4. **Swap the stub for the real artifact** once training has produced it — this should
   be a one-line change because the predictor interface stays identical.

This way the API is complete and tested before the model artifact even exists.

---

## 9. Inference concurrency note (get this right)

PyTorch inference partially holds the GIL (it releases during tensor ops but holds it
for Python overhead), so a single sync worker serializes requests. Correct weekend-scale
answer: **gunicorn with a few sync workers**, each loading its own model copy
(DistilBERT ~250 MB → 2–4 workers is fine), OR async Django running inference in a
threadpool executor. Do **not** build a queue/dedicated inference server — that belongs
in Tier 4 of the README, not in the code.

---

## 10. Definition of done (Tier 1 acceptance checklist)

- [ ] `docker-compose up` brings the service up cleanly.
- [ ] Model loads exactly once at startup; `/ready` flips to 200 only after load + warm-up.
- [ ] `POST /api/v1/classify` returns the §7 schema for valid input.
- [ ] Empty / missing / oversized `abstract` → 400 with the standard error envelope.
- [ ] No stack traces leak to the client on any error path.
- [ ] Docker image uses the CPU-only torch wheel; image is not bloated with CUDA.
- [ ] `pytest` passes (validation, happy path, error path, health/ready).
- [ ] README covers: setup/run, example curl, label-scheme choice, multi-label caveat,
      preprocessing-vs-BERT note, metrics table, and the Tier 4 "beyond a weekend" section.
- [ ] The 4 GB dataset and the model artifact are gitignored.

---

## 11. Things NOT to do

- Don't fine-tune for hours or do hyperparameter search — the model is a checkbox.
- Don't do heavy classical preprocessing (stopword removal, stemming) on text fed to
  DistilBERT — it has its own subword tokenizer and wants natural text. (Do that
  preprocessing only if you build a TF-IDF baseline, which is optional here.)
- Don't load the model per request.
- Don't commit the dataset or the artifact.
- Don't put CUDA torch in the serving image.
- Don't build Tier 3/Tier 4 features — reference them in the README instead.