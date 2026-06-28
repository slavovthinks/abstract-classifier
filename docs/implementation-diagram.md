# Implementation Diagram

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    abstract-classifier/                          │
│                                                                 │
│  ┌─────────────────┐   ┌──────────────────┐  ┌──────────────┐  │
│  │   arxiv_ml/     │   │   classifier/    │  │  training/   │  │
│  │  (ML CORE)      │   │  (DRF ADAPTER)   │  │  (scripts)   │  │
│  │  NO Django      │◄──│  THIN LAYER      │  │  NO serving  │  │
│  │  NO DRF         │   │                  │  │              │  │
│  └─────────────────┘   └──────────────────┘  └──────┬───────┘  │
│         ▲                                           │          │
│         └───────────────────────────────────────────┘          │
│                    Both depend on arxiv_ml                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Build Stages (MVP-first)

```
STAGE 0 — Contract (both workstreams depend on this)
├── arxiv_ml/enums.py     → Category, ModelBackend, InferenceDevice StrEnums
├── arxiv_ml/contract.py  → CategoryScore, Prediction dataclasses + Predictor ABC
└── arxiv_ml/config.py    → InferenceConfig frozen dataclass
         │
         ▼
STAGE 1 — Working Slice (FIRST SHIPPABLE)
├── WS-A: Django/DRF scaffold + StubPredictor end-to-end
│   ├── classifier/apps.py        → AppConfig.ready() loads predictor
│   ├── classifier/registry.py    → process-wide singleton
│   ├── classifier/views.py       → POST /api/v1/classify
│   ├── classifier/serializers.py → request validation + response
│   ├── classifier/exceptions.py  → error envelope
│   ├── GET /health  (liveness)
│   ├── GET /ready   (503 until model warmed up, then 200)
│   └── structured (JSON) logging + request-ID filter
├── WS-B: arxiv_ml/predictors/stub.py → StubPredictor
├── WS-C: Dockerfile (uv + CPU-only torch wheel) + docker-compose
└── pytest: validation, happy path, error path, health/ready
         │
         ▼ ✓ docker-compose up → stub predictions end-to-end
         │
STAGE 2 — TF-IDF Baseline (first real model)
├── training/prepare_data.py   → stream JSON, map→Category, capped subsample, split
├── training/eda.py            → class distribution + length percentiles
├── training/train_baseline.py → TF-IDF + LogisticRegression → artifact
├── training/evaluate.py       → macro P/R/F1 + confusion matrix
├── arxiv_ml/predictors/tfidf.py → TfidfPredictor
└── bake artifact; switch MODEL_BACKEND=tfidf
         │
         ▼ ✓ docker-compose up → real TF-IDF predictions
         │
STAGE 3 — DistilBERT (RunPod GPU)
├── training/train_distilbert.py    → tokenize(256) + HF Trainer 1-2 epochs
├── arxiv_ml/predictors/distilbert.py → DistilBertPredictor (device-aware)
└── re-evaluate → update README metrics
         │
         ▼ ✓ MODEL_BACKEND=distilbert, drop-in via contract
         │
STAGE 4 — Phase 2 Polish
├── drf-spectacular  → OpenAPI + Swagger UI at /api/schema/
├── API versioning /api/v1/, throttling (429), body-size limit
└── gunicorn.conf.py → worker count + memory tradeoff comment
         │
STAGE 5 — README
└── PyTorch vs TF justification, metrics, RunPod workflow,
    Phase 3 deferrals, Phase 4 "beyond a weekend" section
```

---

## Dependency Flow

```
                ┌─────────────┐
                │  enums.py   │
                │ contract.py │  ← Stage 0: commit FIRST
                │  config.py  │
                └──────┬──────┘
           ┌───────────┼───────────┐
           ▼           ▼           ▼
     ┌──────────┐ ┌──────────┐ ┌──────────┐
     │  WS-A    │ │  WS-B    │ │  WS-C    │
     │  DRF API │ │ ML core+ │ │  Infra   │
     │          │ │ training │ │  Docker  │
     └──────────┘ └──────────┘ └──────────┘
           │           │           │
           └───────────▼───────────┘
              docker-compose up ✓
```

---

## Model Swap (config only, no code change)

```
StubPredictor → TfidfPredictor → DistilBertPredictor
     ↑                ↑                  ↑
MODEL_BACKEND=stub  =tfidf         =distilbert
     │                │                  │
     └────────────────┴──────────────────┘
              all satisfy Predictor ABC
              view never changes
```

---

## Phases (what gets graded)

```
Phase 1 — REQUIRED (graded minimum)
  ✓ docker-compose up works
  ✓ /health always 200; /ready gates on model warm-up
  ✓ POST /api/v1/classify with full error envelope
  ✓ Structured JSON logging from day 1
  ✓ CPU-only Docker image (no CUDA bloat)
  ✓ pytest suite passes

Phase 2 — Build if time (high signal)
  ✓ Swagger UI via drf-spectacular
  ✓ Throttling (429), body-size limit
  ✓ gunicorn.conf.py with workers note

Phase 3 — Mention in README only (no code)
  → Prometheus /metrics, token auth, CI, batch endpoint

Phase 4 — Explicitly out of scope (mention in README)
  → TorchServe/Triton, Celery queue, S3 artifact registry,
     GPU serving image, blue-green rollout
```

---

## File Layout

```
abstract-classifier/
├── README.md  AGENTS.md  CLAUDE.md
├── docs/                           # implementation-plan.md, implementation-diagram.md, issues.md, Task 1 AI Engineer.md
├── Dockerfile  docker-compose.yml  .dockerignore  .gitignore
├── pyproject.toml  uv.lock  Makefile  gunicorn.conf.py  manage.py
├── arxiv_ml/                      # standalone ML core — no Django/DRF
│   ├── enums.py                   # Category, ModelBackend, ModelSourceKind, InferenceDevice
│   ├── contract.py                # Prediction, CategoryScore dataclasses + Predictor ABC
│   ├── config.py                  # InferenceConfig frozen dataclass
│   ├── loader.py                  # ArtifactSource ABC + LocalArtifactSource + build_predictor()
│   └── predictors/
│       ├── stub.py                # StubPredictor
│       ├── tfidf.py               # TfidfPredictor
│       └── distilbert.py          # DistilBertPredictor
├── config/                        # Django project settings, urls, wsgi, asgi
├── classifier/                    # DRF app — thin adapter; never imports training/
│   ├── apps.py                    # AppConfig.ready() → build_predictor() + warm-up
│   ├── registry.py                # process-wide predictor singleton + ready flag
│   ├── serializers.py
│   ├── views.py
│   ├── urls.py
│   ├── exceptions.py              # consistent error envelope
│   └── tests/
│       ├── conftest.py
│       ├── test_validation.py
│       ├── test_predict.py
│       └── test_health.py
├── training/                      # not served; produces artifacts
│   ├── config.py                  # arxiv→group map, caps, seed, paths
│   ├── eda.py
│   ├── prepare_data.py
│   ├── train_baseline.py
│   ├── train_distilbert.py
│   └── evaluate.py
└── artifacts/                     # GITIGNORED — baked into image / mounted in dev
```
