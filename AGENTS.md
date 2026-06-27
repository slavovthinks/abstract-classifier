# AGENTS.md

Working agreement for AI agents (Codex, Claude Code) on **abstract-classifier**.
This file is the source of truth for *how* to work in this repo. Read it fully
before acting.

## Project in one line

A Django REST Framework API that classifies research-paper abstracts (arXiv) into
top-level category groups, backed by a fine-tuned transformer model. **The API is
the graded deliverable; the model is a means to an end, not the focus.**

See `implementation-plan.md` for the current plan, build order, and decisions.

## How to work

- **Deliver a working vertical slice first, then improve.** Never let polish block a
  runnable baseline. (Build order detailed in `implementation-plan.md` §10.)
- **Set up proper logging early.** Configure structured logging as part of the first
  working slice, not as a later polish step.
- **Don't over-engineer.** Add an abstraction or design pattern only when it earns
  its place. When in doubt about whether something is worth building, ask rather
  than gold-plate.
- **Docs-first, not assumptions.** Look up official/current documentation before
  committing to an approach (use context7 or web search for library docs).
- **Best-practice guardrail.** If a request conflicts with Django / DRF best
  practice, stop and ask for explicit confirmation before proceeding — call out the
  conflict and the recommended alternative.
- **Automate repeatable steps.** Where a task is repeatable, write a shell/Python
  script that can later be lifted into CI/CD, rather than documenting manual steps.

## Conventions

- **Comments:** only where they add value the code itself cannot convey. No
  narrating-the-obvious comments.
- **Testing:** pytest (with `pytest-django`), unless there is a clearly better
  idiomatic Django/DRF way. Prioritize Django and DRF conventions throughout.
- **Task automation:** a `Makefile` is fine if it fits Django workflows; otherwise
  use management commands / the Django-native approach.
- **Closed value sets:** use `enum.StrEnum` (e.g. categories, model backends, device
  selection) — never bare string literals. Members are real strings, so they stay
  JSON-serializable and parse cleanly from env/settings.
- **Boundary types & interfaces:** represent contract/DTO types as frozen stdlib
  `dataclass`es (keep them dependency-light and framework-agnostic); define extension
  points as `abc.ABC` base classes when you own the implementations and want runtime
  enforcement + shared helpers.

## Tooling & environment

- **Package management:** `uv`.
- **ML framework:** PyTorch. Keep the framework/model choice swappable via
  settings/env vars — do not hard-code it at call sites.
- **Training:** runs on Runpod (GPU). Serving runs on CPU by default but the device
  must be switchable (see `implementation-plan.md` §6).
- **Model artifacts:** loaded behind a clean loader/source seam — simple source now,
  designed to add an external registry/store (MLflow or S3) later (see
  `implementation-plan.md` §6, §13). Artifacts and the dataset are never committed.
