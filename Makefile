.DEFAULT_GOAL := help
.PHONY: help install test run serve docker-build docker-up docker-down lint

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Sync the virtualenv from the lockfile (incl. dev deps)
	uv sync

test: ## Run the test suite
	uv run pytest

run: ## Run the dev server (Django autoreload, stub backend)
	MODEL_BACKEND=stub uv run python manage.py runserver 0.0.0.0:8000

serve: ## Run the production server (gunicorn)
	uv run gunicorn --config gunicorn.conf.py config.wsgi:application

docker-build: ## Build the Docker image
	docker compose build

docker-up: ## Start the service via docker-compose
	docker compose up

docker-down: ## Stop and remove the service
	docker compose down
