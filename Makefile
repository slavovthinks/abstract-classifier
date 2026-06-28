.DEFAULT_GOAL := help
.PHONY: help install test run serve docker-build docker-up docker-down lint dataset eda prepare-data train-baseline promote-model refresh-doc-assets

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Sync the virtualenv from the lockfile (incl. dev deps)
	uv sync

dataset: ## Download the arXiv metadata snapshot from Kaggle into data/ (needs ~/.kaggle/kaggle.json)
	mkdir -p data
	uv run kaggle datasets download -d Cornell-University/arxiv -p data/ --unzip

eda: ## Stream the dataset and report class distribution + abstract-length stats
	uv run python -m training.eda

prepare-data: ## Subsample + split the dataset into train/val/test parquet files
	uv run python -m training.prepare_data

train-baseline: ## Train the TF-IDF baseline and write artifacts/tfidf/
	uv run python -m training.train_baseline

promote-model: ## Copy the freshly trained artifacts/tfidf/ over the committed pretrained/tfidf/ (the shipped/baked model)
	rm -rf pretrained/tfidf
	mkdir -p pretrained/tfidf
	cp artifacts/tfidf/pipeline.joblib artifacts/tfidf/labels.json artifacts/tfidf/meta.json pretrained/tfidf/

refresh-doc-assets: ## Copy the latest EDA + evaluation plots into the committed docs/assets/ used by the README
	mkdir -p docs/assets
	cp artifacts/eda/class_distribution.png artifacts/eda/abstract_length_hist.png docs/assets/
	cp artifacts/metrics/confusion_test.png docs/assets/

test: ## Run the test suite
	uv run pytest

MODEL_BACKEND ?= stub

run: ## Run the dev server (Django autoreload; override MODEL_BACKEND, e.g. MODEL_BACKEND=tfidf)
	MODEL_BACKEND=$(MODEL_BACKEND) uv run python manage.py runserver 0.0.0.0:8000

serve: ## Run the production server (gunicorn)
	uv run gunicorn --config gunicorn.conf.py config.wsgi:application

docker-build: ## Build the Docker image
	docker compose build

docker-up: ## Start the service via docker-compose
	docker compose up

docker-down: ## Stop and remove the service
	docker compose down
