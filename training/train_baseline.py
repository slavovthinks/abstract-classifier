"""Train the TF-IDF + LogisticRegression baseline.

Reads the prepared splits, fits an sklearn Pipeline whose vectorizer uses the
shared :func:`arxiv_ml.predictors.text.clean` preprocessor (so serving applies
identical preprocessing), evaluates on val + test, and writes the artifact the
``TfidfPredictor`` loads: ``pipeline.joblib`` + ``labels.json`` + ``meta.json``.

Run with: ``make train-baseline`` or ``uv run python -m training.train_baseline``.
"""

import datetime as dt
import json
import logging

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from arxiv_ml.predictors.text import clean
from training import config, evaluate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("training.train_baseline")

MODEL_VERSION = "tfidf-arxiv-v1"


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean,
                    ngram_range=(1, 2),
                    min_df=5,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                ),
            ),
        ]
    )


def _load_split(name: str) -> pd.DataFrame:
    path = config.PREPARED_DIR / f"{name}.parquet"
    if not path.exists():
        raise SystemExit(
            f"Missing split {path}. Run `make prepare-data` first."
        )
    return pd.read_parquet(path)


def run() -> None:
    train = _load_split("train")
    val = _load_split("val")
    test = _load_split("test")
    logger.info("loaded splits: train=%d val=%d test=%d", len(train), len(val), len(test))

    pipeline = _build_pipeline()
    logger.info("fitting TF-IDF + LogisticRegression on %d rows...", len(train))
    pipeline.fit(train["abstract"], train["label"])

    labels = list(pipeline.named_steps["clf"].classes_)
    evaluate.evaluate(val["label"], pipeline.predict(val["abstract"]), labels, config.METRICS_DIR, "val")
    evaluate.evaluate(test["label"], pipeline.predict(test["abstract"]), labels, config.METRICS_DIR, "test")

    _save_artifact(pipeline, labels, len(train))


def _save_artifact(pipeline: Pipeline, labels: list[str], train_rows: int) -> None:
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.ARTIFACT_DIR / "pipeline.joblib")
    (config.ARTIFACT_DIR / "labels.json").write_text(
        json.dumps({"labels": labels}, indent=2) + "\n"
    )
    meta = {
        "model_version": MODEL_VERSION,
        "backend": "tfidf",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "train_rows": train_rows,
        "per_class_cap": config.PER_CLASS_CAP,
        "seed": config.SEED,
    }
    (config.ARTIFACT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("artifact written to %s (model_version=%s)", config.ARTIFACT_DIR, MODEL_VERSION)


if __name__ == "__main__":
    run()
