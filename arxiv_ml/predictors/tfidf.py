"""TF-IDF + LogisticRegression predictor.

Loads the artifact produced by ``training/train_baseline.py``: a pickled sklearn
Pipeline plus a ``labels.json`` recording the class order. The pipeline's
vectorizer already applies the shared :func:`arxiv_ml.predictors.text.clean`
preprocessor, so inference needs no separate preprocessing step.
"""

import json
from pathlib import Path

import joblib

from arxiv_ml.contract import Prediction, Predictor
from arxiv_ml.enums import Category


class TfidfPredictor(Predictor):
    def __init__(self, artifact_dir: Path, model_version: str) -> None:
        artifact_dir = Path(artifact_dir)
        self.model_version = model_version
        self._pipeline = joblib.load(artifact_dir / "pipeline.joblib")
        labels = json.loads((artifact_dir / "labels.json").read_text())["labels"]
        self.labels: list[Category] = [Category(label) for label in labels]

    def predict(self, abstract: str, top_k: int = 3) -> Prediction:
        probabilities = self._pipeline.predict_proba([abstract])[0]
        scores = [
            (label, float(probability))
            for label, probability in zip(self.labels, probabilities, strict=True)
        ]
        return self._build_prediction(scores, top_k)