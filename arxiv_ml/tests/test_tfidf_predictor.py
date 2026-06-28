"""TfidfPredictor unit tests against a tiny in-test artifact.

Trains a 3-class micro-pipeline in a temp dir so CI never needs the 4 GB
dataset or a baked artifact; it exercises the real load + predict path and the
shared ``clean`` preprocessor.
"""

import json

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from arxiv_ml.contract import Prediction
from arxiv_ml.enums import Category
from arxiv_ml.predictors.text import clean
from arxiv_ml.predictors.tfidf import TfidfPredictor

_SAMPLES = [
    ("neural network deep learning model trained on data", Category.CS),
    ("algorithm complexity software computation programming", Category.CS),
    ("compiler optimization runtime memory allocation", Category.CS),
    ("galaxy cosmology quantum particle field theory", Category.PHYSICS),
    ("telescope astronomy star formation universe expansion", Category.PHYSICS),
    ("relativity spacetime gravitation black hole", Category.PHYSICS),
    ("theorem proof algebra topology manifold group", Category.MATH),
    ("integral derivative function continuous limit", Category.MATH),
    ("prime number ring field polynomial equation", Category.MATH),
]


@pytest.fixture
def tfidf_artifact(tmp_path):
    texts = [text for text, _ in _SAMPLES]
    labels = [category.value for _, category in _SAMPLES]
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(preprocessor=clean, min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, tmp_path / "pipeline.joblib")
    (tmp_path / "labels.json").write_text(
        json.dumps({"labels": list(pipeline.named_steps["clf"].classes_)})
    )
    return tmp_path


def test_predict_returns_valid_prediction(tfidf_artifact):
    predictor = TfidfPredictor(tfidf_artifact, model_version="tfidf-test-v1")
    prediction = predictor.predict("deep neural network architecture for vision", top_k=3)

    assert isinstance(prediction, Prediction)
    assert prediction.model_version == "tfidf-test-v1"
    assert prediction.predicted_category in set(Category)
    assert 0.0 <= prediction.confidence <= 1.0

    assert len(prediction.top_k) == 3
    confidences = [score.confidence for score in prediction.top_k]
    assert confidences == sorted(confidences, reverse=True)
    assert prediction.top_k[0].category == prediction.predicted_category
    assert prediction.top_k[0].confidence == pytest.approx(prediction.confidence)


def test_labels_are_categories(tfidf_artifact):
    predictor = TfidfPredictor(tfidf_artifact, model_version="tfidf-test-v1")
    assert predictor.labels
    assert all(isinstance(label, Category) for label in predictor.labels)


def test_topk_respects_k(tfidf_artifact):
    predictor = TfidfPredictor(tfidf_artifact, model_version="tfidf-test-v1")
    prediction = predictor.predict("polynomial ring algebra", top_k=2)
    assert len(prediction.top_k) == 2