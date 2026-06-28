import hashlib
import math

from arxiv_ml.contract import Prediction, Predictor
from arxiv_ml.enums import Category


class StubPredictor(Predictor):
    """A dependency-free predictor that satisfies the contract without a model.

    It produces a deterministic, defensible distribution over the real label set
    derived from a hash of the abstract, so the API can be built and tested
    end-to-end before any trained artifact exists. Swapping in a real predictor
    is a config change (``MODEL_BACKEND``), not a code change.
    """

    def __init__(self, model_version: str = "stub-v1") -> None:
        self.model_version = model_version
        self.labels: list[Category] = list(Category)

    def predict(self, abstract: str, top_k: int = 3) -> Prediction:
        scores = self._score(abstract)
        return self._build_prediction(scores, top_k)

    def _score(self, abstract: str) -> list[tuple[Category, float]]:
        """Deterministic pseudo-logits per label, softmax-normalised."""
        logits: list[float] = []
        for label in self.labels:
            digest = hashlib.sha256(f"{label.value}:{abstract}".encode()).digest()
            # Map first two bytes to a stable logit in a small range.
            logits.append((digest[0] + digest[1] / 256.0) / 64.0)

        max_logit = max(logits)
        exps = [math.exp(logit - max_logit) for logit in logits]
        total = sum(exps)
        return [
            (label, exp / total) for label, exp in zip(self.labels, exps, strict=True)
        ]
