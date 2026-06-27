from abc import ABC, abstractmethod
from dataclasses import dataclass

from arxiv_ml.enums import Category


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

    def _build_prediction(
        self,
        scores: list[tuple[Category, float]],
        top_k: int,
    ) -> Prediction:
        """Build a Prediction from a full list of (category, score) pairs."""
        if not scores:
            raise ValueError("scores must contain at least one category score")

        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
        best_cat, best_conf = scores_sorted[0]
        top_k_scores = [
            CategoryScore(category=cat, confidence=conf)
            for cat, conf in scores_sorted[:top_k]
        ]
        return Prediction(
            predicted_category=best_cat,
            confidence=best_conf,
            top_k=top_k_scores,
            model_version=self.model_version,
        )
