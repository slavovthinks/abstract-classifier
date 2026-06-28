from abc import ABC, abstractmethod
from pathlib import Path

from arxiv_ml.config import InferenceConfig
from arxiv_ml.contract import Predictor
from arxiv_ml.enums import ModelBackend, ModelSourceKind


class ArtifactSource(ABC):
    """Seam between the predictor and wherever its artifact physically lives.

    Implemented now: :class:`LocalArtifactSource`. Designed so an external store
    (HuggingFace Hub, S3, an MLflow registry) drops in later as a sibling subclass
    without touching predictors or the web layer (see docs/implementation-plan.md §6, §13).
    """

    @abstractmethod
    def ensure_local(self) -> Path:
        """Resolve the artifact to a local directory and return its path."""


class LocalArtifactSource(ArtifactSource):
    """Reads an artifact already on disk (baked into the image or mounted)."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def ensure_local(self) -> Path:
        return self._path


def _build_source(cfg: InferenceConfig) -> ArtifactSource:
    if cfg.model_source is ModelSourceKind.LOCAL:
        return LocalArtifactSource(cfg.model_path)
    raise NotImplementedError(
        f"Artifact source '{cfg.model_source}' is not implemented yet; "
        "only LOCAL is available in this stage."
    )


def build_predictor(cfg: InferenceConfig) -> Predictor:
    """Factory: resolve the artifact and instantiate the configured predictor.

    Predictors are imported lazily so selecting the stub backend never pulls in
    heavyweight ML dependencies (sklearn / torch).
    """
    if cfg.model_backend is ModelBackend.STUB:
        from arxiv_ml.predictors.stub import StubPredictor

        return StubPredictor(model_version=cfg.model_version)

    # artifact_dir = _build_source(cfg).ensure_local()
    #
    # if cfg.model_backend is ModelBackend.TFIDF:
    #     from arxiv_ml.predictors.tfidf import TfidfPredictor
    #
    #     return TfidfPredictor(artifact_dir, model_version=cfg.model_version)

    # if cfg.model_backend is ModelBackend.DISTILBERT:
    #     from arxiv_ml.predictors.distilbert import DistilBertPredictor
    #
    #     return DistilBertPredictor(
    #         artifact_dir,
    #         model_version=cfg.model_version,
    #         device=cfg.inference_device,
    #     )

    raise NotImplementedError(f"Unknown model backend: {cfg.model_backend!r}")
