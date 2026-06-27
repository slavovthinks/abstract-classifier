from dataclasses import dataclass, field
from pathlib import Path

from arxiv_ml.enums import InferenceDevice, ModelBackend, ModelSourceKind


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    model_backend: ModelBackend = ModelBackend.STUB
    model_source: ModelSourceKind = ModelSourceKind.LOCAL
    model_path: Path = field(default_factory=lambda: Path("artifacts"))
    model_version: str = "stub-v1"
    inference_device: InferenceDevice = InferenceDevice.CPU
    max_abstract_chars: int = 20_000
