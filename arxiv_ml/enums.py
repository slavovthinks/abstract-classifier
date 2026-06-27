from enum import StrEnum


class Category(StrEnum):
    CS = "cs"
    MATH = "math"
    PHYSICS = "physics"
    Q_BIO = "q-bio"
    Q_FIN = "q-fin"
    STAT = "stat"
    EESS = "eess"
    ECON = "econ"


class ModelBackend(StrEnum):
    STUB = "stub"
    TFIDF = "tfidf"
    DISTILBERT = "distilbert"


class ModelSourceKind(StrEnum):
    LOCAL = "local"
    HF = "hf"
    S3 = "s3"


class InferenceDevice(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"
