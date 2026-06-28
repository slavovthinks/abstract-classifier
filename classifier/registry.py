"""Process-wide predictor singleton and readiness flag.

The model is expensive to load, so it is built once at startup (see
:mod:`classifier.apps`) and held here for the lifetime of the worker process.
"""

import threading

from arxiv_ml.contract import Predictor

_lock = threading.Lock()
_predictor: Predictor | None = None
_ready: bool = False


def set_ready_predictor(predictor: Predictor) -> None:
    """Publish a predictor only after it has been loaded and warmed up."""
    global _predictor, _ready
    with _lock:
        _predictor = predictor
        _ready = True


def get_predictor() -> Predictor:
    with _lock:
        if _predictor is None:
            raise RuntimeError("Predictor is not loaded yet.")
        return _predictor


def is_ready() -> bool:
    with _lock:
        return _ready and _predictor is not None


def reset() -> None:
    """Test hook: clear the singleton between cases."""
    global _predictor, _ready
    with _lock:
        _predictor = None
        _ready = False
