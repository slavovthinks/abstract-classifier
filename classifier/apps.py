import logging
import os

from django.apps import AppConfig
from django.conf import settings

from arxiv_ml.loader import build_predictor
from classifier import registry

logger = logging.getLogger("classifier")


class ClassifierConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "classifier"

    def ready(self) -> None:
        # Avoid loading the model in the autoreloader's parent process; the
        # child (RUN_MAIN) is the one that actually serves.
        if os.environ.get("RUN_MAIN") == "false":
            return

        cfg = settings.INFERENCE_CONFIG
        logger.info(
            "loading predictor",
            extra={"backend": str(cfg.model_backend), "version": cfg.model_version},
        )
        predictor = build_predictor(cfg)

        # Warm-up inference: the first call is always the slowest, so do it now
        # and only flip /ready to 200 once the model can actually serve.
        predictor.predict("warm-up inference at startup", top_k=1)
        registry.set_ready_predictor(predictor)
        logger.info(
            "predictor ready",
            extra={"backend": str(cfg.model_backend), "version": cfg.model_version},
        )
