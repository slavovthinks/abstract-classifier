import logging
import time
from dataclasses import asdict

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from classifier import registry
from classifier.serializers import (
    ClassifyRequestSerializer,
    PredictionResponseSerializer,
)

logger = logging.getLogger("classifier")


class ClassifyView(APIView):
    """POST /api/v1/classify — classify an arXiv abstract into a category group."""

    def post(self, request):
        request_serializer = ClassifyRequestSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        abstract = request_serializer.validated_data["abstract"]

        predictor = registry.get_predictor()
        start = time.perf_counter()
        prediction = predictor.predict(abstract)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "classification",
            extra={
                "inference_latency_ms": latency_ms,
                "predicted_category": str(prediction.predicted_category),
                "confidence": round(prediction.confidence, 4),
                "model_version": prediction.model_version,
            },
        )

        response_serializer = PredictionResponseSerializer(asdict(prediction))
        return Response(response_serializer.data, status=status.HTTP_200_OK)


@api_view(["GET"])
def health(request):
    """Liveness: the process is up. Always 200."""
    return Response({"status": "ok"}, status=status.HTTP_200_OK)


@api_view(["GET"])
def ready(request):
    """Readiness: 200 once the model is loaded and warmed up, else 503."""
    if registry.is_ready():
        return Response({"status": "ready"}, status=status.HTTP_200_OK)
    return Response(
        {"status": "not_ready"}, status=status.HTTP_503_SERVICE_UNAVAILABLE
    )
