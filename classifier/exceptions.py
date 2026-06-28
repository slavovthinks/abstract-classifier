"""Consistent error envelope for all API responses (see docs/implementation-plan.md §9).

Every error — DRF validation, known API exceptions, or an unexpected crash — is
shaped into ``{"error": {"code", "message", "details"}}``. Stack traces are
logged server-side only and never leak to the client.
"""

import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

logger = logging.getLogger("classifier")


def _envelope(code: str, message: str, details=None, *, status_code: int) -> Response:
    return Response(
        {"error": {"code": code, "message": message, "details": details or {}}},
        status=status_code,
    )


def exception_handler(exc, context):
    response = drf_exception_handler(exc, context)

    if response is None:
        # Unhandled exception: log full detail server-side, return an opaque 500.
        logger.exception("unhandled exception in %s", context.get("view"))
        return _envelope(
            "internal_error",
            "An unexpected error occurred.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    if response.status_code == status.HTTP_400_BAD_REQUEST:
        code = "validation_error"
        message = "Request validation failed."
    else:
        code = getattr(exc, "default_code", "error")
        message = str(getattr(exc, "detail", exc)) or "Request could not be processed."

    return _envelope(
        code, message, details=response.data, status_code=response.status_code
    )
