import logging
import time
import uuid

from classifier.observability import request_id_var

logger = logging.getLogger("classifier.request")


class RequestIDMiddleware:
    """Assign a request ID, expose it on the response, and log each request.

    The ID is taken from an inbound ``X-Request-ID`` header when present (so it
    can be correlated across services) and otherwise generated.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4()
        token = request_id_var.set(request_id)
        request.request_id = request_id
        start = time.perf_counter()
        try:
            response = self.get_response(request)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            response["X-Request-ID"] = request_id
            logger.info(
                "request handled",
                extra={
                    "method": request.method,
                    "path": request.path,
                    "status": response.status_code,
                    "duration_ms": elapsed_ms,
                },
            )
            return response
        finally:
            request_id_var.reset(token)
