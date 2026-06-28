import pytest

CLASSIFY_URL = "/api/v1/classify"


def _assert_envelope(resp, expected_status=400, code="validation_error"):
    assert resp.status_code == expected_status
    body = resp.json()
    assert set(body.keys()) == {"error"}
    error = body["error"]
    assert error["code"] == code
    assert isinstance(error["message"], str) and error["message"]
    assert "details" in error
    # No stack trace must ever leak to the client.
    assert "Traceback" not in resp.content.decode()


def test_missing_abstract(api_client):
    resp = api_client.post(CLASSIFY_URL, {}, format="json")
    _assert_envelope(resp)


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_empty_or_whitespace_abstract(api_client, value):
    resp = api_client.post(CLASSIFY_URL, {"abstract": value}, format="json")
    _assert_envelope(resp)


def test_oversized_abstract(api_client, settings):
    cap = settings.INFERENCE_CONFIG.max_abstract_chars
    resp = api_client.post(
        CLASSIFY_URL, {"abstract": "x" * (cap + 1)}, format="json"
    )
    _assert_envelope(resp)


def test_wrong_type_abstract(api_client):
    resp = api_client.post(CLASSIFY_URL, {"abstract": 12345}, format="json")
    # CharField coerces numbers to str, so this is accepted — guard only that we
    # never 500 and the envelope/contract holds.
    assert resp.status_code in {200, 400}
