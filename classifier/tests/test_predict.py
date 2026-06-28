from arxiv_ml.enums import Category

CLASSIFY_URL = "/api/v1/classify"
VALID_LABELS = {c.value for c in Category}


def test_classify_happy_path(api_client):
    resp = api_client.post(
        CLASSIFY_URL,
        {"abstract": "We present a transformer-based approach to classification."},
        format="json",
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["predicted_category"] in VALID_LABELS
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"] == "stub-test-v1"

    assert len(body["top_k"]) == 3
    for item in body["top_k"]:
        assert item["category"] in VALID_LABELS
        assert 0.0 <= item["confidence"] <= 1.0

    # top_k is sorted descending and the top entry matches the prediction.
    confidences = [item["confidence"] for item in body["top_k"]]
    assert confidences == sorted(confidences, reverse=True)
    assert body["top_k"][0]["category"] == body["predicted_category"]
    assert body["top_k"][0]["confidence"] == body["confidence"]


def test_classify_is_deterministic(api_client):
    payload = {"abstract": "Stable input yields a stable label."}
    first = api_client.post(CLASSIFY_URL, payload, format="json").json()
    second = api_client.post(CLASSIFY_URL, payload, format="json").json()
    assert first == second


def test_classify_sets_request_id_header(api_client):
    resp = api_client.post(
        CLASSIFY_URL, {"abstract": "anything"}, format="json"
    )
    assert resp.headers.get("X-Request-ID")
