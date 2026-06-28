from classifier import registry


def test_health_always_ok(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_when_loaded(api_client):
    resp = api_client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


def test_ready_returns_503_before_model_loaded(api_client):
    registry.reset()
    resp = api_client.get("/ready")
    assert resp.status_code == 503
    assert resp.json() == {"status": "not_ready"}
