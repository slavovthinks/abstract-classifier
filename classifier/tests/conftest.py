import pytest
from rest_framework.test import APIClient

from arxiv_ml.predictors.stub import StubPredictor
from classifier import registry


@pytest.fixture
def api_client() -> APIClient:
    return APIClient()


@pytest.fixture(autouse=True)
def stub_predictor():
    """Guarantee a loaded, ready StubPredictor for every test, restored after."""
    registry.set_ready_predictor(StubPredictor(model_version="stub-test-v1"))
    yield
    registry.reset()
