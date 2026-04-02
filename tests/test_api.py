"""
tests/test_api.py
-----------------
Integration tests for the FastAPI endpoints using TestClient.
The Predictor is patched to use the stub ensemble so tests
run without a trained model on disk.
"""

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def patch_predictor():
    """Replace Predictor.from_config with a stub so API tests are self-contained."""
    from src.api.predictor import _StubEnsemble, Predictor

    stub_predictor = Predictor(_StubEnsemble(), model_version="test-stub")

    with patch("src.api.main.Predictor") as MockPredictor:
        MockPredictor.from_config.return_value = stub_predictor
        yield


@pytest.fixture()
def client():
    # Import app after patching
    from src.api.main import app, predictor
    import src.api.main as main_module

    # Manually set the global predictor for TestClient (bypasses lifespan)
    from src.api.predictor import _StubEnsemble, Predictor
    main_module.predictor = Predictor(_StubEnsemble(), model_version="test-stub")

    return TestClient(app)


SAMPLE_REQUEST = {
    "patient_id": "P-00042",
    "vitals_window": [
        {
            "timestamp": "2024-01-15T08:00:00Z",
            "hr": 88, "sbp": 122, "dbp": 78,
            "spo2": 97, "rr": 16, "temp": 37.1,
        },
        {
            "timestamp": "2024-01-15T08:30:00Z",
            "hr": 91, "sbp": 118, "dbp": 76,
            "spo2": 96, "rr": 17, "temp": 37.2,
        },
        {
            "timestamp": "2024-01-15T09:00:00Z",
            "hr": 95, "sbp": 115, "dbp": 74,
            "spo2": 95, "rr": 18, "temp": 37.3,
        },
    ],
}


class TestPredictEndpoint:
    def test_returns_200(self, client):
        resp = client.post("/predict", json=SAMPLE_REQUEST)
        assert resp.status_code == 200

    def test_response_schema(self, client):
        resp = client.post("/predict", json=SAMPLE_REQUEST)
        body = resp.json()
        assert "deterioration_probability" in body
        assert "risk_level" in body
        assert "model_version" in body
        assert body["patient_id"] == "P-00042"

    def test_probability_in_range(self, client):
        resp = client.post("/predict", json=SAMPLE_REQUEST)
        prob = resp.json()["deterioration_probability"]
        assert 0.0 <= prob <= 1.0

    def test_risk_level_valid(self, client):
        resp = client.post("/predict", json=SAMPLE_REQUEST)
        assert resp.json()["risk_level"] in {"LOW", "MEDIUM", "HIGH"}

    def test_too_few_readings_returns_422(self, client):
        bad_request = {
            "patient_id": "P-00001",
            "vitals_window": [SAMPLE_REQUEST["vitals_window"][0]],  # only 1 reading
        }
        resp = client.post("/predict", json=bad_request)
        assert resp.status_code == 422

    def test_missing_vital_values_accepted(self, client):
        """Vitals are optional — partial readings should still work."""
        request = {
            "patient_id": "P-00099",
            "vitals_window": [
                {"timestamp": "2024-01-15T08:00:00Z", "hr": 88},
                {"timestamp": "2024-01-15T08:30:00Z", "hr": 91, "spo2": 96},
                {"timestamp": "2024-01-15T09:00:00Z", "hr": 95, "sbp": 115},
            ],
        }
        resp = client.post("/predict", json=request)
        assert resp.status_code == 200


class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"


class TestMetricsEndpoint:
    def test_returns_prometheus_format(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert b"pulseml_requests_total" in resp.content or b"#" in resp.content
