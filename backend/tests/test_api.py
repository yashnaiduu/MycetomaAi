import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.api.main import app


def _make_test_image() -> bytes:
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "model_loaded" in data


class TestPredictEndpoint:
    def test_predict_valid(self, client):
        img_bytes = _make_test_image()
        resp = client.post(
            "/predict/",
            files={"file": ("test.png", io.BytesIO(img_bytes), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "class_name" in data
        assert "heatmap_base64" in data

    def test_predict_invalid_file(self, client):
        resp = client.post(
            "/predict/",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert resp.status_code == 400


class TestExplainEndpoint:
    def test_explain(self, client):
        resp = client.post(
            "/explain/",
            json={
                "class_name": "Eumycetoma",
                "confidence": 0.92,
                "subtype": "Madurella mycetomatis",
                "probabilities": {"Eumycetoma": 0.92, "Actinomycetoma": 0.06, "Normal": 0.02},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "explanation" in data
