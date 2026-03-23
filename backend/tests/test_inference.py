import numpy as np
import io
import torch
import pytest
from PIL import Image

from backend.src.inference.engine import InferenceEngine


def _make_dummy_image() -> bytes:
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestInferenceEngine:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = InferenceEngine(checkpoint_path=None)

    def test_preprocess_shape(self):
        img_bytes = _make_dummy_image()
        tensor, rgb = self.engine.preprocess(img_bytes)
        assert tensor.shape == (1, 3, 224, 224)
        assert rgb.shape == (224, 224, 3)

    def test_predict_keys(self):
        img_bytes = _make_dummy_image()
        result = self.engine.predict(img_bytes)
        expected_keys = {"class_name", "class_id", "confidence", "probabilities",
                         "bounding_box", "subtype", "subtype_id", "heatmap", "heatmap_raw"}
        assert expected_keys.issubset(result.keys())

    def test_predict_confidence_range(self):
        img_bytes = _make_dummy_image()
        result = self.engine.predict(img_bytes)
        assert 0 <= result["confidence"] <= 1
