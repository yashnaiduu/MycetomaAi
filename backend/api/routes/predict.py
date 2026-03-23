import io
import base64
import logging

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


class PredictionResponse(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    probabilities: dict
    bounding_box: list
    subtype: str
    subtype_id: int
    heatmap_base64: str


def encode_heatmap(heatmap: np.ndarray) -> str:
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", heatmap_bgr)
    return base64.b64encode(buffer).decode("utf-8")


@router.post("/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    from backend.api.main import app_state
    engine = app_state.get("engine")
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = engine.predict(image_bytes)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    heatmap_b64 = encode_heatmap(result["heatmap"])

    return PredictionResponse(
        class_name=result["class_name"],
        class_id=result["class_id"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        bounding_box=result["bounding_box"],
        subtype=result["subtype"],
        subtype_id=result["subtype_id"],
        heatmap_base64=heatmap_b64,
    )
