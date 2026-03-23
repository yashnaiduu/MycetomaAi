import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.services.explanation_service import ExplanationService

router = APIRouter()
logger = logging.getLogger(__name__)

_explanation_service = ExplanationService()


class ExplainRequest(BaseModel):
    class_name: str
    confidence: float
    subtype: str
    probabilities: dict


class ExplainResponse(BaseModel):
    explanation: str
    cached: bool


@router.post("/", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    try:
        result = await _explanation_service.generate(
            class_name=request.class_name,
            confidence=request.confidence,
            subtype=request.subtype,
            probabilities=request.probabilities,
        )
        return ExplainResponse(**result)
    except Exception as e:
        logger.exception("Explanation failed")
        raise HTTPException(status_code=500, detail=str(e))
