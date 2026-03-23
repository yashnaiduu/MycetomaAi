import time

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float


@router.get("/health", response_model=HealthResponse)
async def health():
    from backend.api.main import app_state

    engine = app_state.get("engine")
    start_time = app_state.get("start_time", time.time())

    return HealthResponse(
        status="ok" if engine else "degraded",
        model_loaded=engine is not None,
        device=str(engine.device) if engine else "N/A",
        uptime_seconds=round(time.time() - start_time, 2),
    )
