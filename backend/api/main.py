import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.src.utils.config import load_config
from backend.src.inference.engine import InferenceEngine
from backend.api.routes import predict, explain, health

logger = logging.getLogger(__name__)

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config("api")

    log_cfg = cfg.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s - %(levelname)s - %(message)s"),
    )

    logger.info("Loading inference engine...")
    ckpt = cfg.get("inference", {}).get("model_path")
    app_state["engine"] = InferenceEngine(checkpoint_path=ckpt)
    app_state["start_time"] = time.time()
    logger.info("Engine ready")

    yield

    app_state.clear()


app = FastAPI(
    title="Mycetoma AI Diagnostics",
    version="1.0.0",
    lifespan=lifespan,
)

cfg = load_config("api")
origins = cfg.get("server", {}).get("cors_origins", ["*"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(explain.router, prefix="/explain", tags=["Explanation"])
app.include_router(health.router, tags=["Health"])
