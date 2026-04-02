"""
src/api/main.py
---------------
FastAPI serving layer for PulseML.

Endpoints:
  POST /predict    — real-time deterioration prediction
  GET  /health     — liveness probe
  GET  /metrics    — Prometheus metrics
"""
from __future__ import annotations
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response

from src.api.schemas import PredictRequest, PredictResponse, HealthResponse
from src.api.predictor import Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "pulseml_requests_total",
    "Total prediction requests",
    ["risk_level"],
)
REQUEST_LATENCY = Histogram(
    "pulseml_request_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)
PREDICTION_PROB = Histogram(
    "pulseml_prediction_probability",
    "Distribution of predicted deterioration probabilities",
    buckets=[0.1 * i for i in range(11)],
)
MODEL_VERSION_GAUGE = Gauge("pulseml_model_version_info", "Model version", ["version"])
STARTUP_TIME = time.time()

# ---------------------------------------------------------------------------
# App lifespan: load model once at startup
# ---------------------------------------------------------------------------

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    log.info("Loading model...")
    predictor = Predictor.from_config()
    MODEL_VERSION_GAUGE.labels(version=predictor.model_version).set(1)
    log.info("Model loaded. Version: %s", predictor.model_version)
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="PulseML",
    description="Real-time ICU patient deterioration prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if len(request.vitals_window) < 3:
        raise HTTPException(
            status_code=422,
            detail="At least 3 vital sign readings are required.",
        )

    start = time.perf_counter()

    try:
        result = predictor.predict(request)
    except Exception as e:
        log.exception("Prediction failed for patient %s", request.patient_id)
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.perf_counter() - start

    REQUEST_COUNT.labels(risk_level=result["risk_level"]).inc()
    REQUEST_LATENCY.observe(latency)
    PREDICTION_PROB.observe(result["probability"])

    log.info(
        "patient=%s prob=%.3f risk=%s latency_ms=%.1f",
        request.patient_id,
        result["probability"],
        result["risk_level"],
        latency * 1000,
    )

    return PredictResponse(
        patient_id=request.patient_id,
        deterioration_probability=result["probability"],
        risk_level=result["risk_level"],
        predicted_at=datetime.now(timezone.utc),
        model_version=predictor.model_version,
        top_features=result.get("top_features", []),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_version=predictor.model_version if predictor else "not_loaded",
        uptime_seconds=round(time.time() - STARTUP_TIME, 1),
    )


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)