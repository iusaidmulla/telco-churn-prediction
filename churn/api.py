"""Small FastAPI wrapper around the saved pipeline.

    uvicorn churn.api:app --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churn import __version__
from churn.inference import ChurnPredictor
from churn.schemas import CustomerProfile, PredictionResponse


@asynccontextmanager
async def lifespan(app):
    app.state.predictor = ChurnPredictor.load()  # once per process, not per request
    yield


app = FastAPI(title="Telco churn API", version=__version__, lifespan=lifespan)


@app.exception_handler(RequestValidationError)
def invalid_payload(request, exc):
    # FastAPI's default 422 echoes the rejected value back, and NaN / Infinity can't be
    # written as JSON (that turned into a 500), so only report where and why
    detail = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/health")
def health(request: Request):
    predictor = request.app.state.predictor
    return {
        "status": "ok",
        "model": predictor.metadata.get("model_name"),
        "decision_threshold": predictor.threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(profile: CustomerProfile, request: Request):
    # a plain `def` so FastAPI runs it in a worker thread and the event loop stays free
    return request.app.state.predictor.predict(profile)
