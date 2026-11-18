"""FastAPI HTTP layer for the micro-batch image classifier.

Endpoints
---------
POST /predict                 submit an image + deadline_ms (from acceptance)
GET  /orders/{order_id}       query status / result by order id
POST /orders/{order_id}/cancel  cancel a queued or in-flight order
GET  /stats                   capacity and state counters (demo/observability)
GET  /health                  liveness
Interactive docs are served at /docs (Swagger UI).
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import Settings
from .engine import COMPLETED, BatchEngine, Order
from .preprocessing import preprocess


def order_view(o: Order) -> dict:
    remaining_ms = max(0, round((o.deadline_at - time.monotonic()) * 1000))
    return {
        "order_id": o.id,
        "status": o.state,
        "deadline_ms": o.deadline_ms,
        "remaining_ms": remaining_ms,
        "batch_id": o.batch_id,
        "result": o.result if o.state == COMPLETED else None,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    engine = BatchEngine(
        model_path=settings.model_path,
        max_capacity=settings.max_capacity,
        batch_size=settings.batch_size,
        batch_wait_ms=settings.batch_wait_ms,
        inference_delay_ms=settings.inference_delay_ms,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start()
        yield
        await engine.stop()

    app = FastAPI(
        title="Micro-batch Image Classifier",
        version="1.0.0",
        description=(
            "Time-limited image classification for multiple cameras. "
            "Single-process micro-batching: at most 16 queued/in-flight images, "
            "batches of up to 4, idle model waits up to 50 ms to fill a batch, "
            "one batch infers at a time. Cancelled/expired orders never yield results."
        ),
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.settings = settings

    @app.post("/predict", status_code=201)
    async def predict(file: UploadFile = File(...), deadline_ms: int = Form(...)):
        """Submit an image with a deadline in milliseconds counted from acceptance."""
        if deadline_ms <= 0:
            raise HTTPException(400, "deadline_ms must be a positive integer")
        data = await file.read()
        try:
            tensor = preprocess(data)
        except ValueError as exc:
            raise HTTPException(400, f"invalid image: {exc}") from exc
        order = engine.submit(tensor, deadline_ms)
        if order is None:
            raise HTTPException(
                429,
                f"capacity full: at most {engine.max_capacity} images queued or in-flight",
            )
        return order_view(order)

    @app.get("/orders/{order_id}")
    async def get_order(order_id: str):
        order = engine.get(order_id)
        if order is None:
            raise HTTPException(404, "unknown order_id")
        return order_view(order)

    @app.post("/orders/{order_id}/cancel")
    async def cancel_order(order_id: str):
        order = engine.cancel(order_id)
        if order is None:
            raise HTTPException(404, "unknown order_id")
        if order.state == COMPLETED:
            raise HTTPException(409, "order already completed; cannot cancel")
        return order_view(order)

    @app.get("/stats")
    async def stats():
        return engine.stats()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
