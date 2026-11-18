"""FastAPI HTTP layer for the micro-batch image classifier.

Endpoints
---------
POST /predict                 submit an image + deadline_ms (from acceptance)
GET  /orders/{order_id}       query status / result by order id
POST /orders/{order_id}/cancel  cancel a queued or in-flight order
POST /probes                  time the fixed model on sample batches of 1, 2, 4
GET  /stats                   capacity and state counters (demo/observability)
GET  /health                  liveness
Interactive docs are served at /docs (Swagger UI).
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import Settings
from .engine import COMPLETED, PROBE_BATCH_SIZES, BatchEngine, Order
from .preprocessing import preprocess
from .samples import SampleLoadError, load_sample_tensors


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
        version="1.1.0",
        description=(
            "Time-limited image classification for multiple cameras. "
            "Single-process micro-batching: at most 16 queued/in-flight images, "
            "batches of up to 4, idle model waits up to 50 ms to fill a batch, "
            "one batch infers at a time. Cancelled/expired orders never yield results. "
            "POST /probes time-trials the fixed model on sample batches of 1, 2 and 4 "
            "images on the same compute channel, without creating orders or using capacity."
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

    @app.post("/probes", status_code=200)
    async def create_probe():
        """Time the fixed model on sample-image batches of 1, 2 and 4.

        The bundled samples are preprocessed exactly like `/predict` uploads.
        Probes share the single compute channel with regular inference:
        accepted pending orders are processed first, a running batch is never
        preempted, and probes create no orders and consume no capacity slot.
        Only one probe at a time is allowed (409 otherwise); after a failure
        or client disconnect the probe can simply be retried.
        """
        # Fast rejection while another probe is running. The engine re-checks
        # atomically inside start_probe, so a concurrent pair cannot both win.
        if engine.probe_active():
            raise HTTPException(409, "another probe is already running; retry after it finishes")

        try:
            sample_tensors = load_sample_tensors(
                settings.samples_dir, min_images=max(PROBE_BATCH_SIZES)
            )
        except SampleLoadError as exc:
            raise HTTPException(500, f"probe unavailable: {exc}") from exc

        future = engine.start_probe(sample_tensors, PROBE_BATCH_SIZES)
        if future is None:
            # Lost a near-simultaneous race with another probe.
            raise HTTPException(409, "another probe is already running; retry after it finishes")

        try:
            return await future
        except asyncio.CancelledError:
            # Client went away: tell the worker to abandon the probe so the
            # next POST /probes can start immediately after the current
            # (uninterruptible) compute finishes.
            engine.cancel_probe()
            raise
        except Exception as exc:
            # The probe slot is already released by the engine, so the client
            # may retry immediately with another POST /probes.
            raise HTTPException(500, f"probe failed: {exc}") from exc

    @app.get("/stats")
    async def stats():
        return engine.stats()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
