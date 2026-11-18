"""End-to-end tests for the micro-batch classifier API.

Each test gets a fresh app/engine via the `client` factory. Inference delay
is injected through Settings to simulate a slow model without sleeping the
event loop.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager

from app.config import Settings
from app.main import create_app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
MODEL = Path(__file__).resolve().parent.parent / "app" / "model.pt"
SAMPLE_NAMES = ["sample_gradient.png", "sample_circles.png", "sample_stripes.png", "sample_noise.png"]


def make_settings(**kw) -> Settings:
    kw.setdefault("model_path", str(MODEL))
    return Settings(**kw)


class Client:
    """httpx client wrapper with helpers."""

    def __init__(self, http: httpx.AsyncClient, app):
        self.http = http
        self.app = app

    async def submit(self, name: str, deadline_ms: int = 60_000):
        data = (SAMPLES / name).read_bytes()
        return await self.http.post(
            "/predict",
            files={"file": (name, data, "image/png")},
            data={"deadline_ms": str(deadline_ms)},
        )

    async def get(self, order_id: str):
        return (await self.http.get(f"/orders/{order_id}")).json()

    async def cancel(self, order_id: str):
        return await self.http.post(f"/orders/{order_id}/cancel")

    async def wait_state(self, order_id: str, states, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = await self.get(order_id)
            if body["status"] in states:
                return body
            await asyncio.sleep(0.01)
        raise AssertionError(f"order {order_id} never reached {states}: {body}")


@pytest.fixture
async def make_client():
    clients = []

    async def factory(**settings_kw):
        app = create_app(make_settings(**settings_kw))
        cm = LifespanManager(app)
        await cm.__aenter__()
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
        client = Client(http, app)
        clients.append((client, cm))
        return client

    yield factory
    for client, cm in clients:
        await client.http.aclose()
        await cm.__aexit__(None, None, None)


# --------------------------------------------------------------------- #
# basic flow
# --------------------------------------------------------------------- #
async def test_single_prediction_completes(make_client):
    c = await make_client()
    r = await c.submit("sample_gradient.png", deadline_ms=5000)
    assert r.status_code == 201, r.text
    oid = r.json()["order_id"]
    body = await c.wait_state(oid, {"completed"})
    scores = body["result"]["scores"]
    assert len(scores) == 10
    assert abs(sum(scores) - 1.0) < 1e-4
    assert 0 <= body["result"]["predicted_class"] < 10


async def test_validation_and_404(make_client):
    c = await make_client()
    # non-positive deadline rejected
    r = await c.submit("sample_gradient.png", deadline_ms=0)
    assert r.status_code == 400
    # garbage bytes rejected
    r = await c.http.post(
        "/predict",
        files={"file": ("x.bin", b"not an image", "application/octet-stream")},
        data={"deadline_ms": "1000"},
    )
    assert r.status_code == 400
    # unknown order
    assert (await c.http.get("/orders/nope")).status_code == 404
    assert (await c.http.post("/orders/nope/cancel")).status_code == 404


# --------------------------------------------------------------------- #
# batching correctness: batched scores match per-image inference <= 1e-5
# --------------------------------------------------------------------- #
async def test_batch_scores_match_single_inference(make_client):
    c = await make_client()

    # reference: each image inferred alone
    single_scores = {}
    for name in SAMPLE_NAMES:
        r = await c.submit(name)
        body = await c.wait_state(r.json()["order_id"], {"completed"})
        single_scores[name] = body["result"]["scores"]

    # batched: 4 concurrent submissions must ride one batch of 4
    responses = await asyncio.gather(*[c.submit(n) for n in SAMPLE_NAMES])
    ids = [r.json()["order_id"] for r in responses]
    bodies = await asyncio.gather(*[c.wait_state(i, {"completed"}) for i in ids])
    batch_ids = {b["batch_id"] for b in bodies}
    assert len(batch_ids) == 1, f"expected one shared batch, got {batch_ids}"
    assert c.app.state.engine.last_batch_size == 4

    for name, body in zip(SAMPLE_NAMES, bodies):
        for a, b in zip(body["result"]["scores"], single_scores[name]):
            assert abs(a - b) <= 1e-5, f"{name}: {a} vs {b}"


async def test_idle_model_waits_to_fill_batch(make_client):
    c = await make_client(batch_wait_ms=200)
    r1 = await c.submit("sample_gradient.png")
    await asyncio.sleep(0.05)  # inside the batch window
    r2 = await c.submit("sample_circles.png")
    b1 = await c.wait_state(r1.json()["order_id"], {"completed"})
    b2 = await c.wait_state(r2.json()["order_id"], {"completed"})
    assert b1["batch_id"] == b2["batch_id"]
    assert c.app.state.engine.last_batch_size == 2


# --------------------------------------------------------------------- #
# capacity: 16 queued+in-flight, then 429
# --------------------------------------------------------------------- #
async def test_capacity_full_rejects_429(make_client):
    c = await make_client(inference_delay_ms=1500)
    accepted = []
    for i in range(16):
        r = await c.submit(SAMPLE_NAMES[i % 4])
        assert r.status_code == 201, (i, r.text)
        accepted.append(r.json()["order_id"])
    r = await c.submit("sample_gradient.png")
    assert r.status_code == 429, r.text
    stats = (await c.http.get("/stats")).json()
    assert stats["capacity_used"] == 16


# --------------------------------------------------------------------- #
# cancellation
# --------------------------------------------------------------------- #
async def test_cancel_queued_order_never_enters_model(make_client):
    c = await make_client(inference_delay_ms=1200)
    ids = []
    for i in range(5):  # 4 fill the running batch, 5th stays queued
        r = await c.submit(SAMPLE_NAMES[i % 4])
        ids.append(r.json()["order_id"])
    await asyncio.sleep(0.2)  # let the first batch start
    victim = ids[4]
    body = await c.get(victim)
    assert body["status"] == "queued"
    r = await c.cancel(victim)
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    # after everything settles it must still be cancelled, with no result/batch
    await asyncio.sleep(1.6)
    body = await c.get(victim)
    assert body["status"] == "cancelled"
    assert body["result"] is None
    assert body["batch_id"] is None


async def test_cancel_processing_marks_now_but_holds_capacity(make_client):
    c = await make_client(inference_delay_ms=1500)
    ids = []
    for i in range(4):  # exactly one batch
        r = await c.submit(SAMPLE_NAMES[i])
        ids.append(r.json()["order_id"])
    await asyncio.sleep(0.2)  # batch is now computing
    victim, others = ids[0], ids[1:]
    assert (await c.get(victim))["status"] == "processing"

    r = await c.cancel(victim)
    assert r.json()["status"] == "cancelled"  # marked immediately
    assert (await c.get(victim))["status"] == "cancelled"

    # slot is still occupied while the batch computes: only 12 free slots
    extra = []
    for _ in range(12):
        r = await c.submit("sample_gradient.png")
        assert r.status_code == 201
        extra.append(r.json()["order_id"])
    r = await c.submit("sample_gradient.png")
    assert r.status_code == 429, "cancelled in-flight order must still occupy capacity"

    # no new batch may start while the cancelled one's batch is computing
    assert c.app.state.engine.batches_started == 1

    # after compute ends: victim stays cancelled without result, others complete
    await asyncio.sleep(1.6)
    body = await c.get(victim)
    assert body["status"] == "cancelled" and body["result"] is None
    for oid in others:
        assert (await c.get(oid))["status"] == "completed"


async def test_cancel_completed_order_conflicts(make_client):
    c = await make_client()
    r = await c.submit("sample_gradient.png")
    oid = r.json()["order_id"]
    await c.wait_state(oid, {"completed"})
    r = await c.cancel(oid)
    assert r.status_code == 409


# --------------------------------------------------------------------- #
# expiry
# --------------------------------------------------------------------- #
async def test_queued_order_expires_and_never_enters_model(make_client):
    c = await make_client(inference_delay_ms=1200)
    ids = []
    for i in range(4):  # occupy the model with one batch
        r = await c.submit(SAMPLE_NAMES[i])
        ids.append(r.json()["order_id"])
    await asyncio.sleep(0.2)
    r = await c.submit("sample_gradient.png", deadline_ms=300)  # expires while queued
    victim = r.json()["order_id"]
    await asyncio.sleep(0.5)
    body = await c.get(victim)
    assert body["status"] == "expired"
    await asyncio.sleep(1.5)  # let the running batch finish and next batch form
    body = await c.get(victim)
    assert body["status"] == "expired"
    assert body["result"] is None
    assert body["batch_id"] is None  # never entered the model


async def test_processing_order_expires_midflight_no_result(make_client):
    c = await make_client(inference_delay_ms=1000)
    r = await c.submit("sample_gradient.png", deadline_ms=400)
    oid = r.json()["order_id"]
    await asyncio.sleep(1.6)  # batch computed (late), deadline long past
    body = await c.get(oid)
    assert body["status"] == "expired"
    assert body["result"] is None


async def test_late_result_does_not_revive_cancelled_order(make_client):
    c = await make_client(inference_delay_ms=800)
    r = await c.submit("sample_gradient.png", deadline_ms=60_000)
    oid = r.json()["order_id"]
    await asyncio.sleep(0.2)  # in-flight now
    await c.cancel(oid)
    await asyncio.sleep(1.2)  # batch compute finished, result arrived late
    body = await c.get(oid)
    assert body["status"] == "cancelled"
    assert body["result"] is None


# --------------------------------------------------------------------- #
# slow inference must not block submit / query / cancel
# --------------------------------------------------------------------- #
async def test_slow_inference_does_not_block_api(make_client):
    c = await make_client(inference_delay_ms=2000)
    r = await c.submit("sample_gradient.png")
    oid = r.json()["order_id"]
    await asyncio.sleep(0.2)  # batch is computing for ~2s

    async def timed(coro):
        t0 = time.monotonic()
        res = await coro
        return res, time.monotonic() - t0

    (resp, dt1) = await timed(c.http.get(f"/orders/{oid}"))
    (resp2, dt2) = await timed(c.submit("sample_circles.png"))
    (resp3, dt3) = await timed(c.cancel(resp2.json()["order_id"]))
    (resp4, dt4) = await timed(c.http.get("/stats"))
    for dt in (dt1, dt2, dt3, dt4):
        assert dt < 0.5, f"API blocked for {dt:.3f}s during slow inference"
    assert resp.status_code == 200
    assert resp2.status_code == 201
    assert resp3.status_code == 200
    assert resp4.status_code == 200
