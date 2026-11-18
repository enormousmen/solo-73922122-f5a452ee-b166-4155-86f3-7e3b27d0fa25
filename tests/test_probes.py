"""Tests for POST /probes (batch-size capacity probes).

The probe shares the engine's single compute channel with regular orders.
These tests focus on the required interleavings:

* timing payload and no order/capacity side effects
* only one concurrent probe (409), retry after failure and after completion
* accepted pending orders are drained before every probe trial ("orders first")
* a running batch is never preempted by a probe ("no preemption")
* submit / query / cancel stay responsive while a probe is computing
* probe samples use exactly the same preprocessing as /predict uploads
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.main import create_app
from app.preprocessing import preprocess
from app.samples import load_sample_tensors

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
MODEL = Path(__file__).resolve().parent.parent / "app" / "model.pt"
SAMPLE_NAMES = ["sample_gradient.png", "sample_circles.png", "sample_stripes.png", "sample_noise.png"]


def make_settings(**kw) -> Settings:
    kw.setdefault("model_path", str(MODEL))
    kw.setdefault("samples_dir", str(SAMPLES))
    return Settings(**kw)


class Client:
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
        body = {}
        while time.monotonic() < deadline:
            body = await self.get(order_id)
            if body["status"] in states:
                return body
            await asyncio.sleep(0.01)
        raise AssertionError(f"order {order_id} never reached {states}: {body}")

    async def probe(self):
        return await self.http.post("/probes")


@pytest.fixture
async def make_client():
    clients = []

    async def factory(**settings_kw):
        app = create_app(make_settings(**settings_kw))
        from asgi_lifespan import LifespanManager

        cm = LifespanManager(app)
        await cm.__aenter__()
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=30)
        client = Client(http, app)
        clients.append((client, cm))
        return client

    yield factory
    for client, cm in clients:
        await client.http.aclose()
        await cm.__aexit__(None, None, None)


# --------------------------------------------------------------------- #
# basic payload and isolation
# --------------------------------------------------------------------- #
async def test_probe_returns_timings_for_1_2_4_without_side_effects(make_client):
    c = await make_client()
    r = await c.probe()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sample_count"] == 4
    assert body["batch_sizes"] == [1, 2, 4]
    assert [t["batch_size"] for t in body["trials"]] == [1, 2, 4]

    for trial in body["trials"]:
        n = trial["batch_size"]
        assert trial["images"] == n
        assert trial["compute_ms"] > 0
        assert trial["forward_ms"] > 0
        # throughput must be consistent with the reported wall-clock compute
        # time; both are rounded to 3 decimals and the box can be noisy, so a
        # loose relative tolerance is used.
        expected = n / (trial["compute_ms"] / 1000.0)
        assert abs(trial["throughput_images_per_s"] - expected) / expected < 0.02
        # pure model forward cannot exceed the wall-clock channel occupancy
        assert trial["forward_ms"] <= trial["compute_ms"] + 1.0

    # A probe creates no orders and occupies none of the 16 capacity slots.
    stats = (await c.http.get("/stats")).json()
    assert stats["capacity_used"] == 0
    assert stats["queued"] == 0
    assert stats["inflight"] == 0
    assert stats["probe_active"] is False
    assert stats["orders"] == {}
    # Probe trials are not counted as order batches.
    assert stats["batches_started"] == 0


async def test_probe_can_be_repeated(make_client):
    c = await make_client()
    first = await c.probe()
    second = await c.probe()
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["probe_id"] != second.json()["probe_id"]


async def test_probe_samples_use_predict_preprocessing(make_client):
    c = await make_client()
    # The probe loader sorts filenames; compare against the same sorted order,
    # each tensor produced by the exact preprocess() used on /predict uploads.
    sorted_names = sorted(SAMPLE_NAMES)
    direct = {name: preprocess((SAMPLES / name).read_bytes()) for name in sorted_names}
    probe_tensors = load_sample_tensors(str(SAMPLES), min_images=4)
    assert len(probe_tensors) == 4
    for name, tensor in zip(sorted_names, probe_tensors):
        ref = direct[name]
        assert tensor.shape == ref.shape == (3, 64, 64)
        assert float((tensor - ref).abs().sum()) == 0.0


# --------------------------------------------------------------------- #
# concurrency: one active probe, 409 otherwise
# --------------------------------------------------------------------- #
async def test_concurrent_probe_rejected_409_and_retryable(make_client):
    c = await make_client(inference_delay_ms=300)  # each probe run takes >1s
    first = asyncio.create_task(c.probe())
    await asyncio.sleep(0.15)  # probe registered, warmup/trials in progress

    second = await c.probe()
    assert second.status_code == 409, second.text

    body = await first
    assert body.status_code == 200, body.text

    # After it finishes the slot is free and a retry succeeds.
    retry = await c.probe()
    assert retry.status_code == 200, retry.text
    stats = (await c.http.get("/stats")).json()
    assert stats["probe_active"] is False


# --------------------------------------------------------------------- #
# ordering: accepted pending orders are processed before probe trials
# --------------------------------------------------------------------- #
async def test_probe_drains_pending_orders_first(make_client):
    c = await make_client(inference_delay_ms=200)
    # Three accepted orders sitting in the queue when the probe starts.
    ids = []
    for i in range(3):
        r = await c.submit(SAMPLE_NAMES[i])
        assert r.status_code == 201
        ids.append(r.json()["order_id"])

    probe_task = asyncio.create_task(c.probe())
    # All accepted orders must complete before the probe response is returned,
    # even though the probe runs three trials plus a warmup on the same channel.
    for oid in ids:
        body = await c.wait_state(oid, {"completed", "cancelled", "expired"})
        assert body["status"] == "completed", body
    probe = await probe_task
    assert probe.status_code == 200, probe.text

    # Orders went through the normal batch machinery; probe trials did not.
    stats = (await c.http.get("/stats")).json()
    assert stats["batches_started"] >= 1
    assert stats["capacity_used"] == 0


async def test_order_during_probe_is_served_between_trials(make_client):
    c = await make_client(inference_delay_ms=250)
    probe_task = asyncio.create_task(c.probe())
    await asyncio.sleep(0.2)  # warmup/early trial in progress

    # Submitted while a probe is active: must be accepted (no capacity used by
    # the probe) and completed before the probe finishes, i.e. drained before
    # one of the remaining trials.
    r = await c.submit("sample_gradient.png")
    assert r.status_code == 201
    oid = r.json()["order_id"]
    body = await c.wait_state(oid, {"completed"})
    assert body["batch_id"] is not None  # went through a real order batch

    probe = await probe_task
    assert probe.status_code == 200


async def test_probe_does_not_preempt_running_order_batch(make_client):
    c = await make_client(inference_delay_ms=400)
    # Start one order batch first and let it enter compute.
    r = await c.submit("sample_gradient.png")
    oid = r.json()["order_id"]
    await asyncio.sleep(0.15)
    assert (await c.get(oid))["status"] == "processing"
    engine = c.app.state.engine
    assert engine._inflight == 1

    t0 = time.monotonic()
    probe_task = asyncio.create_task(c.probe())
    await asyncio.sleep(0.05)
    # The order batch is still the one computing: it cannot be preempted.
    assert engine._inflight == 1
    assert (await c.get(oid))["status"] == "processing"

    order_done = asyncio.create_task(c.wait_state(oid, {"completed"}))
    done, pending = await asyncio.wait({order_done}, timeout=0.5)
    assert done, "running order batch must finish ahead of probe compute"
    await order_done

    probe = await probe_task
    assert probe.status_code == 200
    # Probe took at least one uninterrupted order-batch delay before its trials.
    assert time.monotonic() - t0 >= 0.35


# --------------------------------------------------------------------- #
# failure: slot released, 500 surfaced, immediate retry works
# --------------------------------------------------------------------- #
async def test_failed_probe_releases_slot_and_is_retryable(make_client):
    c = await make_client()
    engine = c.app.state.engine

    def boom(self, batch_tensor):
        raise RuntimeError("simulated model failure")

    engine._infer_timed = boom.__get__(engine, type(engine))

    r = await c.probe()
    assert r.status_code == 500, r.text
    assert engine.probe_active() is False

    # Restore and retry: the failure must not poison the compute channel.
    from app.engine import BatchEngine

    engine._infer_timed = BatchEngine._infer_timed.__get__(engine, BatchEngine)
    retry = await c.probe()
    assert retry.status_code == 200, retry.text

    # Regular inference still works on the same channel afterwards.
    ok = await c.submit("sample_gradient.png")
    body = await c.wait_state(ok.json()["order_id"], {"completed"})
    assert body["status"] == "completed"


# --------------------------------------------------------------------- #
# probe compute never blocks the API and never consumes capacity
# --------------------------------------------------------------------- #
async def test_api_responsive_and_capacity_available_during_probe(make_client):
    c = await make_client(inference_delay_ms=300)
    probe_task = asyncio.create_task(c.probe())
    await asyncio.sleep(0.2)

    async def timed(coro):
        t0 = time.monotonic()
        res = await coro
        return res, time.monotonic() - t0

    # Fill every capacity slot while the probe is running: the probe itself
    # holds zero slots, so all 16 orders are accepted.
    ids = []
    for i in range(16):
        resp, dt = await timed(c.submit(SAMPLE_NAMES[i % 4], deadline_ms=60_000))
        assert dt < 0.5
        assert resp.status_code == 201, (i, resp.text)
        ids.append(resp.json()["order_id"])

    rejected, _ = await timed(c.submit("sample_gradient.png"))
    assert rejected.status_code == 429  # real capacity limit still enforced

    query, dt = await timed(c.http.get(f"/orders/{ids[0]}"))
    cancel, dt2 = await timed(c.cancel(ids[-1]))
    stats, dt3 = await timed(c.http.get("/stats"))
    for dt in (dt, dt2, dt3):
        assert dt < 0.5, "API blocked while probe compute was running"
    assert query.status_code == 200
    assert cancel.status_code == 200
    assert stats.json()["probe_active"] is True

    probe = await probe_task
    assert probe.status_code == 200

    # Everything the probe touched must leave capacity accounting untouched.
    final = (await c.http.get("/stats")).json()
    assert final["probe_active"] is False
    assert final["capacity_used"] <= 15  # 16 submitted, one cancelled
