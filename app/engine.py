"""Single-process micro-batch inference engine.

Concurrency model
-----------------
* Everything except the blocking model forward pass runs on the FastAPI
  asyncio event loop, so no locks are needed for the shared state below.
* Exactly one worker task forms batches and exactly one executor thread runs
  the TorchScript model, therefore at most ONE batch is ever actually
  inferring at any moment. Regular orders and capacity probes share this
  single compute channel: the worker dispatches both, and the executor
  serializes them (a running batch can never be preempted).
* Capacity accounting: an order occupies a slot while it is queued AND while
  its batch is computing (even if it was cancelled or expired mid-flight).
  The slot is released only when the batch compute actually finishes.
  Probes never create orders and never occupy a capacity slot.

State machine per order
-----------------------
queued -> processing -> completed
queued -> cancelled            (cancelled before entering a batch; never runs)
queued -> expired              (deadline passed while queued; never runs)
processing -> cancelled        (marked immediately, result discarded, slot held)
processing -> expired          (deadline passed mid-flight; result discarded)
processing -> failed           (model raised; defensive only)
Late results are written back ONLY to orders still in `processing` whose
deadline has not passed, so a late result can never revive a cancelled or
expired order.

Capacity probes (POST /probes)
------------------------------
A probe preprocesses the bundled sample images once and asks the worker to
time trial batches of 1, 2 and 4 images on the same model/executor. Before
every trial the worker first drains all already-accepted pending orders
("orders first"); a batch currently computing is awaited to completion
("no preemption"). Only one probe may exist at a time -- starting another
while one is active is rejected with 409. A failed or cancelled probe always
releases the slot and can be retried.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import torch

log = logging.getLogger("batch-engine")

QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
CANCELLED = "cancelled"
EXPIRED = "expired"
FAILED = "failed"

# Batch sizes timed by POST /probes, in execution order.
PROBE_BATCH_SIZES = (1, 2, 4)


# eq=False: orders are compared by identity. The default dataclass __eq__
# would compare tensors element-wise and raise on multi-element tensors.
@dataclass(eq=False)
class Order:
    id: str
    tensor: torch.Tensor
    deadline_ms: int
    deadline_at: float  # time.monotonic() at which the order expires
    state: str = QUEUED
    result: Optional[dict] = None
    batch_id: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)


@dataclass(eq=False)
class ProbeState:
    """One active capacity probe; held outside the order/capacity machinery."""

    id: str
    sample_tensors: list[torch.Tensor]
    batch_sizes: tuple[int, ...]
    future: "asyncio.Future[dict]"
    cancelled: bool = False


class BatchEngine:
    def __init__(
        self,
        model_path: str,
        max_capacity: int = 16,
        batch_size: int = 4,
        batch_wait_ms: int = 50,
        inference_delay_ms: int = 0,
    ) -> None:
        self.max_capacity = max_capacity
        self.batch_size = batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.inference_delay_s = inference_delay_ms / 1000.0

        # Single-threaded CPU inference keeps results deterministic and makes
        # the "one batch at a time" guarantee easy to reason about.
        torch.set_num_threads(1)
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()

        self._pending: deque[Order] = deque()
        self._orders: dict[str, Order] = {}
        self._inflight = 0  # orders inside the batch currently being computed
        # Set whenever the worker has something to do: a new order, a probe
        # request, or a probe finishing and handing control back to orders.
        self._wake = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._worker: Optional[asyncio.Task] = None
        # At most one active probe; None when no probe is registered.
        self._probe: Optional[ProbeState] = None
        # Counters for /stats and tests.
        self.batches_started = 0
        self.last_batch_size = 0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        self._worker = asyncio.create_task(self._run(), name="batch-worker")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------ #
    # helpers (event-loop thread only)
    # ------------------------------------------------------------------ #
    def _purge_expired_pending(self) -> None:
        """Mark queued orders whose deadline passed; they never enter the model."""
        now = time.monotonic()
        kept: list[Order] = []
        for o in self._pending:
            if o.state == QUEUED and o.deadline_at <= now:
                o.state = EXPIRED
            else:
                kept.append(o)
        if len(kept) != len(self._pending):
            self._pending.clear()
            self._pending.extend(kept)

    def _refresh_state(self, o: Order) -> None:
        """Lazy expiry when an order is observed."""
        if o.state in (QUEUED, PROCESSING) and o.deadline_at <= time.monotonic():
            if o.state == QUEUED:
                try:
                    self._pending.remove(o)
                except ValueError:
                    pass
            o.state = EXPIRED

    def capacity_used(self) -> int:
        self._purge_expired_pending()
        return len(self._pending) + self._inflight

    def probe_active(self) -> bool:
        return self._probe is not None

    def stats(self) -> dict:
        self._purge_expired_pending()
        counts: dict[str, int] = {}
        for o in self._orders.values():
            self._refresh_state(o)
            counts[o.state] = counts.get(o.state, 0) + 1
        return {
            "capacity_used": len(self._pending) + self._inflight,
            "max_capacity": self.max_capacity,
            "queued": len(self._pending),
            "inflight": self._inflight,
            "batches_started": self.batches_started,
            "last_batch_size": self.last_batch_size,
            "probe_active": self._probe is not None,
            "orders": counts,
        }

    # ------------------------------------------------------------------ #
    # public API used by the HTTP layer
    # ------------------------------------------------------------------ #
    def submit(self, tensor: torch.Tensor, deadline_ms: int) -> Optional[Order]:
        """Accept an image, or return None when at capacity (16 by default)."""
        if self.capacity_used() >= self.max_capacity:
            return None
        order = Order(
            id=uuid.uuid4().hex[:12],
            tensor=tensor,
            deadline_ms=deadline_ms,
            deadline_at=time.monotonic() + deadline_ms / 1000.0,
        )
        self._orders[order.id] = order
        self._pending.append(order)
        self._wake.set()
        return order

    def get(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is not None:
            self._refresh_state(o)
        return o

    def cancel(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is None:
            return None
        self._refresh_state(o)
        if o.state == QUEUED:
            try:
                self._pending.remove(o)
            except ValueError:
                pass
            o.state = CANCELLED
        elif o.state == PROCESSING:
            # Mark immediately. The slot stays occupied until the batch
            # compute actually ends; no new batch is started because of it.
            o.state = CANCELLED
        return o

    def start_probe(
        self,
        sample_tensors: list[torch.Tensor],
        batch_sizes: tuple[int, ...] = PROBE_BATCH_SIZES,
    ) -> Optional["asyncio.Future[dict]"]:
        """Register a probe and wake the worker.

        Returns the future the HTTP layer awaits, or None when another probe
        is already running (the caller answers 409; retrying later is allowed).
        """
        if self._probe is not None:
            return None
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[dict]" = loop.create_future()
        self._probe = ProbeState(
            id=uuid.uuid4().hex[:12],
            sample_tensors=list(sample_tensors),
            batch_sizes=tuple(batch_sizes),
            future=future,
        )
        self._wake.set()
        return future

    def cancel_probe(self) -> None:
        """Ask the worker to abandon the current probe (e.g. client disconnect).

        The currently computing batch still finishes; the probe slot is
        released at the next worker await point and a new probe may start.
        """
        if self._probe is not None:
            self._probe.cancelled = True
            self._wake.set()

    # ------------------------------------------------------------------ #
    # single worker task: serializes orders and probes on one channel
    # ------------------------------------------------------------------ #
    async def _run(self) -> None:
        while True:
            # Sleep only when there is genuinely no work. The clear/check
            # sequence is atomic w.r.t. other coroutines (no await between
            # them), so a wakeup can never be lost.
            if self._probe is None and not self._pending:
                self._wake.clear()
                if self._probe is None and not self._pending:
                    await self._wake.wait()

            if self._probe is not None:
                await self._process_probe()
                continue

            batch = await self._form_batch()
            if batch:
                await self._run_order_batch(batch)

    async def _form_batch(self) -> list[Order]:
        """Take queued orders into one batch, waiting at most batch_wait_s."""
        loop = asyncio.get_running_loop()
        batch: list[Order] = []
        window_end = loop.time() + self.batch_wait_s
        while len(batch) < self.batch_size:
            self._purge_expired_pending()
            while self._pending and len(batch) < self.batch_size:
                batch.append(self._pending.popleft())
            if len(batch) >= self.batch_size:
                break
            remaining = window_end - loop.time()
            if remaining <= 0:
                break
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), remaining)
            except asyncio.TimeoutError:
                break

        # Final gate: cancelled/expired orders never enter the model.
        now = time.monotonic()
        return [o for o in batch if o.state == QUEUED and o.deadline_at > now]

    async def _run_order_batch(self, batch: list[Order]) -> None:
        loop = asyncio.get_running_loop()
        batch_id = uuid.uuid4().hex[:8]
        for o in batch:
            o.state = PROCESSING
            o.batch_id = batch_id
        self._inflight = len(batch)
        self.batches_started += 1
        self.last_batch_size = len(batch)
        tensors = torch.stack([o.tensor for o in batch])

        try:
            scores = await loop.run_in_executor(self._executor, self._infer, tensors)
        except Exception:  # noqa: BLE001 - defensive: model failure must not kill the worker
            log.exception("inference failed for batch %s", batch_id)
            for o in batch:
                if o.state == PROCESSING:
                    o.state = FAILED
            self._inflight = 0
            return

        now = time.monotonic()
        for o, sc in zip(batch, scores):
            if o.state != PROCESSING:
                # Cancelled or expired mid-flight: result is discarded and
                # the late result must NOT revive the order.
                continue
            if o.deadline_at <= now:
                o.state = EXPIRED
                continue
            o.state = COMPLETED
            predicted = max(range(len(sc)), key=lambda i: sc[i])
            o.result = {"predicted_class": predicted, "scores": sc}
        self._inflight = 0

    async def _process_probe(self) -> None:
        probe = self._probe
        assert probe is not None
        loop = asyncio.get_running_loop()
        trials: list[dict] = []

        try:
            # Orders first: drain everything accepted before the probe even
            # starts (including the idle batch-formation window), so the warmup
            # and timed trials do not queue behind pending work.
            while self._pending:
                batch = await self._form_batch()
                if batch:
                    await self._run_order_batch(batch)
            if probe.cancelled or probe.future.cancelled():
                self._discard_probe(probe)
                return

            # One untimed warmup on the shared channel: the first forward of a
            # process pays one-off allocator/JIT costs unrelated to batch size,
            # which would otherwise make the size-1 trial look artificially slow.
            warmup = torch.stack(probe.sample_tensors[:1])
            try:
                await loop.run_in_executor(self._executor, self._infer, warmup)
            except Exception as exc:  # noqa: BLE001 - release the probe so it can be retried
                log.exception("probe %s warmup failed", probe.id)
                self._finish_probe(probe, exc=exc)
                return

            for n in probe.batch_sizes:
                # Orders first: every accepted pending order is processed
                # before each probe trial. A batch dispatched here (or a
                # probe trial below) runs on the single executor and cannot
                # be preempted, so the wait is bounded by its compute time.
                while self._pending:
                    batch = await self._form_batch()
                    if batch:
                        await self._run_order_batch(batch)
                    if probe.cancelled or probe.future.cancelled():
                        self._discard_probe(probe)
                        return

                if probe.cancelled or probe.future.cancelled():
                    self._discard_probe(probe)
                    return

                if n > len(probe.sample_tensors):
                    self._finish_probe(
                        probe,
                        exc=RuntimeError(
                            f"not enough sample images ({len(probe.sample_tensors)}) "
                            f"for a probe batch of {n}"
                        ),
                    )
                    return

                batch_tensor = torch.stack(probe.sample_tensors[:n])
                t0 = loop.time()
                try:
                    probs, forward_ms = await loop.run_in_executor(
                        self._executor, self._infer_timed, batch_tensor
                    )
                except Exception as exc:  # noqa: BLE001 - release the probe so it can be retried
                    log.exception("probe %s inference failed for batch size %s", probe.id, n)
                    self._finish_probe(probe, exc=exc)
                    return
                wall_ms = (loop.time() - t0) * 1000.0

                # Sanity-check the shared channel still returns valid output;
                # probe results themselves are not stored anywhere.
                if len(probs) != n or any(len(row) == 0 for row in probs):
                    self._finish_probe(
                        probe,
                        exc=RuntimeError(f"probe trial of {n} image(s) returned malformed output"),
                    )
                    return

                trials.append(
                    {
                        "batch_size": n,
                        "images": n,
                        # Wall-clock time the single compute channel was busy
                        # (includes INFERENCE_DELAY_MS when that demo knob is set).
                        "compute_ms": round(wall_ms, 3),
                        # Pure model forward time measured inside the worker thread.
                        "forward_ms": round(forward_ms, 3),
                        "throughput_images_per_s": round(n / (wall_ms / 1000.0), 3),
                    }
                )

            self._finish_probe(
                probe,
                result={
                    "probe_id": probe.id,
                    "sample_count": len(probe.sample_tensors),
                    "batch_sizes": list(probe.batch_sizes),
                    "trials": trials,
                },
            )
        finally:
            # Defensive: never leave the worker parked if control escapes.
            self._wake.set()

    def _discard_probe(self, probe: ProbeState) -> None:
        """Release a cancelled probe without fulfilling its future."""
        if self._probe is probe:
            self._probe = None
        self._wake.set()
        if not probe.future.done():
            probe.future.cancel()

    def _finish_probe(
        self,
        probe: ProbeState,
        result: Optional[dict] = None,
        exc: Optional[BaseException] = None,
    ) -> None:
        """Release the probe slot and resolve its future (unless cancelled)."""
        if self._probe is probe:
            self._probe = None
        self._wake.set()
        if probe.future.done() or probe.cancelled:
            return
        if exc is not None:
            probe.future.set_exception(exc)
        else:
            probe.future.set_result(result)

    # ------------------------------------------------------------------ #
    # runs on the single executor thread — never touches shared state
    # ------------------------------------------------------------------ #
    def _infer(self, batch_tensor: torch.Tensor) -> list[list[float]]:
        if self.inference_delay_s > 0:
            time.sleep(self.inference_delay_s)  # demo: simulate a slow model
        with torch.inference_mode():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
        return [[float(v) for v in row] for row in probs]

    def _infer_timed(
        self, batch_tensor: torch.Tensor
    ) -> tuple[list[list[float]], float]:
        """Same compute path as orders, additionally reporting pure forward ms."""
        if self.inference_delay_s > 0:
            time.sleep(self.inference_delay_s)  # demo: simulate a slow model
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
        forward_ms = (time.perf_counter() - t0) * 1000.0
        return [[float(v) for v in row] for row in probs], forward_ms
