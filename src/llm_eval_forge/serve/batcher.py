"""Micro-batching: coalesce concurrent requests into one forward pass.

The first request in a batch waits at most `max_wait_ms` for others to join (or until
`max_batch` requests are queued); the batch then runs in a worker thread so the event loop
keeps accepting requests. On a GPU this trades a few milliseconds of queueing for much
higher throughput under load.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class BatchStats:
    batches: int = 0
    items: int = 0
    recent_sizes: deque = field(default_factory=lambda: deque(maxlen=1000))

    @property
    def mean_batch_size(self) -> float:
        return self.items / self.batches if self.batches else 0.0


class MicroBatcher:
    def __init__(self, runtime, max_batch: int = 32, max_wait_ms: float = 3.0):
        self.runtime = runtime
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.queue: asyncio.Queue = asyncio.Queue()
        self.stats = BatchStats()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, prompt: str, response: str) -> float:
        self.start()
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put(((prompt, response), fut))
        return await fut

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = time.perf_counter() + self.max_wait
            while len(batch) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            pairs = [item for item, _ in batch]
            try:
                probs = await loop.run_in_executor(None, self.runtime.predict, pairs)
            except Exception as exc:  # fail every request in the batch, keep the worker alive
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            self.stats.batches += 1
            self.stats.items += len(batch)
            self.stats.recent_sizes.append(len(batch))
            for (_, fut), p in zip(batch, probs):
                if not fut.done():
                    fut.set_result(p)
