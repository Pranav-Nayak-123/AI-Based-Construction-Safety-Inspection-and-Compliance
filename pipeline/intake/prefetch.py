"""Decode ahead on a background thread so decoding overlaps GPU inference and encoding.

OpenCV decoding and PyTorch inference both release the GIL, so a single producer thread
gives real parallelism without multiprocessing.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")
_DONE = object()


class Prefetch(Generic[T]):
    """Iterate `source` on a worker thread, keeping up to `depth` items ready."""

    def __init__(self, source: Iterable[T], depth: int = 8) -> None:
        self._source = source
        self._queue: queue.Queue[object] = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="prefetch", daemon=True)
        self._error: BaseException | None = None
        self.wait_s = 0.0  # time the consumer spent blocked waiting for items

    def _run(self) -> None:
        try:
            for item in self._source:
                while not self._stop.is_set():
                    try:
                        self._queue.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if self._stop.is_set():
                    return
        except BaseException as error:  # re-raised on the consumer thread
            self._error = error
        finally:
            self._queue.put(_DONE)

    def __iter__(self) -> Iterator[T]:
        self._thread.start()
        try:
            while True:
                tick = time.perf_counter()
                item = self._queue.get()
                self.wait_s += time.perf_counter() - tick
                if item is _DONE:
                    if self._error is not None:
                        raise self._error
                    return
                yield item  # type: ignore[misc]
        finally:
            self._stop.set()
            while self._thread.is_alive():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    self._thread.join(timeout=0.1)
