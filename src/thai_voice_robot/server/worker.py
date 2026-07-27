"""Explicit serialized worker that keeps the FastAPI event loop responsive."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any, Callable


@dataclass(slots=True)
class _WorkItem:
    function: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    done: Event = field(default_factory=Event)
    result: Any = None
    error: BaseException | None = None


class SerializedWorker:
    """Run blocking LLM/TTS calls one at a time on one owned thread.

    Polling the completion event with ``asyncio.sleep`` avoids FastAPI's hidden
    AnyIO thread-pool bridge and leaves health/dashboard routes responsive.
    """

    def __init__(self, name: str = "nong-yoong-conversation") -> None:
        self._queue: Queue[_WorkItem | None] = Queue()
        self._thread: Thread | None = None
        self._start_lock = Lock()
        self._name = name

    def _ensure_started(self) -> None:
        with self._start_lock:
            if self._thread is None:
                self._thread = Thread(target=self._loop, name=self._name, daemon=True)
                self._thread.start()

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                item.result = item.function(*item.args, **item.kwargs)
            except BaseException as exc:
                item.error = exc
            finally:
                item.done.set()

    async def run(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._ensure_started()
        item = _WorkItem(function=function, args=args, kwargs=kwargs)
        self._queue.put(item)
        while not item.done.is_set():
            await asyncio.sleep(0.01)
        if item.error is not None:
            raise item.error
        return item.result

    def close(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
