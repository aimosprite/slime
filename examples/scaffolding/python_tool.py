"""Sandboxed Python execution with persistent per-attempt state."""

from __future__ import annotations

import asyncio
import io
import multiprocessing
import queue
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any


def _worker_loop(request_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue) -> None:
    namespace: dict[str, Any] = {"__name__": "__main__"}
    while True:
        item = request_queue.get()
        if item is None:
            break

        request_id, code = item
        buf_out, buf_err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                exec(compile(code, "<model>", "exec"), namespace, namespace)
            response_queue.put(
                {
                    "request_id": request_id,
                    "ok": True,
                    "stdout": buf_out.getvalue(),
                    "stderr": buf_err.getvalue(),
                }
            )
        except Exception:
            response_queue.put(
                {
                    "request_id": request_id,
                    "ok": False,
                    "stdout": buf_out.getvalue(),
                    "stderr": buf_err.getvalue() + traceback.format_exc(),
                }
            )


def _format_result(result: dict[str, Any]) -> str:
    out = (result.get("stdout") or "").strip()
    err = (result.get("stderr") or "").strip()
    if result.get("ok"):
        text = out if out else "(no stdout)"
        if err:
            text += f"\n[stderr]\n{err}"
        return text
    return f"[ERROR]\n{err or out}"


@dataclass
class PersistentPythonSession:
    """Single-worker persistent Python session for one attempt."""

    _ctx: multiprocessing.context.BaseContext | None = None
    _request_queue: multiprocessing.Queue | None = None
    _response_queue: multiprocessing.Queue | None = None
    _proc: multiprocessing.Process | None = None
    _request_id: int = 0

    def start(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._ctx = multiprocessing.get_context("spawn")
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._proc = self._ctx.Process(target=_worker_loop, args=(self._request_queue, self._response_queue))
        self._proc.start()

    def _stop_process(self) -> None:
        if self._request_queue is not None:
            try:
                self._request_queue.put_nowait(None)
            except Exception:
                pass
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.kill()
        self._proc = None
        self._request_queue = None
        self._response_queue = None

    async def execute(self, code: str, timeout_sec: float) -> str:
        self.start()
        assert self._request_queue is not None
        assert self._response_queue is not None

        self._request_id += 1
        request_id = self._request_id
        self._request_queue.put((request_id, code))

        try:
            result = await asyncio.to_thread(self._response_queue.get, True, timeout_sec)
        except queue.Empty:
            self._stop_process()
            return f"[ERROR] Execution timed out after {timeout_sec} seconds"
        except Exception:
            self._stop_process()
            return "[ERROR] Python tool process failed"

        if result.get("request_id") != request_id:
            return "[ERROR] Python tool response mismatch"
        return _format_result(result)

    def close(self) -> None:
        self._stop_process()
