"""Sandboxed Python execution with persistent per-attempt state."""

from __future__ import annotations

import asyncio
import io
import multiprocessing
import os
import queue
import re
import signal
import subprocess
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any

_PRELUDE = "import math, numpy, sympy, mpmath, itertools, collections\nmpmath.mp.dps = 64\n"
_MAX_RESULT_CHARS = 2000
_MAX_RESULT_LINES = 60
_CBC_KEEP_SUBSTRINGS = (
    "Welcome to the CBC MILP Solver",
    "Result -",
    "No feasible solution found",
    "Objective value:",
    "Lower bound:",
    "Enumerated nodes:",
    "Total iterations:",
    "Time (CPU seconds):",
    "Time (Wallclock seconds):",
    "Stopped on time limit",
    "Optimal - objective value",
    "Integer solution of",
)


class _ExecutionTimedOut(Exception):
    pass


def _ensure_last_print(code: str) -> str:
    lines = code.strip().split("\n")
    if not lines:
        return code

    last = lines[-1].strip()
    if any(token in last for token in ("print", "import")) or not last or last.startswith("#"):
        return code

    lines[-1] = f"print({last})"
    return "\n".join(lines)


def _is_shell_cell(code: str) -> bool:
    stripped_lines = [line.strip() for line in code.splitlines() if line.strip()]
    return bool(stripped_lines) and all(line.startswith("!") for line in stripped_lines)


def _run_shell_cell(code: str, timeout_sec: float) -> tuple[bool, str, str]:
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    deadline = time.monotonic() + timeout_sec

    for raw_line in code.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith("!"):
            return False, "", "Mixed shell/Python cells are not supported"

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False, "".join(stdout_parts), f"Execution timed out after {timeout_sec} seconds"

        try:
            completed = subprocess.run(
                line[1:].strip(),
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                timeout=remaining,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return False, "".join(stdout_parts), f"Execution timed out after {timeout_sec} seconds"

        if completed.stdout:
            stdout_parts.append(completed.stdout)
        if completed.stderr:
            stderr_parts.append(completed.stderr)
        if completed.returncode != 0:
            stderr = "".join(stderr_parts) or f"Shell command failed with exit code {completed.returncode}"
            return False, "".join(stdout_parts), stderr

    return True, "".join(stdout_parts), "".join(stderr_parts)


def _summarize_cbc_output(text: str) -> str | None:
    if "CBC MILP Solver" not in text and not re.search(r"\bCbc\d{4}[A-Z]?\b", text):
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    kept: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if any(marker in line for marker in _CBC_KEEP_SUBSTRINGS):
            if line not in seen:
                kept.append(line)
                seen.add(line)

    if not kept:
        return "[solver log summarized]\nCBC solver ran, but only emitted verbose progress output."

    summary = "[solver log summarized]\n" + "\n".join(kept)
    if "Stopped on time limit" in summary or "No feasible solution found" in summary:
        summary += "\nTry a smaller or different formulation instead of repeating the same large MILP."
    return summary


def _truncate_output(text: str) -> str:
    lines = text.splitlines()
    if len(text) <= _MAX_RESULT_CHARS and len(lines) <= _MAX_RESULT_LINES:
        return text

    head_lines = lines[:20]
    tail_lines = lines[-20:] if len(lines) > 20 else []
    pieces = ["[output truncated]"]
    if head_lines:
        pieces.extend(head_lines)
    if tail_lines:
        pieces.append("...")
        pieces.extend(tail_lines)
    truncated = "\n".join(pieces)
    if len(truncated) > _MAX_RESULT_CHARS:
        return truncated[: _MAX_RESULT_CHARS - 3] + "..."
    return truncated


def _summarize_output(text: str) -> str:
    if not text:
        return text
    cbc_summary = _summarize_cbc_output(text)
    if cbc_summary is not None:
        return cbc_summary
    return _truncate_output(text)


def _worker_loop(request_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue) -> None:
    def _handle_timeout(signum: int, frame: Any) -> None:
        raise _ExecutionTimedOut

    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec(compile(_PRELUDE, "<prelude>", "exec"), namespace, namespace)
    response_queue.put({"request_id": 0, "ok": True, "stdout": "", "stderr": "", "ready": True})
    while True:
        item = request_queue.get()
        if item is None:
            break

        request_id, code, timeout_sec = item
        buf_out, buf_err = io.StringIO(), io.StringIO()
        old_handler = signal.getsignal(signal.SIGALRM)
        try:
            if _is_shell_cell(code):
                ok, stdout, stderr = _run_shell_cell(code, timeout_sec)
                response_queue.put(
                    {
                        "request_id": request_id,
                        "ok": ok,
                        "stdout": stdout,
                        "stderr": stderr,
                    }
                )
            else:
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.setitimer(signal.ITIMER_REAL, timeout_sec)
                with redirect_stdout(buf_out), redirect_stderr(buf_err):
                    exec(compile(_ensure_last_print(code), "<model>", "exec"), namespace, namespace)
                response_queue.put(
                    {
                        "request_id": request_id,
                        "ok": True,
                        "stdout": buf_out.getvalue(),
                        "stderr": buf_err.getvalue(),
                    }
                )
        except _ExecutionTimedOut:
            response_queue.put(
                {
                    "request_id": request_id,
                    "ok": False,
                    "stdout": buf_out.getvalue(),
                    "stderr": f"Execution timed out after {timeout_sec} seconds",
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
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)


def _format_result(result: dict[str, Any]) -> str:
    out = _summarize_output((result.get("stdout") or "").strip())
    err = _summarize_output((result.get("stderr") or "").strip())
    if result.get("ok"):
        text = (
            out
            if out
            else "[WARN] No visible output. Your next python message must run a concrete computation and use "
            "print(...) to show the result. Do not send imports-only, comments-only, or a prose restatement "
            "of the problem."
        )
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
    _started: bool = False

    def start(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._ctx = multiprocessing.get_context("spawn")
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._proc = self._ctx.Process(target=_worker_loop, args=(self._request_queue, self._response_queue))
        self._proc.start()
        self._started = False
        assert self._response_queue is not None
        try:
            ready = self._response_queue.get(True, 30.0)
        except queue.Empty:
            self._stop_process()
            raise RuntimeError("Python tool worker failed to start within 30 seconds")
        if not ready.get("ready"):
            self._stop_process()
            raise RuntimeError("Python tool worker failed to report ready state")
        self._started = True

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
        self._started = False

    async def execute(self, code: str, timeout_sec: float) -> str:
        try:
            self.start()
        except Exception as exc:
            self._stop_process()
            return f"[ERROR] Python tool failed to start: {exc}"
        assert self._request_queue is not None
        assert self._response_queue is not None

        self._request_id += 1
        request_id = self._request_id
        self._request_queue.put((request_id, code, timeout_sec))

        try:
            result = await asyncio.to_thread(self._response_queue.get, True, timeout_sec + 1.0)
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
