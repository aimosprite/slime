"""Lightweight stateful Python executor for tool-call-enabled OPD rollouts.

Each executor spawns a persistent Python subprocess (Jupyter-notebook–like REPL)
where state (variables, imports) persists across calls within a single rollout
sample.  Communication happens via stdin/stdout with a sentinel delimiter.

No restrictive safety checks — allows numpy, sympy, scipy, math, etc.
Code comes from the model, not untrusted users.
"""

import asyncio
import logging
import os
import subprocess
import sys
import textwrap

logger = logging.getLogger(__name__)

SENTINEL = "__SLIME_EXEC_DONE__"

# Global concurrency limiter (set once at import time; override via set_concurrency)
_semaphore: asyncio.Semaphore | None = None


def get_semaphore(concurrency: int = 32) -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(concurrency)
    return _semaphore


def set_concurrency(concurrency: int) -> None:
    global _semaphore
    _semaphore = asyncio.Semaphore(concurrency)


# ---------------------------------------------------------------------------
# Subprocess REPL script
# ---------------------------------------------------------------------------

_REPL_SCRIPT = textwrap.dedent(r'''
    import sys, traceback, io, signal

    # Optionally cap virtual memory (best-effort; some OS don't support RLIMIT_AS)
    try:
        import resource
        _EIGHT_GB = 8 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (_EIGHT_GB, _EIGHT_GB))
    except Exception:
        pass

    SENTINEL = "__SLIME_EXEC_DONE__"
    _globals = {"__builtins__": __builtins__}

    def _run():
        while True:
            # Read lines until we see the sentinel
            lines = []
            for line in sys.stdin:
                line = line.rstrip("\n")
                if line == SENTINEL:
                    break
                lines.append(line)
            else:
                # stdin closed
                break

            code = "\n".join(lines)
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = stdout_buf, stderr_buf
            try:
                exec(compile(code, "<tool>", "exec"), _globals)
            except Exception:
                traceback.print_exc(file=stderr_buf)
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

            out = stdout_buf.getvalue()
            err = stderr_buf.getvalue()
            result_parts = []
            if out:
                result_parts.append(out.rstrip("\n"))
            if err:
                result_parts.append(err.rstrip("\n"))
            result = "\n".join(result_parts) if result_parts else ""

            # Send result then sentinel on its own line
            sys.stdout.write(result + "\n" + SENTINEL + "\n")
            sys.stdout.flush()

    _run()
''').lstrip()


class StatefulPythonExecutor:
    """A persistent subprocess-based Python interpreter.

    State (variables, imports) persists across ``execute()`` calls within a
    single executor instance.  Call ``close()`` after the rollout is done.
    """

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _start(self) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _REPL_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

    def reset(self) -> None:
        """Kill the current subprocess and start a fresh one."""
        self.close()
        self._start()

    def close(self) -> None:
        """Kill the subprocess."""
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, code: str) -> str:
        """Execute *code* in the persistent REPL and return stdout+stderr.

        Acquires the global concurrency semaphore before running.
        """
        sem = get_semaphore()
        async with sem:
            return await self._execute_inner(code)

    async def _execute_inner(self, code: str) -> str:
        if self._proc is None or self._proc.poll() is not None:
            self._start()

        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._send_and_recv, code),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Tool executor timed out after %ss, resetting subprocess", self.timeout)
            self.reset()
            result = f"Error: Code execution timed out after {self.timeout} seconds"
        except Exception as e:
            logger.warning("Tool executor error: %s, resetting subprocess", e)
            self.reset()
            result = f"Error: {e}"
        return result

    def _send_and_recv(self, code: str) -> str:
        """Blocking: send code + sentinel, read lines until sentinel."""
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        # Send code followed by the sentinel line
        self._proc.stdin.write(code + "\n" + SENTINEL + "\n")
        self._proc.stdin.flush()

        lines: list[str] = []
        for raw_line in self._proc.stdout:
            line = raw_line.rstrip("\n")
            if line == SENTINEL:
                break
            lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# OpenAI-format tool specification
# ---------------------------------------------------------------------------

def get_tool_spec() -> dict:
    """Return the python tool spec in OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": "python",
            "description": (
                "Execute Python code in a stateful session. "
                "Use print() for outputs. "
                "Libraries available: math, numpy, sympy, itertools, collections."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    },
                },
                "required": ["code"],
            },
        },
    }
