"""Worker daemon with embedded IPython shell for distributed SPMD execution.

The worker is a lightweight Python process that runs on each non-rank-0 node.
It connects to rank-0's gateway via WebSocket, receives code to execute, and
streams back outputs.
"""

import asyncio
import json
import logging
import os
import signal
import socket
import sys
import time
import traceback
from io import StringIO
from pathlib import Path

import websockets

logger = logging.getLogger(__name__)


class Worker:
    """A worker daemon that connects to the gateway and executes code.

    Each worker maintains an embedded IPython InteractiveShell for code
    execution. It connects to rank-0's WebSocket gateway, registers itself,
    and then enters a message loop that dispatches execute, interrupt, reset,
    and shutdown commands.
    """

    def __init__(
        self,
        rank: int,
        server_url: str,
        auth_token: str,
        log_dir: str | None = None,
    ) -> None:
        self.rank = rank
        self.server_url = server_url
        self.auth_token = auth_token
        self.log_dir = log_dir

        self._shutdown = False
        self._shell = None
        self._log_file = None
        self._ws = None

    def shutdown(self) -> None:
        """Set the shutdown flag and close the WebSocket to unblock the loop."""
        self._shutdown = True
        if self._ws is not None:
            # Close the websocket to break out of the message loop.
            # Schedule this as a coroutine-safe close via the running loop.
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._ws.close())
            except RuntimeError:
                pass

    async def run(self) -> None:
        """Main entry point for the worker.

        Initializes the IPython shell, sets up logging, connects to the
        gateway, registers, and enters the message loop.
        """
        self._init_shell()
        self._init_logging()

        try:
            async with websockets.connect(self.server_url) as ws:
                self._ws = ws
                # Send registration message
                register_msg = json.dumps({
                    "type": "register",
                    "rank": self.rank,
                    "hostname": socket.gethostname(),
                    "gpu_id": int(os.environ.get("LOCAL_RANK", -1)),
                    "pid": os.getpid(),
                    "token": self.auth_token,
                })
                await ws.send(register_msg)

                # Read registration response
                raw = await ws.recv()
                response = json.loads(raw)

                if response.get("type") == "error":
                    logger.error(
                        "Registration failed for rank %d: %s",
                        self.rank,
                        response.get("message", "unknown error"),
                    )
                    return

                logger.info(
                    "Worker rank %d registered (world_size=%d)",
                    self.rank,
                    response.get("world_size", -1),
                )

                await self._message_loop(ws)
        except websockets.exceptions.ConnectionClosed:
            if not self._shutdown:
                logger.warning("Worker rank %d: connection closed", self.rank)
        except Exception:
            logger.exception("Worker rank %d: unexpected error", self.rank)
        finally:
            self._ws = None
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None

    def _init_shell(self) -> None:
        """Create an embedded IPython InteractiveShell instance."""
        from IPython.core.interactiveshell import InteractiveShell

        self._shell = InteractiveShell.instance()

    def _init_logging(self) -> None:
        """Set up optional file-based logging."""
        if self.log_dir is not None:
            log_path = Path(self.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            log_file = log_path / f"worker-rank-{self.rank}.log"
            self._log_file = open(log_file, "a")  # noqa: SIM115

            handler = logging.StreamHandler(self._log_file)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s %(name)s: %(message)s"
                )
            )
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

    async def _message_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Read messages from the gateway and dispatch by type."""
        async for raw_msg in ws:
            if self._shutdown:
                break

            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                logger.warning(
                    "Worker rank %d: invalid JSON message", self.rank
                )
                continue

            msg_type = msg.get("type")

            if msg_type == "execute":
                await self._handle_execute(ws, msg)
            elif msg_type == "interrupt":
                self._handle_interrupt()
            elif msg_type == "reset":
                self._handle_reset()
            elif msg_type == "shutdown":
                self._shutdown = True
                break
            else:
                logger.warning(
                    "Worker rank %d: unknown message type '%s'",
                    self.rank,
                    msg_type,
                )

    def _run_cell_sync(self, code: str) -> dict:
        """Run code in the IPython shell synchronously (called from a thread).

        Returns a dict with stdout, stderr, status, error info, and timing.
        This method is designed to be called via ``asyncio.to_thread()``
        so that the event loop remains free to process interrupt messages.
        """
        start_time = time.monotonic()

        stdout_capture = StringIO()
        stderr_capture = StringIO()

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        try:
            result = self._shell.run_cell(code)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        execution_time = time.monotonic() - start_time

        status = "ok"
        ename = ""
        evalue = ""
        tb_lines: list[str] = []

        if result.error_before_exec is not None:
            status = "error"
            exc = result.error_before_exec
            ename = type(exc).__name__
            evalue = str(exc)
            tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        elif result.error_in_exec is not None:
            status = "error"
            exc = result.error_in_exec
            ename = type(exc).__name__
            evalue = str(exc)
            tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)

        return {
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "status": status,
            "ename": ename,
            "evalue": evalue,
            "traceback": tb_lines,
            "execution_time": execution_time,
        }

    async def _handle_execute(
        self, ws: websockets.WebSocketClientProtocol, msg: dict
    ) -> None:
        """Execute code and send back results and outputs.

        Runs the code in a separate thread via ``asyncio.to_thread()`` so
        that the event loop remains responsive to incoming messages (e.g.
        interrupt). After execution completes, sends back stream messages
        and execute_complete on the event loop thread.
        """
        msg_id = msg.get("msg_id", "")
        code = msg.get("code", "")

        # Store the execution thread ID so _handle_interrupt can target it
        import threading

        self._exec_thread_id: int | None = None

        def _run_in_thread() -> dict:
            self._exec_thread_id = threading.get_ident()
            try:
                return self._run_cell_sync(code)
            finally:
                self._exec_thread_id = None

        cell_result = await asyncio.to_thread(_run_in_thread)

        status = cell_result["status"]
        ename = cell_result["ename"]
        evalue = cell_result["evalue"]
        tb_lines = cell_result["traceback"]

        # Send error output message if there was an error
        if status == "error":
            error_msg = json.dumps({
                "type": "error",
                "msg_id": msg_id,
                "ename": ename,
                "evalue": evalue,
                "traceback": tb_lines,
            })
            await ws.send(error_msg)

        # Send captured stdout as stream message
        stdout_text = cell_result["stdout"]
        if stdout_text:
            stream_msg = json.dumps({
                "type": "stream",
                "msg_id": msg_id,
                "name": "stdout",
                "text": stdout_text,
            })
            await ws.send(stream_msg)

        # Send captured stderr as stream message
        stderr_text = cell_result["stderr"]
        if stderr_text:
            stream_msg = json.dumps({
                "type": "stream",
                "msg_id": msg_id,
                "name": "stderr",
                "text": stderr_text,
            })
            await ws.send(stream_msg)

        # Send execute_complete
        complete_msg = {
            "type": "execute_complete",
            "msg_id": msg_id,
            "status": status,
            "execution_time": cell_result["execution_time"],
        }
        if status == "error":
            complete_msg["ename"] = ename
            complete_msg["evalue"] = evalue
            complete_msg["traceback"] = tb_lines

        await ws.send(json.dumps(complete_msg))

    def _handle_interrupt(self) -> None:
        """Send SIGINT to the execution thread to interrupt running code.

        If code is executing in a background thread (via ``asyncio.to_thread``),
        we use ``signal.pthread_kill`` (on platforms that support it) to target
        that thread specifically. Otherwise falls back to raising SIGINT on the
        main process.
        """
        import threading

        tid = getattr(self, "_exec_thread_id", None)
        if tid is not None and hasattr(signal, "pthread_kill"):
            # Send SIGINT to the specific execution thread
            try:
                signal.pthread_kill(tid, signal.SIGINT)
                return
            except (OSError, ValueError):
                pass

        # Fallback: raise SIGINT on the process (main thread)
        signal.raise_signal(signal.SIGINT)

    def _handle_reset(self) -> None:
        """Clear the IPython namespace."""
        if self._shell is not None:
            self._shell.reset(new_session=True)
