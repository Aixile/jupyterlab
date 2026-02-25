"""WebSocket gateway for coordinating distributed SPMD workers.

The gateway runs on rank-0 and handles:
- Worker registration and authentication
- Broadcasting cell execution to all workers
- Collecting execution results and outputs from workers
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)

# Maximum buffered output size per msg_id per worker (1 MB)
MAX_OUTPUT_SIZE = 1_048_576


@dataclass
class WorkerInfo:
    """Metadata for a connected worker."""

    rank: int
    hostname: str
    gpu_id: int
    pid: int
    ws: WebSocketServerProtocol
    state: str = "registered"


class Gateway:
    """WebSocket server for distributed kernel coordination.

    Runs on rank-0. Workers connect, register, and then receive
    broadcast messages for cell execution, interrupt, etc.
    """

    def __init__(
        self,
        port: int,
        auth_token: str,
        expected_workers: int,
        timeout: float = 30.0,
    ) -> None:
        self.port = port
        self.auth_token = auth_token
        self.expected_workers = expected_workers
        self.timeout = timeout

        self.workers: dict[int, WorkerInfo] = {}
        self._pending_results: dict[str, dict[int, dict[str, Any]]] = {}
        self._pending_outputs: dict[str, dict[int, list[dict[str, Any]]]] = {}
        self._completion_events: dict[str, asyncio.Event] = {}
        self._output_sizes: dict[str, dict[int, int]] = {}
        self._server: Any = None

    async def start(self) -> None:
        """Start the WebSocket server."""
        self._server = await websockets.serve(
            self._handle_connection,
            "0.0.0.0",
            self.port,
        )
        # If port was 0, update to the actual assigned port
        for sock in self._server.sockets:
            addr = sock.getsockname()
            self.port = addr[1]
            break
        logger.info("Gateway started on port %d", self.port)

    async def stop(self) -> None:
        """Stop the WebSocket server and close all connections."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self.workers.clear()
        logger.info("Gateway stopped")

    def all_workers_registered(self) -> bool:
        """Check whether all expected workers have registered."""
        return len(self.workers) >= self.expected_workers

    async def broadcast_execute(self, code: str, cell_id: str) -> str:
        """Broadcast an execute message to all connected workers.

        Returns the generated msg_id for tracking results.
        """
        msg_id = str(uuid.uuid4())
        self._pending_results[msg_id] = {}
        self._pending_outputs[msg_id] = {}
        self._output_sizes[msg_id] = {}
        self._completion_events[msg_id] = asyncio.Event()

        message = json.dumps({
            "type": "execute",
            "code": code,
            "cell_id": cell_id,
            "msg_id": msg_id,
        })

        await self._broadcast(message)
        return msg_id

    async def broadcast_interrupt(self) -> None:
        """Send an interrupt message to all connected workers."""
        message = json.dumps({"type": "interrupt"})
        await self._broadcast(message)

    async def broadcast_reset(self) -> None:
        """Send a reset message to all connected workers."""
        message = json.dumps({"type": "reset"})
        await self._broadcast(message)

    async def broadcast_shutdown(self) -> None:
        """Send a shutdown message to all connected workers."""
        message = json.dumps({"type": "shutdown"})
        await self._broadcast(message)

    async def collect_results(self, msg_id: str) -> dict[int, dict[str, Any]]:
        """Wait for all workers to send execute_complete for msg_id.

        Returns a dict mapping rank -> result dict.
        Raises asyncio.TimeoutError if not all workers respond in time.
        """
        event = self._completion_events.get(msg_id)
        if event is None:
            return {}

        await asyncio.wait_for(event.wait(), timeout=self.timeout)
        return self._pending_results.get(msg_id, {})

    def get_outputs(self, msg_id: str) -> dict[int, list[dict[str, Any]]]:
        """Get buffered outputs for a given execution msg_id."""
        return self._pending_outputs.get(msg_id, {})

    async def _broadcast(self, message: str) -> None:
        """Send a message to all connected workers."""
        disconnected: list[int] = []
        for rank, worker in self.workers.items():
            try:
                await worker.ws.send(message)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Worker rank %d disconnected during broadcast", rank)
                disconnected.append(rank)

        for rank in disconnected:
            del self.workers[rank]

    async def _handle_connection(self, ws: WebSocketServerProtocol) -> None:
        """Handle a new WebSocket connection from a worker."""
        try:
            # First message must be a registration message
            raw = await ws.recv()
            msg = json.loads(raw)

            if msg.get("type") != "register":
                await ws.send(json.dumps({
                    "type": "error",
                    "message": "First message must be a register message",
                }))
                await ws.close()
                return

            # Validate auth token
            if msg.get("token") != self.auth_token:
                await ws.send(json.dumps({
                    "type": "error",
                    "message": "Authentication failed: invalid token",
                }))
                await ws.close()
                return

            rank = msg["rank"]
            worker = WorkerInfo(
                rank=rank,
                hostname=msg.get("hostname", "unknown"),
                gpu_id=msg.get("gpu_id", -1),
                pid=msg.get("pid", -1),
                ws=ws,
            )
            self.workers[rank] = worker

            # Acknowledge registration
            await ws.send(json.dumps({
                "type": "registered",
                "rank": rank,
                "world_size": self.expected_workers,
            }))

            logger.info(
                "Worker rank %d registered (host=%s, gpu=%d, pid=%d)",
                rank,
                worker.hostname,
                worker.gpu_id,
                worker.pid,
            )

            # Enter message loop
            async for raw_msg in ws:
                try:
                    worker_msg = json.loads(raw_msg)
                    await self._handle_worker_message(rank, worker_msg)
                except json.JSONDecodeError:
                    logger.warning(
                        "Invalid JSON from worker rank %d: %s", rank, raw_msg
                    )
                except Exception:
                    logger.exception(
                        "Error handling message from worker rank %d", rank
                    )

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            logger.exception("Unexpected error in connection handler")
        finally:
            # Clean up disconnected worker
            for r, w in list(self.workers.items()):
                if w.ws is ws:
                    del self.workers[r]
                    logger.info("Worker rank %d disconnected", r)
                    break

    async def _handle_worker_message(
        self, rank: int, msg: dict[str, Any]
    ) -> None:
        """Dispatch an incoming message from a worker by type."""
        msg_type = msg.get("type")
        msg_id = msg.get("msg_id")

        if msg_type == "execute_complete":
            if msg_id is None:
                return
            if msg_id not in self._pending_results:
                self._pending_results[msg_id] = {}
            self._pending_results[msg_id][rank] = msg

            # Check if all connected workers have reported
            if len(self._pending_results[msg_id]) >= len(self.workers):
                event = self._completion_events.get(msg_id)
                if event is not None:
                    event.set()

        elif msg_type in ("stream", "display_data", "execute_result", "error"):
            if msg_id is None:
                return
            if msg_id not in self._pending_outputs:
                self._pending_outputs[msg_id] = {}
            if rank not in self._pending_outputs[msg_id]:
                self._pending_outputs[msg_id][rank] = []
            if msg_id not in self._output_sizes:
                self._output_sizes[msg_id] = {}
            if rank not in self._output_sizes[msg_id]:
                self._output_sizes[msg_id][rank] = 0

            # Enforce per-worker output size limit
            msg_size = len(json.dumps(msg))
            if self._output_sizes[msg_id][rank] + msg_size <= MAX_OUTPUT_SIZE:
                self._pending_outputs[msg_id][rank].append(msg)
                self._output_sizes[msg_id][rank] += msg_size
            else:
                logger.warning(
                    "Output size limit exceeded for rank %d, msg_id %s",
                    rank,
                    msg_id,
                )

        elif msg_type == "heartbeat":
            worker = self.workers.get(rank)
            if worker is not None:
                worker.state = "alive"

        else:
            logger.warning(
                "Unknown message type '%s' from worker rank %d", msg_type, rank
            )
