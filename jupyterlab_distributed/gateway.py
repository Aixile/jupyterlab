"""WebSocket gateway for coordinating distributed SPMD workers.

The gateway runs on rank-0 and handles:
- Worker registration and authentication
- Broadcasting cell execution to all workers
- Collecting execution results and outputs from workers
"""

import asyncio
import json
import logging
import threading
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
        # Use threading.Event (not asyncio.Event) because the gateway runs
        # in a background thread while collect_results is called from the
        # kernel's event loop thread. asyncio.Event is NOT thread-safe.
        self._completion_events: dict[str, threading.Event] = {}
        self._output_sizes: dict[str, dict[int, int]] = {}
        self._target_ranks: dict[str, set[int]] = {}
        self._server: Any = None

    async def start(self) -> None:
        """Start the WebSocket server."""
        # websockets < 14: serve() returns a coroutine (needs await)
        # websockets >= 14: serve() returns a Server directly
        server = websockets.serve(
            self._handle_connection,
            "0.0.0.0",
            self.port,
        )
        if hasattr(server, "__await__"):
            self._server = await server
        else:
            self._server = server
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

        A snapshot of the currently connected worker ranks is stored so
        that the completion check compares against the workers that were
        present at broadcast time, not at result-arrival time.
        """
        msg_id = str(uuid.uuid4())
        self._pending_results[msg_id] = {}
        self._pending_outputs[msg_id] = {}
        self._output_sizes[msg_id] = {}
        self._completion_events[msg_id] = threading.Event()

        # Snapshot: record which ranks should respond to this message
        self._target_ranks[msg_id] = set(self.workers.keys())

        message = json.dumps({
            "type": "execute",
            "code": code,
            "cell_id": cell_id,
            "msg_id": msg_id,
        })

        await self._broadcast(message)
        return msg_id

    async def send_to_rank(self, rank: int, code: str, cell_id: str = "") -> str:
        """Send an execute message to a specific worker only.

        Returns the generated msg_id for tracking results.
        Raises KeyError if the rank is not connected.
        """
        if rank not in self.workers:
            raise KeyError(f"Rank {rank} is not connected")

        msg_id = str(uuid.uuid4())
        self._pending_results[msg_id] = {}
        self._pending_outputs[msg_id] = {}
        self._output_sizes[msg_id] = {}
        self._completion_events[msg_id] = threading.Event()
        self._target_ranks[msg_id] = {rank}

        message = json.dumps({
            "type": "execute",
            "code": code,
            "cell_id": cell_id,
            "msg_id": msg_id,
        })

        worker = self.workers[rank]
        await worker.ws.send(message)
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

        Uses threading.Event (not asyncio.Event) because the gateway's
        WebSocket handler runs in a different thread. We await via
        asyncio.to_thread so the kernel's event loop stays responsive.
        """
        event = self._completion_events.get(msg_id)
        if event is None:
            return {}

        # threading.Event.wait() is blocking — run in executor to avoid
        # blocking the kernel's event loop
        completed = await asyncio.to_thread(event.wait, self.timeout)
        if not completed:
            logger.warning(
                "Timeout collecting results for %s. Got %d/%d",
                msg_id,
                len(self._pending_results.get(msg_id, {})),
                len(self._target_ranks.get(msg_id, set())),
            )
        return self._pending_results.get(msg_id, {})

    def get_outputs(self, msg_id: str) -> dict[int, list[dict[str, Any]]]:
        """Get buffered outputs for a given execution msg_id."""
        return self._pending_outputs.get(msg_id, {})

    def _fail_pending_for_rank(self, rank: int) -> None:
        """Mark any pending executions for a disconnected rank as failed.

        When a worker disconnects mid-execution, its result will never arrive.
        This method fills in an error result for the rank and checks whether
        all target ranks now have results, setting the completion event if so.
        """
        for msg_id, targets in list(self._target_ranks.items()):
            if rank in targets:
                if msg_id not in self._pending_results:
                    self._pending_results[msg_id] = {}
                if rank not in self._pending_results[msg_id]:
                    self._pending_results[msg_id][rank] = {
                        "status": "error",
                        "ename": "WorkerDisconnected",
                        "evalue": f"Worker rank {rank} disconnected",
                        "outputs": [],
                        "execution_time": 0,
                    }
                # Check if all target ranks now have results
                if targets.issubset(
                    set(self._pending_results[msg_id].keys())
                ):
                    event = self._completion_events.get(msg_id)
                    if event:
                        event.set()

    async def _broadcast(self, message: str) -> None:
        """Send a message to all connected workers in parallel."""
        async def _send(rank: int, worker: WorkerInfo) -> int | None:
            try:
                await worker.ws.send(message)
                return None
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Worker rank %d disconnected during broadcast", rank)
                return rank

        results = await asyncio.gather(
            *[_send(r, w) for r, w in self.workers.items()]
        )
        for rank in results:
            if rank is not None:
                del self.workers[rank]
                self._fail_pending_for_rank(rank)

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
            rank = None
            for r, w in list(self.workers.items()):
                if w.ws is ws:
                    rank = r
                    del self.workers[r]
                    logger.info("Worker rank %d disconnected", r)
                    break

            if rank is not None:
                self._fail_pending_for_rank(rank)

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

            # Merge buffered outputs into the result so that
            # collect_results() returns complete per-rank data
            # including all stream/display/error messages.
            outputs = self._pending_outputs.get(msg_id, {}).get(rank, [])
            msg["outputs"] = outputs

            self._pending_results[msg_id][rank] = msg

            # Check if all target ranks have reported (use snapshot if available)
            target_ranks = self._target_ranks.get(msg_id)
            if target_ranks is not None:
                if target_ranks.issubset(self._pending_results[msg_id].keys()):
                    event = self._completion_events.get(msg_id)
                    if event is not None:
                        event.set()
            else:
                # Fallback for messages without a snapshot (e.g. old API usage)
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
