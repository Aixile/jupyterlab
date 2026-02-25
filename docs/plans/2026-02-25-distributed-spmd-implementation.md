# Distributed SPMD Cell Execution — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add SPMD distributed cell execution to JupyterLab where rank-0 is the real kernel and all ranks execute cells simultaneously via torchrun.

**Architecture:** A persistent launcher process (rank-0) hosts both the IPython kernel and a WebSocket gateway. Worker daemons on ranks 1-N embed IPython shells and connect to rank-0's gateway. A kernel provisioner on the Jupyter server side bridges the remote kernel to JupyterLab. The frontend renders per-rank outputs via a custom MIME renderer and shows cluster health in a sidebar panel.

**Tech Stack:** Python (ipykernel, websockets, asyncio, torch.distributed), TypeScript (Lumino widgets, React 18), Jupyter messaging protocol (ZMQ).

**Design doc:** `docs/plans/2026-02-25-distributed-spmd-design.md`

---

## Phase 1: Core Python Infrastructure

### Task 1: Session Config Module

**Files:**
- Create: `jupyterlab_distributed/__init__.py`
- Create: `jupyterlab_distributed/config.py`
- Create: `tests/test_distributed/__init__.py`
- Create: `tests/test_distributed/test_config.py`

**Step 1: Write the failing tests**

```python
# tests/test_distributed/test_config.py
import json
import os
import tempfile
import pytest
from jupyterlab_distributed.config import SessionConfig


class TestSessionConfig:
    def test_create_writes_json(self, tmp_path):
        config = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-kernel-001",
            world_size=4,
            gateway_port=9876,
        )
        assert config.path.exists()
        data = json.loads(config.path.read_text())
        assert data["kernel_id"] == "test-kernel-001"
        assert data["world_size"] == 4
        assert data["status"] == "waiting"
        assert data["host"] is None
        assert len(data["auth_token"]) >= 32

    def test_create_sets_restrictive_permissions(self, tmp_path):
        config = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-kernel-002",
            world_size=2,
            gateway_port=9877,
        )
        stat = os.stat(config.path)
        assert oct(stat.st_mode & 0o777) == "0o600"

    def test_atomic_write_no_partial_file(self, tmp_path):
        config = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-kernel-003",
            world_size=2,
            gateway_port=9878,
        )
        # File should be valid JSON (no partial writes)
        data = json.loads(config.path.read_text())
        assert data["kernel_id"] == "test-kernel-003"

    def test_load_reads_existing(self, tmp_path):
        created = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-kernel-004",
            world_size=8,
            gateway_port=9879,
        )
        loaded = SessionConfig.load(created.path)
        assert loaded.kernel_id == "test-kernel-004"
        assert loaded.world_size == 8

    def test_update_status(self, tmp_path):
        config = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-kernel-005",
            world_size=2,
            gateway_port=9880,
        )
        config.update(status="running", host="compute-01")
        reloaded = SessionConfig.load(config.path)
        assert reloaded.status == "running"
        assert reloaded.host == "compute-01"

    def test_zmq_ports_allocated(self, tmp_path):
        config = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-kernel-006",
            world_size=2,
            gateway_port=9881,
        )
        ports = config.zmq_ports
        assert "shell" in ports
        assert "iopub" in ports
        assert "stdin" in ports
        assert "control" in ports
        assert "hb" in ports
        # All ports should be distinct
        assert len(set(ports.values())) == 5

    def test_ttl_cleanup(self, tmp_path):
        config = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-kernel-007",
            world_size=2,
            gateway_port=9882,
            ttl_seconds=0,  # Immediately stale
        )
        stale = SessionConfig.cleanup_stale(str(tmp_path))
        assert len(stale) == 1
        assert not config.path.exists()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_distributed/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jupyterlab_distributed'`

**Step 3: Write minimal implementation**

```python
# jupyterlab_distributed/__init__.py
"""JupyterLab Distributed SPMD execution support."""
__version__ = "0.1.0"
```

```python
# jupyterlab_distributed/config.py
"""Session config for distributed kernel coordination via shared filesystem."""

import json
import os
import secrets
import socket
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _find_free_ports(count: int) -> list[int]:
    """Find N free TCP ports."""
    sockets = []
    ports = []
    for _ in range(count):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        ports.append(s.getsockname()[1])
        sockets.append(s)
    for s in sockets:
        s.close()
    return ports


@dataclass
class SessionConfig:
    kernel_id: str
    world_size: int
    gateway_port: int
    zmq_ports: dict[str, int] = field(default_factory=dict)
    auth_token: str = ""
    status: str = "waiting"
    host: str | None = None
    created_at: float = 0.0
    ttl_seconds: int = 3600
    path: Path = field(default_factory=lambda: Path(""))

    @classmethod
    def create(
        cls,
        base_dir: str,
        kernel_id: str,
        world_size: int,
        gateway_port: int,
        ttl_seconds: int = 3600,
    ) -> "SessionConfig":
        session_dir = Path(base_dir) / ".jupyter" / "distributed"
        session_dir.mkdir(parents=True, exist_ok=True)

        ports = _find_free_ports(5)
        zmq_ports = {
            "shell": ports[0],
            "iopub": ports[1],
            "stdin": ports[2],
            "control": ports[3],
            "hb": ports[4],
        }

        config = cls(
            kernel_id=kernel_id,
            world_size=world_size,
            gateway_port=gateway_port,
            zmq_ports=zmq_ports,
            auth_token=secrets.token_hex(32),
            status="waiting",
            host=None,
            created_at=time.time(),
            ttl_seconds=ttl_seconds,
            path=session_dir / f"{kernel_id}.json",
        )
        config._write()
        return config

    @classmethod
    def load(cls, path: Path | str) -> "SessionConfig":
        path = Path(path)
        data = json.loads(path.read_text())
        data["path"] = path
        return cls(**data)

    def update(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._write()

    def _write(self) -> None:
        data = asdict(self)
        data["path"] = str(data["path"])
        # Atomic write: write to temp, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self.path)
        except BaseException:
            os.unlink(tmp_path)
            raise

    @staticmethod
    def cleanup_stale(base_dir: str) -> list[Path]:
        session_dir = Path(base_dir) / ".jupyter" / "distributed"
        cleaned = []
        if not session_dir.exists():
            return cleaned
        now = time.time()
        for f in session_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                created = data.get("created_at", 0)
                ttl = data.get("ttl_seconds", 3600)
                if now - created > ttl:
                    f.unlink()
                    cleaned.append(f)
            except (json.JSONDecodeError, OSError):
                f.unlink()
                cleaned.append(f)
        return cleaned
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_distributed/test_config.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add jupyterlab_distributed/__init__.py jupyterlab_distributed/config.py \
    tests/test_distributed/__init__.py tests/test_distributed/test_config.py
git commit -m "feat(distributed): add session config module with atomic write and TTL cleanup"
```

---

### Task 2: WebSocket Gateway

**Files:**
- Create: `jupyterlab_distributed/gateway.py`
- Create: `tests/test_distributed/test_gateway.py`

**Step 1: Write the failing tests**

```python
# tests/test_distributed/test_gateway.py
import asyncio
import json
import pytest
import websockets
from jupyterlab_distributed.gateway import Gateway


@pytest.fixture
async def gateway():
    gw = Gateway(port=0, auth_token="test-secret", expected_workers=2)
    await gw.start()
    yield gw
    await gw.stop()


@pytest.fixture
def gateway_url(gateway):
    return f"ws://localhost:{gateway.port}/ws"


class TestGateway:
    @pytest.mark.asyncio
    async def test_rejects_bad_auth(self, gateway_url):
        async with websockets.connect(gateway_url) as ws:
            await ws.send(json.dumps({
                "type": "register", "rank": 0,
                "hostname": "node-1", "gpu_id": 0, "pid": 1234,
                "token": "wrong-token",
            }))
            resp = json.loads(await ws.recv())
            assert resp["type"] == "error"
            assert "auth" in resp["message"].lower()

    @pytest.mark.asyncio
    async def test_accepts_valid_registration(self, gateway, gateway_url):
        async with websockets.connect(gateway_url) as ws:
            await ws.send(json.dumps({
                "type": "register", "rank": 1,
                "hostname": "node-1", "gpu_id": 0, "pid": 1234,
                "token": "test-secret",
            }))
            resp = json.loads(await ws.recv())
            assert resp["type"] == "registered"
            assert resp["rank"] == 1
        assert 1 in gateway.workers

    @pytest.mark.asyncio
    async def test_broadcast_execute(self, gateway, gateway_url):
        # Register a worker
        async with websockets.connect(gateway_url) as ws:
            await ws.send(json.dumps({
                "type": "register", "rank": 1,
                "hostname": "node-1", "gpu_id": 0, "pid": 1234,
                "token": "test-secret",
            }))
            await ws.recv()  # registered ack

            # Broadcast execute
            msg_id = gateway.broadcast_execute("print('hello')", "cell-1")

            # Worker should receive the execute message
            exec_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
            assert exec_msg["type"] == "execute"
            assert exec_msg["code"] == "print('hello')"
            assert exec_msg["msg_id"] == msg_id

    @pytest.mark.asyncio
    async def test_collect_worker_output(self, gateway, gateway_url):
        async with websockets.connect(gateway_url) as ws:
            await ws.send(json.dumps({
                "type": "register", "rank": 1,
                "hostname": "node-1", "gpu_id": 0, "pid": 1234,
                "token": "test-secret",
            }))
            await ws.recv()

            msg_id = gateway.broadcast_execute("x = 1", "cell-2")

            # Worker sends back completion
            await ws.send(json.dumps({
                "type": "execute_complete",
                "msg_id": msg_id,
                "status": "ok",
                "execution_time": 0.5,
            }))

            results = await asyncio.wait_for(
                gateway.collect_results(msg_id), timeout=5
            )
            assert 1 in results
            assert results[1]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_broadcast_interrupt(self, gateway, gateway_url):
        async with websockets.connect(gateway_url) as ws:
            await ws.send(json.dumps({
                "type": "register", "rank": 1,
                "hostname": "node-1", "gpu_id": 0, "pid": 1234,
                "token": "test-secret",
            }))
            await ws.recv()

            gateway.broadcast_interrupt()

            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
            assert msg["type"] == "interrupt"

    @pytest.mark.asyncio
    async def test_all_workers_registered(self, gateway, gateway_url):
        assert not gateway.all_workers_registered()
        async with websockets.connect(gateway_url) as ws1:
            await ws1.send(json.dumps({
                "type": "register", "rank": 0,
                "hostname": "n1", "gpu_id": 0, "pid": 1,
                "token": "test-secret",
            }))
            await ws1.recv()
            assert not gateway.all_workers_registered()

            async with websockets.connect(gateway_url) as ws2:
                await ws2.send(json.dumps({
                    "type": "register", "rank": 1,
                    "hostname": "n2", "gpu_id": 0, "pid": 2,
                    "token": "test-secret",
                }))
                await ws2.recv()
                assert gateway.all_workers_registered()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_distributed/test_gateway.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jupyterlab_distributed.gateway'`

**Step 3: Write minimal implementation**

```python
# jupyterlab_distributed/gateway.py
"""WebSocket gateway for distributed worker management."""

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

import websockets
from websockets.server import serve

logger = logging.getLogger(__name__)

MAX_OUTPUT_SIZE = 1024 * 1024  # 1MB per rank per execution


@dataclass
class WorkerInfo:
    rank: int
    hostname: str
    gpu_id: int
    pid: int
    ws: object = None  # websockets connection
    state: str = "idle"


class Gateway:
    def __init__(
        self,
        port: int,
        auth_token: str,
        expected_workers: int,
        timeout: float = 30.0,
    ):
        self.port = port
        self.auth_token = auth_token
        self.expected_workers = expected_workers
        self.timeout = timeout
        self.workers: dict[int, WorkerInfo] = {}
        self._server = None
        self._pending_results: dict[str, dict[int, dict]] = defaultdict(dict)
        self._pending_outputs: dict[str, dict[int, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._completion_events: dict[str, asyncio.Event] = {}
        self._completed_ranks: dict[str, set] = defaultdict(set)

    async def start(self) -> None:
        self._server = await serve(
            self._handle_connection, "0.0.0.0", self.port
        )
        # Update port if it was 0 (auto-assigned)
        for sock in self._server.sockets:
            addr = sock.getsockname()
            self.port = addr[1]
            break

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    def all_workers_registered(self) -> bool:
        return len(self.workers) >= self.expected_workers

    def broadcast_execute(self, code: str, cell_id: str) -> str:
        msg_id = str(uuid.uuid4())
        self._completion_events[msg_id] = asyncio.Event()
        self._completed_ranks[msg_id] = set()
        msg = json.dumps({
            "type": "execute",
            "msg_id": msg_id,
            "code": code,
            "cell_id": cell_id,
        })
        for worker in self.workers.values():
            if worker.ws is not None:
                asyncio.ensure_future(worker.ws.send(msg))
        return msg_id

    def broadcast_interrupt(self) -> None:
        msg = json.dumps({"type": "interrupt"})
        for worker in self.workers.values():
            if worker.ws is not None:
                asyncio.ensure_future(worker.ws.send(msg))

    def broadcast_reset(self) -> None:
        msg = json.dumps({"type": "reset"})
        for worker in self.workers.values():
            if worker.ws is not None:
                asyncio.ensure_future(worker.ws.send(msg))

    def broadcast_shutdown(self) -> None:
        msg = json.dumps({"type": "shutdown"})
        for worker in self.workers.values():
            if worker.ws is not None:
                asyncio.ensure_future(worker.ws.send(msg))

    async def collect_results(self, msg_id: str) -> dict[int, dict]:
        event = self._completion_events.get(msg_id)
        if event:
            try:
                await asyncio.wait_for(event.wait(), timeout=self.timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout waiting for workers. Completed: %s/%s",
                    len(self._completed_ranks[msg_id]),
                    len(self.workers),
                )
        results = dict(self._pending_results.pop(msg_id, {}))
        self._pending_outputs.pop(msg_id, None)
        self._completion_events.pop(msg_id, None)
        self._completed_ranks.pop(msg_id, None)
        return results

    def get_outputs(self, msg_id: str) -> dict[int, list]:
        return dict(self._pending_outputs.get(msg_id, {}))

    async def _handle_connection(self, ws) -> None:
        rank = None
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            msg = json.loads(raw)
            if msg.get("type") != "register":
                await ws.send(json.dumps({
                    "type": "error",
                    "message": "First message must be register",
                }))
                return
            if msg.get("token") != self.auth_token:
                await ws.send(json.dumps({
                    "type": "error",
                    "message": "Authentication failed",
                }))
                return

            rank = msg["rank"]
            self.workers[rank] = WorkerInfo(
                rank=rank,
                hostname=msg.get("hostname", "unknown"),
                gpu_id=msg.get("gpu_id", 0),
                pid=msg.get("pid", 0),
                ws=ws,
            )
            await ws.send(json.dumps({
                "type": "registered",
                "rank": rank,
                "world_size": self.expected_workers,
            }))
            logger.info(
                "Worker rank %d registered from %s",
                rank, msg.get("hostname"),
            )

            async for raw in ws:
                msg = json.loads(raw)
                await self._handle_worker_message(rank, msg)

        except websockets.ConnectionClosed:
            logger.info("Worker rank %s disconnected", rank)
        except Exception:
            logger.exception("Error handling worker rank %s", rank)
        finally:
            if rank is not None and rank in self.workers:
                self.workers[rank].ws = None

    async def _handle_worker_message(self, rank: int, msg: dict) -> None:
        msg_type = msg.get("type")
        msg_id = msg.get("msg_id")

        if msg_type == "execute_complete":
            self._pending_results[msg_id][rank] = {
                "status": msg.get("status", "ok"),
                "execution_time": msg.get("execution_time", 0),
                "outputs": list(self._pending_outputs.get(msg_id, {}).get(rank, [])),
            }
            self._completed_ranks[msg_id].add(rank)
            if len(self._completed_ranks[msg_id]) >= len(self.workers):
                event = self._completion_events.get(msg_id)
                if event:
                    event.set()

        elif msg_type in ("stream", "display_data", "execute_result", "error"):
            outputs = self._pending_outputs[msg_id][rank]
            # Enforce output size limit
            total = sum(len(json.dumps(o)) for o in outputs)
            if total < MAX_OUTPUT_SIZE:
                outputs.append(msg)

        elif msg_type == "heartbeat":
            if rank in self.workers:
                self.workers[rank].state = msg.get("state", "idle")
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_distributed/test_gateway.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add jupyterlab_distributed/gateway.py tests/test_distributed/test_gateway.py
git commit -m "feat(distributed): add WebSocket gateway with auth, broadcast, and output collection"
```

---

### Task 3: Worker Daemon with Embedded IPython Shell

**Files:**
- Create: `jupyterlab_distributed/worker.py`
- Create: `tests/test_distributed/test_worker.py`

**Step 1: Write the failing tests**

```python
# tests/test_distributed/test_worker.py
import asyncio
import json
import pytest
import websockets
from jupyterlab_distributed.gateway import Gateway
from jupyterlab_distributed.worker import Worker


@pytest.fixture
async def gateway():
    gw = Gateway(port=0, auth_token="test-token", expected_workers=1)
    await gw.start()
    yield gw
    await gw.stop()


class TestWorker:
    @pytest.mark.asyncio
    async def test_worker_connects_and_registers(self, gateway):
        url = f"ws://localhost:{gateway.port}/ws"
        worker = Worker(rank=1, server_url=url, auth_token="test-token")
        task = asyncio.create_task(worker.run())
        try:
            # Wait for registration
            for _ in range(50):
                if 1 in gateway.workers:
                    break
                await asyncio.sleep(0.1)
            assert 1 in gateway.workers
            assert gateway.workers[1].hostname != ""
        finally:
            worker.shutdown()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_worker_executes_code(self, gateway):
        url = f"ws://localhost:{gateway.port}/ws"
        worker = Worker(rank=1, server_url=url, auth_token="test-token")
        task = asyncio.create_task(worker.run())
        try:
            for _ in range(50):
                if 1 in gateway.workers:
                    break
                await asyncio.sleep(0.1)

            msg_id = gateway.broadcast_execute("x = 42", "cell-1")
            results = await asyncio.wait_for(
                gateway.collect_results(msg_id), timeout=10
            )
            assert 1 in results
            assert results[1]["status"] == "ok"
        finally:
            worker.shutdown()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_worker_captures_stdout(self, gateway):
        url = f"ws://localhost:{gateway.port}/ws"
        worker = Worker(rank=1, server_url=url, auth_token="test-token")
        task = asyncio.create_task(worker.run())
        try:
            for _ in range(50):
                if 1 in gateway.workers:
                    break
                await asyncio.sleep(0.1)

            msg_id = gateway.broadcast_execute("print('hello world')", "cell-2")
            results = await asyncio.wait_for(
                gateway.collect_results(msg_id), timeout=10
            )
            assert 1 in results
            outputs = results[1]["outputs"]
            stdout_msgs = [o for o in outputs if o.get("type") == "stream" and o.get("name") == "stdout"]
            assert any("hello world" in m.get("text", "") for m in stdout_msgs)
        finally:
            worker.shutdown()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_worker_captures_error(self, gateway):
        url = f"ws://localhost:{gateway.port}/ws"
        worker = Worker(rank=1, server_url=url, auth_token="test-token")
        task = asyncio.create_task(worker.run())
        try:
            for _ in range(50):
                if 1 in gateway.workers:
                    break
                await asyncio.sleep(0.1)

            msg_id = gateway.broadcast_execute("1/0", "cell-3")
            results = await asyncio.wait_for(
                gateway.collect_results(msg_id), timeout=10
            )
            assert 1 in results
            assert results[1]["status"] == "error"
        finally:
            worker.shutdown()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_worker_reset_clears_namespace(self, gateway):
        url = f"ws://localhost:{gateway.port}/ws"
        worker = Worker(rank=1, server_url=url, auth_token="test-token")
        task = asyncio.create_task(worker.run())
        try:
            for _ in range(50):
                if 1 in gateway.workers:
                    break
                await asyncio.sleep(0.1)

            # Define a variable
            msg_id = gateway.broadcast_execute("my_var = 999", "cell-4")
            await asyncio.wait_for(
                gateway.collect_results(msg_id), timeout=10
            )

            # Reset
            gateway.broadcast_reset()
            await asyncio.sleep(0.5)

            # Variable should be gone
            msg_id = gateway.broadcast_execute("print(my_var)", "cell-5")
            results = await asyncio.wait_for(
                gateway.collect_results(msg_id), timeout=10
            )
            assert results[1]["status"] == "error"
        finally:
            worker.shutdown()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_distributed/test_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jupyterlab_distributed.worker'`

**Step 3: Write minimal implementation**

```python
# jupyterlab_distributed/worker.py
"""Worker daemon with embedded IPython InteractiveShell."""

import asyncio
import io
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
import traceback

import websockets

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        rank: int,
        server_url: str,
        auth_token: str,
        log_dir: str | None = None,
    ):
        self.rank = rank
        self.server_url = server_url
        self.auth_token = auth_token
        self.log_dir = log_dir
        self._shutdown = False
        self._ws = None
        self._shell = None
        self._log_file = None

    def shutdown(self) -> None:
        self._shutdown = True

    async def run(self) -> None:
        self._init_shell()
        self._init_logging()

        try:
            async with websockets.connect(self.server_url) as ws:
                self._ws = ws
                # Register
                await ws.send(json.dumps({
                    "type": "register",
                    "rank": self.rank,
                    "hostname": socket.gethostname(),
                    "gpu_id": int(os.environ.get("LOCAL_RANK", 0)),
                    "pid": os.getpid(),
                    "token": self.auth_token,
                }))
                resp = json.loads(await ws.recv())
                if resp.get("type") == "error":
                    logger.error("Registration failed: %s", resp.get("message"))
                    return
                logger.info("Registered as rank %d", self.rank)

                await self._message_loop(ws)
        except websockets.ConnectionClosed:
            logger.info("Connection to gateway closed")
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Worker error")
        finally:
            if self._log_file:
                self._log_file.close()

    def _init_shell(self) -> None:
        from IPython.core.interactiveshell import InteractiveShell
        self._shell = InteractiveShell.instance()
        # Disable input() on workers
        self._shell.ask_exit = lambda: None

    def _init_logging(self) -> None:
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            log_path = os.path.join(self.log_dir, f"rank-{self.rank}.log")
            self._log_file = open(log_path, "a", buffering=1)

    async def _message_loop(self, ws) -> None:
        async for raw in ws:
            if self._shutdown:
                break
            msg = json.loads(raw)
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

    async def _handle_execute(self, ws, msg: dict) -> None:
        msg_id = msg["msg_id"]
        code = msg["code"]
        start_time = time.monotonic()
        status = "ok"

        # Capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        try:
            sys.stdout = captured_stdout
            sys.stderr = captured_stderr

            result = self._shell.run_cell(code)

            if result.error_in_exec or result.error_before_exec:
                status = "error"
                tb = "".join(traceback.format_exception(
                    type(result.error_in_exec or result.error_before_exec),
                    result.error_in_exec or result.error_before_exec,
                    (result.error_in_exec or result.error_before_exec).__traceback__,
                ))
                await ws.send(json.dumps({
                    "type": "error",
                    "msg_id": msg_id,
                    "ename": type(result.error_in_exec or result.error_before_exec).__name__,
                    "evalue": str(result.error_in_exec or result.error_before_exec),
                    "traceback": tb.split("\n"),
                }))
        except Exception as e:
            status = "error"
            await ws.send(json.dumps({
                "type": "error",
                "msg_id": msg_id,
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": traceback.format_exc().split("\n"),
            }))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # Send captured stdout
        stdout_text = captured_stdout.getvalue()
        if stdout_text:
            await ws.send(json.dumps({
                "type": "stream",
                "msg_id": msg_id,
                "name": "stdout",
                "text": stdout_text,
            }))
            if self._log_file:
                self._log_file.write(stdout_text)

        # Send captured stderr
        stderr_text = captured_stderr.getvalue()
        if stderr_text:
            await ws.send(json.dumps({
                "type": "stream",
                "msg_id": msg_id,
                "name": "stderr",
                "text": stderr_text,
            }))

        elapsed = time.monotonic() - start_time
        await ws.send(json.dumps({
            "type": "execute_complete",
            "msg_id": msg_id,
            "status": status,
            "execution_time": round(elapsed, 3),
        }))

    def _handle_interrupt(self) -> None:
        # Send SIGINT to the main thread
        signal.raise_signal(signal.SIGINT)

    def _handle_reset(self) -> None:
        if self._shell:
            self._shell.reset(new_session=True)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_distributed/test_worker.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add jupyterlab_distributed/worker.py tests/test_distributed/test_worker.py
git commit -m "feat(distributed): add worker daemon with embedded IPython shell"
```

---

### Task 4: Distributed Kernel (do_execute override)

**Files:**
- Create: `jupyterlab_distributed/kernel.py`
- Create: `tests/test_distributed/test_kernel.py`

**Step 1: Write the failing tests**

```python
# tests/test_distributed/test_kernel.py
import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from jupyterlab_distributed.kernel import DistributedKernel


class TestDistributedKernel:
    def test_kernel_has_distributed_flag(self):
        with patch.object(DistributedKernel, "__init__", lambda self, **kw: None):
            k = DistributedKernel.__new__(DistributedKernel)
            k.distributed_enabled = False
            k._gateway = None
            assert not k.distributed_enabled

    @pytest.mark.asyncio
    async def test_do_execute_without_distributed_calls_super(self):
        """When distributed is disabled, do_execute delegates to parent."""
        with patch.object(DistributedKernel, "__init__", lambda self, **kw: None):
            k = DistributedKernel.__new__(DistributedKernel)
            k.distributed_enabled = False
            k._gateway = None
            # Mock the parent do_execute
            k._parent_do_execute = AsyncMock(return_value={
                "status": "ok",
                "execution_count": 1,
                "payload": [],
                "user_expressions": {},
            })
            result = await k.do_execute("x = 1", False)
            k._parent_do_execute.assert_called_once()
            assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_do_execute_with_distributed_broadcasts(self):
        """When distributed is enabled, do_execute broadcasts and collects."""
        with patch.object(DistributedKernel, "__init__", lambda self, **kw: None):
            k = DistributedKernel.__new__(DistributedKernel)
            k.distributed_enabled = True
            k._gateway = MagicMock()
            k._gateway.all_workers_registered.return_value = True
            k._gateway.broadcast_execute.return_value = "msg-001"
            k._gateway.collect_results = AsyncMock(return_value={
                1: {"status": "ok", "outputs": [], "execution_time": 0.5},
            })
            k._gateway.get_outputs.return_value = {}
            k._parent_do_execute = AsyncMock(return_value={
                "status": "ok",
                "execution_count": 1,
                "payload": [],
                "user_expressions": {},
            })
            k.send_response = MagicMock()
            k.iopub_socket = MagicMock()
            k._publish_distributed_outputs = MagicMock()

            result = await k.do_execute("x = 1", False)
            k._gateway.broadcast_execute.assert_called_once()
            assert result["status"] == "ok"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_distributed/test_kernel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jupyterlab_distributed.kernel'`

**Step 3: Write minimal implementation**

```python
# jupyterlab_distributed/kernel.py
"""Distributed IPython kernel with do_execute override."""

import asyncio
import json
import logging

from ipykernel.ipkernel import IPythonKernel

from .gateway import Gateway

logger = logging.getLogger(__name__)

DISTRIBUTED_MIME = "application/vnd.jupyterlab.distributed+json"


class DistributedKernel(IPythonKernel):
    implementation = "distributed-python"
    implementation_version = "0.1.0"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distributed_enabled = False
        self._gateway: Gateway | None = None

    def set_gateway(self, gateway: Gateway) -> None:
        self._gateway = gateway

    async def do_execute(
        self,
        code,
        silent,
        store_history=True,
        user_expressions=None,
        allow_stdin=False,
    ):
        # Check for magic commands that affect distributed behavior
        stripped = code.strip()
        if stripped.startswith("%%rank0"):
            # Execute only on rank-0 (skip broadcast)
            code = "\n".join(stripped.split("\n")[1:])
            return await self._parent_do_execute(
                code, silent, store_history, user_expressions, allow_stdin
            )

        if (
            self.distributed_enabled
            and self._gateway is not None
            and self._gateway.all_workers_registered()
        ):
            return await self._distributed_execute(
                code, silent, store_history, user_expressions, allow_stdin
            )
        else:
            return await self._parent_do_execute(
                code, silent, store_history, user_expressions, allow_stdin
            )

    async def _parent_do_execute(
        self, code, silent, store_history=True,
        user_expressions=None, allow_stdin=False,
    ):
        return await super().do_execute(
            code, silent, store_history, user_expressions, allow_stdin
        )

    async def _distributed_execute(
        self, code, silent, store_history, user_expressions, allow_stdin
    ):
        # 1. Broadcast to workers
        msg_id = self._gateway.broadcast_execute(code, "")

        # 2. Execute locally on rank-0
        result = await self._parent_do_execute(
            code, silent, store_history, user_expressions, allow_stdin
        )

        # 3. Collect worker results
        try:
            worker_results = await self._gateway.collect_results(msg_id)
        except Exception:
            logger.exception("Error collecting worker results")
            worker_results = {}

        # 4. Publish per-rank outputs
        self._publish_distributed_outputs(worker_results)

        # 5. Check for failures (fail-fast mode)
        failed_ranks = [
            r for r, res in worker_results.items()
            if res.get("status") != "ok"
        ]
        if failed_ranks and result.get("status") == "ok":
            logger.warning("Ranks %s failed", failed_ranks)

        return result

    def _publish_distributed_outputs(self, worker_results: dict) -> None:
        if not worker_results:
            return
        ranks_data = {}
        for rank, result in worker_results.items():
            ranks_data[str(rank)] = {
                "outputs": result.get("outputs", []),
                "status": result.get("status", "unknown"),
                "execution_time": result.get("execution_time", 0),
            }

        content = {
            "data": {
                DISTRIBUTED_MIME: {
                    "type": "rank_outputs",
                    "ranks": ranks_data,
                },
                "text/plain": f"Distributed execution: {len(worker_results)} ranks",
            },
            "metadata": {},
            "transient": {},
        }
        self.send_response(self.iopub_socket, "display_data", content)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_distributed/test_kernel.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add jupyterlab_distributed/kernel.py tests/test_distributed/test_kernel.py
git commit -m "feat(distributed): add DistributedKernel with do_execute override"
```

---

### Task 5: Launcher Entry Point (torchrun compatible)

**Files:**
- Create: `jupyterlab_distributed/launcher.py`
- Create: `tests/test_distributed/test_launcher.py`

**Step 1: Write the failing test**

```python
# tests/test_distributed/test_launcher.py
import os
import pytest
from unittest.mock import patch, MagicMock
from jupyterlab_distributed.launcher import detect_rank, detect_world_size


class TestLauncherDetection:
    def test_detect_rank_from_env(self):
        with patch.dict(os.environ, {"RANK": "3"}):
            assert detect_rank() == 3

    def test_detect_rank_from_slurm(self):
        with patch.dict(os.environ, {"SLURM_PROCID": "5"}, clear=False):
            env = os.environ.copy()
            env.pop("RANK", None)
            with patch.dict(os.environ, env, clear=True):
                with patch.dict(os.environ, {"SLURM_PROCID": "5"}):
                    assert detect_rank() == 5

    def test_detect_world_size_from_env(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "32"}):
            assert detect_world_size() == 32

    def test_detect_world_size_from_slurm(self):
        env = os.environ.copy()
        env.pop("WORLD_SIZE", None)
        with patch.dict(os.environ, env, clear=True):
            with patch.dict(os.environ, {"SLURM_NTASKS": "16"}):
                assert detect_world_size() == 16
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_distributed/test_launcher.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# jupyterlab_distributed/launcher.py
"""Entry point for torchrun: launches rank-0 kernel or worker daemon."""

import argparse
import asyncio
import logging
import os
import sys

from .config import SessionConfig

logger = logging.getLogger(__name__)


def detect_rank() -> int:
    for var in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    return 0


def detect_world_size() -> int:
    for var in ("WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="JupyterLab distributed launcher"
    )
    parser.add_argument(
        "--session-config", required=True,
        help="Path to session config JSON on shared filesystem",
    )
    parser.add_argument(
        "--rank", type=int, default=None,
        help="Override rank (default: auto-detect from env)",
    )
    parser.add_argument(
        "--log-dir", default=None,
        help="Directory for per-rank log files",
    )
    args = parser.parse_args()

    rank = args.rank if args.rank is not None else detect_rank()
    config = SessionConfig.load(args.session_config)

    logging.basicConfig(
        level=logging.INFO,
        format=f"[rank {rank}] %(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if rank == 0:
        _run_rank0(config, args.log_dir)
    else:
        _run_worker(rank, config, args.log_dir)


def _run_rank0(config: SessionConfig, log_dir: str | None) -> None:
    from .gateway import Gateway
    from .kernel import DistributedKernel

    logger.info("Starting rank-0 launcher (kernel + gateway)")

    # Start gateway in background
    loop = asyncio.new_event_loop()
    gateway = Gateway(
        port=config.gateway_port,
        auth_token=config.auth_token,
        expected_workers=config.world_size - 1,  # rank-0 is the kernel
    )
    loop.run_until_complete(gateway.start())
    logger.info("Gateway started on port %d", gateway.port)

    # Update config
    import socket
    config.update(status="running", host=socket.gethostname())

    # Start IPython kernel
    from ipykernel.kernelapp import IPKernelApp
    app = IPKernelApp.instance(kernel_class=DistributedKernel)
    app.initialize([
        f"--shell={config.zmq_ports['shell']}",
        f"--iopub={config.zmq_ports['iopub']}",
        f"--stdin={config.zmq_ports['stdin']}",
        f"--control={config.zmq_ports['control']}",
        f"--hb={config.zmq_ports['hb']}",
    ])
    kernel = app.kernel
    kernel.set_gateway(gateway)
    kernel.distributed_enabled = True
    logger.info("Kernel started, entering event loop")
    app.start()


def _run_worker(rank: int, config: SessionConfig, log_dir: str | None) -> None:
    from .worker import Worker

    logger.info("Starting worker rank %d", rank)

    # Wait for rank-0 to report running
    import time
    for _ in range(120):
        config = SessionConfig.load(config.path)
        if config.status == "running" and config.host:
            break
        time.sleep(1)
    else:
        logger.error("Timeout waiting for rank-0 to start")
        sys.exit(1)

    server_url = f"ws://{config.host}:{config.gateway_port}/ws"
    worker = Worker(
        rank=rank,
        server_url=server_url,
        auth_token=config.auth_token,
        log_dir=log_dir or str(config.path.parent / "logs"),
    )
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_distributed/test_launcher.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add jupyterlab_distributed/launcher.py tests/test_distributed/test_launcher.py
git commit -m "feat(distributed): add torchrun-compatible launcher with rank detection"
```

---

### Task 6: Magic Commands

**Files:**
- Create: `jupyterlab_distributed/magics.py`
- Create: `tests/test_distributed/test_magics.py`

**Step 1: Write the failing tests**

```python
# tests/test_distributed/test_magics.py
import pytest
from unittest.mock import MagicMock
from IPython.core.interactiveshell import InteractiveShell
from jupyterlab_distributed.magics import DistributedMagics


@pytest.fixture
def shell():
    s = InteractiveShell.instance()
    yield s
    InteractiveShell.clear_instance()


@pytest.fixture
def magics(shell):
    m = DistributedMagics(shell=shell)
    m.kernel = MagicMock()
    m.kernel.distributed_enabled = False
    m.kernel._gateway = MagicMock()
    m.kernel._gateway.workers = {0: MagicMock(), 1: MagicMock()}
    m.kernel._gateway.expected_workers = 2
    shell.register_magics(m)
    return m


class TestDistributedMagics:
    def test_distributed_on(self, magics):
        magics.distributed("on")
        assert magics.kernel.distributed_enabled is True

    def test_distributed_off(self, magics):
        magics.kernel.distributed_enabled = True
        magics.distributed("off")
        assert magics.kernel.distributed_enabled is False

    def test_distributed_status(self, magics, capsys):
        magics.distributed("status")
        out = capsys.readouterr().out
        assert "2" in out  # should mention worker count
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_distributed/test_magics.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# jupyterlab_distributed/magics.py
"""IPython magic commands for distributed execution control."""

from IPython.core.magic import Magics, magics_class, line_magic, cell_magic


@magics_class
class DistributedMagics(Magics):
    """Magic commands for controlling distributed cell execution."""

    def __init__(self, shell=None, **kwargs):
        super().__init__(shell, **kwargs)
        self.kernel = None  # Set by the kernel after initialization

    @line_magic
    def distributed(self, line: str) -> None:
        """Control distributed execution.

        Usage:
            %distributed on        - Enable distributed execution
            %distributed off       - Disable distributed execution
            %distributed status    - Show worker status
            %distributed expect N  - Set expected world size
            %distributed timeout N - Set timeout in seconds
            %distributed failure-mode fail-fast|best-effort
            %distributed restart hard - Kill all processes
        """
        args = line.strip().split()
        if not args:
            print("Usage: %distributed <on|off|status|expect|timeout|failure-mode|restart>")
            return

        cmd = args[0].lower()

        if cmd == "on":
            if self.kernel:
                self.kernel.distributed_enabled = True
            print("Distributed execution enabled")

        elif cmd == "off":
            if self.kernel:
                self.kernel.distributed_enabled = False
            print("Distributed execution disabled")

        elif cmd == "status":
            if not self.kernel or not self.kernel._gateway:
                print("No distributed gateway active")
                return
            gw = self.kernel._gateway
            registered = len(gw.workers)
            expected = gw.expected_workers
            print(f"Workers: {registered}/{expected}")
            for rank, info in sorted(gw.workers.items()):
                print(f"  rank {rank}: {info.hostname} GPU{info.gpu_id} "
                      f"PID={info.pid} [{info.state}]")

        elif cmd == "expect" and len(args) > 1:
            if self.kernel and self.kernel._gateway:
                self.kernel._gateway.expected_workers = int(args[1])
            print(f"Expected workers set to {args[1]}")

        elif cmd == "timeout" and len(args) > 1:
            if self.kernel and self.kernel._gateway:
                self.kernel._gateway.timeout = float(args[1])
            print(f"Timeout set to {args[1]}s")

        elif cmd == "restart" and len(args) > 1 and args[1] == "hard":
            if self.kernel and self.kernel._gateway:
                self.kernel._gateway.broadcast_shutdown()
            print("Hard restart: all workers shutting down")

        else:
            print(f"Unknown command: {line}")

    @cell_magic
    def rank0(self, line: str, cell: str) -> None:
        """Execute cell only on rank-0 (skip broadcast to workers)."""
        self.shell.run_cell(cell)

    @cell_magic
    def rank(self, line: str, cell: str) -> None:
        """Execute cell only on specified rank.

        Usage: %%rank N
        """
        target_rank = int(line.strip())
        if target_rank == 0:
            self.shell.run_cell(cell)
        elif self.kernel and self.kernel._gateway:
            gw = self.kernel._gateway
            if target_rank in gw.workers:
                import asyncio
                import json
                worker = gw.workers[target_rank]
                msg = json.dumps({
                    "type": "execute",
                    "msg_id": f"rank-{target_rank}-direct",
                    "code": cell,
                    "cell_id": "",
                })
                if worker.ws:
                    asyncio.ensure_future(worker.ws.send(msg))
            else:
                print(f"Rank {target_rank} not connected")
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_distributed/test_magics.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add jupyterlab_distributed/magics.py tests/test_distributed/test_magics.py
git commit -m "feat(distributed): add %distributed and %%rank magic commands"
```

---

## Phase 2: Kernel Provisioner

### Task 7: Distributed Kernel Provisioner

**Files:**
- Create: `jupyterlab_distributed/provisioner.py`
- Create: `tests/test_distributed/test_provisioner.py`

**Step 1: Write the failing tests**

```python
# tests/test_distributed/test_provisioner.py
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from jupyterlab_distributed.provisioner import DistributedProvisioner


class TestDistributedProvisioner:
    @pytest.mark.asyncio
    async def test_pre_launch_creates_config(self, tmp_path):
        prov = DistributedProvisioner.__new__(DistributedProvisioner)
        prov.kernel_id = "test-123"
        prov.config_base_dir = str(tmp_path)
        prov.world_size = 4
        prov.gateway_port = 9876
        prov._session_config = None

        kwargs = await prov.pre_launch(cmd=["python", "-m", "ipykernel"])
        assert prov._session_config is not None
        assert prov._session_config.path.exists()
        assert prov._session_config.kernel_id == "test-123"

    @pytest.mark.asyncio
    async def test_poll_returns_none_when_waiting(self, tmp_path):
        prov = DistributedProvisioner.__new__(DistributedProvisioner)
        prov._session_config = None
        prov._process = None
        result = await prov.poll()
        # None means still running (convention from subprocess.Popen.poll)
        assert result is None

    def test_get_provisioner_info(self, tmp_path):
        prov = DistributedProvisioner.__new__(DistributedProvisioner)
        prov.kernel_id = "test-456"
        prov.config_base_dir = str(tmp_path)
        prov.world_size = 8
        prov.gateway_port = 9877
        prov._session_config = MagicMock()
        prov._session_config.path = tmp_path / "test-456.json"
        info = prov.get_provisioner_info()
        assert info["config_path"] == str(tmp_path / "test-456.json")
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_distributed/test_provisioner.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# jupyterlab_distributed/provisioner.py
"""Kernel provisioner for distributed SPMD kernels."""

import asyncio
import logging
import os
from pathlib import Path

from jupyter_client.provisioning import KernelProvisionerBase

from .config import SessionConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = os.path.expanduser("~/.jupyter/distributed")


class DistributedProvisioner(KernelProvisionerBase):
    """Provisions a distributed kernel by coordinating with a remote
    rank-0 process launched via torchrun."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config_base_dir = kwargs.get("config_base_dir", DEFAULT_CONFIG_DIR)
        self.world_size = kwargs.get("world_size", 1)
        self.gateway_port = kwargs.get("gateway_port", 9876)
        self._session_config: SessionConfig | None = None
        self._process = None

    async def pre_launch(self, **kwargs) -> dict:
        """Create session config and wait for remote kernel."""
        self._session_config = SessionConfig.create(
            base_dir=self.config_base_dir,
            kernel_id=self.kernel_id,
            world_size=self.world_size,
            gateway_port=self.gateway_port,
        )
        logger.info(
            "Session config written to %s. "
            "Launch torchrun with --session-config %s",
            self._session_config.path,
            self._session_config.path,
        )
        return kwargs

    async def launch_process(self, cmd, **kwargs):
        """Wait for the external torchrun to start the kernel."""
        # Poll the session config for status=running
        config_path = self._session_config.path
        for _ in range(300):  # 5 minute timeout
            try:
                config = SessionConfig.load(config_path)
                if config.status == "running" and config.host:
                    self._session_config = config
                    logger.info(
                        "Remote kernel detected at %s", config.host
                    )
                    # Set connection info for the kernel manager
                    conn_info = {
                        "ip": config.host,
                        "shell_port": config.zmq_ports["shell"],
                        "iopub_port": config.zmq_ports["iopub"],
                        "stdin_port": config.zmq_ports["stdin"],
                        "control_port": config.zmq_ports["control"],
                        "hb_port": config.zmq_ports["hb"],
                    }
                    self.connection_info = conn_info
                    return self
            except Exception:
                pass
            await asyncio.sleep(1)

        msg = "Timeout waiting for distributed kernel to start"
        raise TimeoutError(msg)

    async def poll(self) -> int | None:
        """Check if kernel is still alive. None = alive."""
        if self._session_config:
            try:
                config = SessionConfig.load(self._session_config.path)
                if config.status == "running":
                    return None  # Still running
            except Exception:
                pass
        return None  # Assume alive until proven otherwise

    async def wait(self) -> int | None:
        """Wait for kernel to finish."""
        while True:
            poll = await self.poll()
            if poll is not None:
                return poll
            await asyncio.sleep(1)

    async def send_signal(self, signum: int) -> bool:
        """Send signal to kernel (not directly possible for remote)."""
        return False

    async def kill(self, restart: bool = False) -> None:
        """Kill the kernel."""
        if self._session_config:
            self._session_config.update(status="shutdown")

    async def terminate(self, restart: bool = False) -> None:
        """Terminate the kernel."""
        await self.kill(restart=restart)

    async def cleanup(self, restart: bool = False) -> None:
        """Clean up resources."""
        if self._session_config and not restart:
            try:
                self._session_config.path.unlink(missing_ok=True)
            except Exception:
                pass

    def get_provisioner_info(self) -> dict:
        return {
            "config_path": str(self._session_config.path)
            if self._session_config
            else None,
            "world_size": self.world_size,
            "gateway_port": self.gateway_port,
        }

    def load_provisioner_info(self, info: dict) -> None:
        config_path = info.get("config_path")
        if config_path and Path(config_path).exists():
            self._session_config = SessionConfig.load(config_path)
        self.world_size = info.get("world_size", self.world_size)
        self.gateway_port = info.get("gateway_port", self.gateway_port)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_distributed/test_provisioner.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add jupyterlab_distributed/provisioner.py tests/test_distributed/test_provisioner.py
git commit -m "feat(distributed): add kernel provisioner for remote distributed kernels"
```

---

### Task 8: Server-side Crash Log Handler

**Files:**
- Create: `jupyterlab_distributed/handlers.py`
- Create: `tests/test_distributed/test_handlers.py`

**Step 1: Write the failing test**

```python
# tests/test_distributed/test_handlers.py
import json
import os
import pytest
from jupyterlab_distributed.handlers import read_crash_logs, read_rank_logs


class TestHandlers:
    def test_read_crash_logs_empty(self, tmp_path):
        logs = read_crash_logs(str(tmp_path))
        assert logs == []

    def test_read_crash_logs_finds_files(self, tmp_path):
        crash = {
            "rank": 3, "hostname": "node-2",
            "signal": "SIGSEGV", "traceback": "...",
        }
        (tmp_path / "rank-3.crash.json").write_text(json.dumps(crash))
        logs = read_crash_logs(str(tmp_path))
        assert len(logs) == 1
        assert logs[0]["rank"] == 3

    def test_read_rank_logs(self, tmp_path):
        (tmp_path / "rank-0.log").write_text("hello from rank 0\n")
        log = read_rank_logs(str(tmp_path), rank=0, tail=10)
        assert "hello from rank 0" in log
```

**Step 2: Run, fail, implement, pass pattern**

```python
# jupyterlab_distributed/handlers.py
"""Server-side API handlers for crash logs and rank output retrieval."""

import json
import os
from pathlib import Path


def read_crash_logs(session_dir: str) -> list[dict]:
    """Read all crash report JSON files from a session directory."""
    logs = []
    session_path = Path(session_dir)
    if not session_path.exists():
        return logs
    for f in sorted(session_path.glob("*.crash.json")):
        try:
            logs.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return logs


def read_rank_logs(
    session_dir: str, rank: int, tail: int = 100
) -> str:
    """Read the last N lines of a rank's log file."""
    log_path = Path(session_dir) / f"rank-{rank}.log"
    if not log_path.exists():
        return ""
    lines = log_path.read_text().splitlines()
    return "\n".join(lines[-tail:])
```

Run: `python -m pytest tests/test_distributed/test_handlers.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add jupyterlab_distributed/handlers.py tests/test_distributed/test_handlers.py
git commit -m "feat(distributed): add crash log and rank log reader"
```

---

## Phase 3: Frontend — MIME Renderer and Rank Selector

### Task 9: TypeScript Package Scaffolding

**Files:**
- Create: `packages/distributed/package.json`
- Create: `packages/distributed/tsconfig.json`
- Create: `packages/distributed/src/index.ts`
- Create: `packages/distributed/src/tokens.ts`
- Create: `packages/distributed/src/types.ts`
- Create: `packages/distributed-extension/package.json`
- Create: `packages/distributed-extension/tsconfig.json`
- Create: `packages/distributed-extension/src/index.ts`
- Create: `packages/distributed-extension/style/index.css`
- Create: `packages/distributed-extension/style/index.js`

**Step 1: Create the shared types package**

`packages/distributed/package.json`:
```json
{
  "name": "@jupyterlab/distributed",
  "version": "4.6.0-alpha.3",
  "description": "JupyterLab - Distributed SPMD Types",
  "homepage": "https://github.com/jupyterlab/jupyterlab",
  "license": "BSD-3-Clause",
  "author": "Project Jupyter",
  "main": "lib/index.js",
  "types": "lib/index.d.ts",
  "directories": { "lib": "lib/" },
  "files": ["lib/*.d.ts", "lib/*.js.map", "lib/*.js", "src/**/*.ts"],
  "scripts": {
    "build": "tsc -b",
    "clean": "rimraf lib && rimraf tsconfig.tsbuildinfo",
    "watch": "tsc -b --watch"
  },
  "dependencies": {
    "@lumino/signaling": "^2.1.5"
  },
  "devDependencies": {
    "rimraf": "~5.0.5",
    "typescript": "~5.9.3"
  }
}
```

`packages/distributed/tsconfig.json`:
```json
{
  "extends": "../../tsconfigbase",
  "compilerOptions": { "outDir": "lib", "rootDir": "src" },
  "include": ["src/*"]
}
```

`packages/distributed/src/types.ts` — wire protocol types:
```typescript
export const DISTRIBUTED_MIME =
  'application/vnd.jupyterlab.distributed+json';

export interface IRankOutput {
  outputs: IWorkerOutput[];
  status: 'ok' | 'error' | 'timeout';
  execution_time: number;
}

export interface IWorkerOutput {
  type: 'stream' | 'display_data' | 'execute_result' | 'error';
  msg_id: string;
  name?: string;
  text?: string;
  data?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  ename?: string;
  evalue?: string;
  traceback?: string[];
}

export interface IRankOutputsMessage {
  type: 'rank_outputs';
  msg_id?: string;
  ranks: Record<string, IRankOutput>;
}

export interface IClusterStatusMessage {
  type: 'cluster_status';
  registered: number;
  expected: number;
  nodes: Record<string, INodeStatus>;
  gpu_memory_total_mb: number;
  gpu_memory_used_mb: number;
}

export interface INodeStatus {
  ranks: number[];
  status: 'healthy' | 'degraded' | 'disconnected';
}

export type IDistributedMessage =
  | IRankOutputsMessage
  | IClusterStatusMessage;
```

`packages/distributed/src/tokens.ts`:
```typescript
import { Token } from '@lumino/coreutils';
import type { ISignal } from '@lumino/signaling';
import type { IClusterStatusMessage } from './types';

export interface IDistributedStatus {
  readonly statusChanged: ISignal<IDistributedStatus, IClusterStatusMessage>;
  readonly isDistributed: boolean;
  readonly clusterStatus: IClusterStatusMessage | null;
}

export const IDistributedStatus = new Token<IDistributedStatus>(
  '@jupyterlab/distributed:IDistributedStatus',
  'Provides distributed cluster status.'
);
```

`packages/distributed/src/index.ts`:
```typescript
export * from './tokens';
export * from './types';
```

**Step 2: Create the extension package** (scaffold only, implementation in next tasks)

`packages/distributed-extension/package.json`:
```json
{
  "name": "@jupyterlab/distributed-extension",
  "version": "4.6.0-alpha.3",
  "description": "JupyterLab - Distributed SPMD Extension",
  "homepage": "https://github.com/jupyterlab/jupyterlab",
  "license": "BSD-3-Clause",
  "author": "Project Jupyter",
  "sideEffects": ["style/**/*.css", "style/index.js"],
  "main": "lib/index.js",
  "types": "lib/index.d.ts",
  "style": "style/index.css",
  "directories": { "lib": "lib/" },
  "files": [
    "lib/*.d.ts", "lib/*.js.map", "lib/*.js",
    "style/**/*.css", "style/index.js", "src/**/*.{ts,tsx}"
  ],
  "scripts": {
    "build": "tsc -b",
    "clean": "rimraf lib && rimraf tsconfig.tsbuildinfo",
    "watch": "tsc -b --watch"
  },
  "dependencies": {
    "@jupyterlab/application": "^4.6.0-alpha.3",
    "@jupyterlab/apputils": "^4.6.0-alpha.3",
    "@jupyterlab/distributed": "^4.6.0-alpha.3",
    "@jupyterlab/notebook": "^4.6.0-alpha.3",
    "@jupyterlab/rendermime": "^4.6.0-alpha.3",
    "@jupyterlab/rendermime-interfaces": "^3.14.0-alpha.3",
    "@jupyterlab/services": "^7.6.0-alpha.3",
    "@jupyterlab/translation": "^4.6.0-alpha.3",
    "@jupyterlab/ui-components": "^4.6.0-alpha.3",
    "@lumino/signaling": "^2.1.5",
    "@lumino/widgets": "^2.7.5",
    "react": "^18.2.0"
  },
  "devDependencies": {
    "rimraf": "~5.0.5",
    "typescript": "~5.9.3"
  },
  "jupyterlab": { "extension": true },
  "styleModule": "style/index.js"
}
```

`packages/distributed-extension/tsconfig.json`:
```json
{
  "extends": "../../tsconfigbase",
  "compilerOptions": { "outDir": "lib", "rootDir": "src" },
  "include": ["src/*"],
  "references": [
    { "path": "../application" },
    { "path": "../apputils" },
    { "path": "../distributed" },
    { "path": "../notebook" },
    { "path": "../rendermime" },
    { "path": "../rendermime-interfaces" },
    { "path": "../services" },
    { "path": "../translation" },
    { "path": "../ui-components" }
  ]
}
```

`packages/distributed-extension/src/index.ts`:
```typescript
import type {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

const plugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyterlab/distributed-extension:plugin',
  description: 'Provides distributed SPMD cell execution support.',
  autoStart: true,
  activate: (app: JupyterFrontEnd): void => {
    console.log('Distributed extension activated');
  }
};

export default [plugin];
```

`packages/distributed-extension/style/index.css`:
```css
/* Distributed extension styles */
```

`packages/distributed-extension/style/index.js`:
```javascript
import './index.css';
```

**Step 3: Verify TypeScript compiles**

Run: `cd packages/distributed && npx tsc -b --noEmit`
Expected: No errors

**Step 4: Commit**

```bash
git add packages/distributed/ packages/distributed-extension/
git commit -m "feat(distributed): scaffold TypeScript packages for distributed extension"
```

---

### Task 10: Custom MIME Renderer for Rank Outputs

**Files:**
- Create: `packages/distributed-extension/src/mimeRenderer.ts`
- Modify: `packages/distributed-extension/src/index.ts`

This is the core frontend component that renders per-rank outputs with the rank selector tab bar.

**Step 1: Implement the MIME renderer**

```typescript
// packages/distributed-extension/src/mimeRenderer.ts
import { IRenderMime } from '@jupyterlab/rendermime-interfaces';
import { Widget } from '@lumino/widgets';
import {
  DISTRIBUTED_MIME,
  type IRankOutputsMessage,
  type IRankOutput
} from '@jupyterlab/distributed';

const MIME_TYPE = DISTRIBUTED_MIME;

class DistributedOutputRenderer extends Widget
  implements IRenderMime.IRenderer
{
  private _selectedRank = 0;
  private _ranksData: Record<string, IRankOutput> = {};

  constructor() {
    super();
    this.addClass('jp-DistributedOutput');
  }

  async renderModel(model: IRenderMime.IMimeModel): Promise<void> {
    const data = model.data[MIME_TYPE] as unknown as IRankOutputsMessage;
    if (!data || data.type !== 'rank_outputs') {
      return;
    }
    this._ranksData = data.ranks;
    this._render();
  }

  private _render(): void {
    const node = this.node;
    node.innerHTML = '';

    const ranks = Object.keys(this._ranksData)
      .map(Number)
      .sort((a, b) => a - b);

    if (ranks.length === 0) {
      return;
    }

    // Rank tab bar
    const tabBar = document.createElement('div');
    tabBar.className = 'jp-DistributedOutput-tabBar';

    for (const rank of ranks) {
      const tab = document.createElement('button');
      tab.className = 'jp-DistributedOutput-tab';
      tab.textContent = `${rank}`;

      const rankData = this._ranksData[String(rank)];
      if (rankData.status === 'error') {
        tab.classList.add('jp-mod-error');
      } else if (rankData.status === 'ok') {
        tab.classList.add('jp-mod-ok');
      }
      if (rank === this._selectedRank) {
        tab.classList.add('jp-mod-active');
      }

      tab.addEventListener('click', () => {
        this._selectedRank = rank;
        this._render();
      });
      tabBar.appendChild(tab);
    }

    // "All" button
    const allBtn = document.createElement('button');
    allBtn.className = 'jp-DistributedOutput-tab jp-DistributedOutput-allTab';
    allBtn.textContent = 'All Ranks';
    allBtn.addEventListener('click', () => {
      this._selectedRank = -1; // -1 means "all"
      this._render();
    });
    if (this._selectedRank === -1) {
      allBtn.classList.add('jp-mod-active');
    }
    tabBar.appendChild(allBtn);

    node.appendChild(tabBar);

    // Output area
    const outputArea = document.createElement('div');
    outputArea.className = 'jp-DistributedOutput-content';

    if (this._selectedRank === -1) {
      // Show all ranks
      for (const rank of ranks) {
        const section = this._renderRankSection(rank);
        outputArea.appendChild(section);
      }
    } else {
      const section = this._renderRankOutputs(this._selectedRank);
      outputArea.appendChild(section);
    }

    node.appendChild(outputArea);
  }

  private _renderRankSection(rank: number): HTMLElement {
    const section = document.createElement('details');
    section.className = 'jp-DistributedOutput-rankSection';

    const summary = document.createElement('summary');
    const rankData = this._ranksData[String(rank)];
    const statusIcon = rankData.status === 'ok' ? '\u2713' : '\u2717';
    summary.textContent =
      `Rank ${rank} ${statusIcon} (${rankData.execution_time.toFixed(2)}s)`;
    section.appendChild(summary);

    const content = this._renderRankOutputs(rank);
    section.appendChild(content);

    return section;
  }

  private _renderRankOutputs(rank: number): HTMLElement {
    const container = document.createElement('div');
    container.className = 'jp-DistributedOutput-rankContent';

    const rankData = this._ranksData[String(rank)];
    if (!rankData) {
      container.textContent = `No output for rank ${rank}`;
      return container;
    }

    for (const output of rankData.outputs) {
      const el = document.createElement('pre');
      el.className = 'jp-DistributedOutput-outputItem';

      if (output.type === 'stream') {
        el.textContent = output.text || '';
        if (output.name === 'stderr') {
          el.classList.add('jp-mod-stderr');
        }
      } else if (output.type === 'error') {
        el.textContent = (output.traceback || []).join('\n');
        el.classList.add('jp-mod-error');
      } else if (output.type === 'display_data' || output.type === 'execute_result') {
        const textData = output.data?.['text/plain'];
        el.textContent = typeof textData === 'string' ? textData : JSON.stringify(output.data);
      }

      container.appendChild(el);
    }

    if (rankData.outputs.length === 0) {
      container.textContent = '(no output)';
    }

    return container;
  }
}

export const distributedRendererFactory: IRenderMime.IRendererFactory = {
  safe: true,
  mimeTypes: [MIME_TYPE],
  defaultRank: 80,
  createRenderer: () => new DistributedOutputRenderer()
};
```

**Step 2: Update the plugin index to register the renderer**

```typescript
// packages/distributed-extension/src/index.ts
import type {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import type { IRenderMime } from '@jupyterlab/rendermime-interfaces';
import { distributedRendererFactory } from './mimeRenderer';

const mimePlugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyterlab/distributed-extension:mime-renderer',
  description: 'Renders per-rank distributed output with rank selector.',
  autoStart: true,
  activate: (app: JupyterFrontEnd): void => {
    app.docRegistry.addWidgetExtension('Notebook', {
      createNew: () => ({}) as any // Placeholder — renderer registered via rendermime
    });
    console.log('Distributed MIME renderer registered');
  }
};

export const rendererFactory: IRenderMime.IRendererFactory =
  distributedRendererFactory;

export default [mimePlugin];
```

**Step 3: Add CSS styles**

```css
/* packages/distributed-extension/style/index.css */

.jp-DistributedOutput-tabBar {
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
  padding: 4px;
  border-bottom: 1px solid var(--jp-border-color1);
  background: var(--jp-layout-color1);
}

.jp-DistributedOutput-tab {
  padding: 2px 8px;
  border: 1px solid var(--jp-border-color1);
  border-radius: 3px;
  background: var(--jp-layout-color2);
  cursor: pointer;
  font-size: 11px;
  font-family: var(--jp-code-font-family);
}

.jp-DistributedOutput-tab:hover {
  background: var(--jp-layout-color3);
}

.jp-DistributedOutput-tab.jp-mod-active {
  background: var(--jp-brand-color1);
  color: white;
  border-color: var(--jp-brand-color1);
}

.jp-DistributedOutput-tab.jp-mod-ok {
  border-left: 3px solid var(--jp-success-color1);
}

.jp-DistributedOutput-tab.jp-mod-error {
  border-left: 3px solid var(--jp-error-color1);
}

.jp-DistributedOutput-allTab {
  margin-left: auto;
}

.jp-DistributedOutput-content {
  padding: 4px;
}

.jp-DistributedOutput-rankSection {
  margin: 4px 0;
  border: 1px solid var(--jp-border-color2);
  border-radius: 3px;
}

.jp-DistributedOutput-rankSection > summary {
  padding: 4px 8px;
  background: var(--jp-layout-color2);
  cursor: pointer;
  font-family: var(--jp-code-font-family);
  font-size: 12px;
}

.jp-DistributedOutput-outputItem {
  margin: 2px 0;
  padding: 4px 8px;
  white-space: pre-wrap;
  font-family: var(--jp-code-font-family);
  font-size: var(--jp-code-font-size);
}

.jp-DistributedOutput-outputItem.jp-mod-stderr {
  color: var(--jp-warn-color1);
}

.jp-DistributedOutput-outputItem.jp-mod-error {
  color: var(--jp-error-color1);
}
```

**Step 4: Verify TypeScript compiles**

Run: `cd packages/distributed-extension && npx tsc -b --noEmit`
Expected: No errors

**Step 5: Commit**

```bash
git add packages/distributed-extension/
git commit -m "feat(distributed): add custom MIME renderer with rank selector tab bar"
```

---

### Task 11: Distributed Status Sidebar Panel

**Files:**
- Create: `packages/distributed-extension/src/panel.ts`
- Modify: `packages/distributed-extension/src/index.ts`

Pattern from: `packages/running-extension/src/index.ts:79-129`

**Step 1: Implement the status panel**

```typescript
// packages/distributed-extension/src/panel.ts
import { Widget } from '@lumino/widgets';
import { Signal, ISignal } from '@lumino/signaling';
import type { IClusterStatusMessage, INodeStatus } from '@jupyterlab/distributed';

export class DistributedStatusPanel extends Widget {
  private _status: IClusterStatusMessage | null = null;

  constructor() {
    super();
    this.id = 'jp-distributed-status';
    this.addClass('jp-DistributedStatusPanel');
    this._render();
  }

  updateStatus(status: IClusterStatusMessage): void {
    this._status = status;
    this._render();
  }

  private _render(): void {
    const node = this.node;
    node.innerHTML = '';

    if (!this._status) {
      node.innerHTML = '<div class="jp-DistributedStatusPanel-empty">'
        + 'No distributed session active.</div>';
      return;
    }

    const s = this._status;

    // Session info
    const header = document.createElement('div');
    header.className = 'jp-DistributedStatusPanel-header';
    header.innerHTML = `
      <div class="jp-DistributedStatusPanel-row">
        <span>Workers:</span>
        <strong>${s.registered}/${s.expected}</strong>
      </div>
    `;
    node.appendChild(header);

    // Progress bar
    const progress = document.createElement('div');
    progress.className = 'jp-DistributedStatusPanel-progress';
    const pct = s.expected > 0 ? (s.registered / s.expected) * 100 : 0;
    progress.innerHTML = `<div class="jp-DistributedStatusPanel-progressBar" `
      + `style="width: ${pct}%"></div>`;
    node.appendChild(progress);

    // Nodes tree
    const nodesSection = document.createElement('div');
    nodesSection.className = 'jp-DistributedStatusPanel-nodes';

    for (const [hostname, nodeStatus] of Object.entries(s.nodes)) {
      const nodeEl = document.createElement('details');
      nodeEl.open = true;
      const summary = document.createElement('summary');
      const statusDot = nodeStatus.status === 'healthy' ? '\u25CF' : '\u25CB';
      summary.textContent = `${statusDot} ${hostname} (${nodeStatus.ranks.length} ranks)`;
      nodeEl.appendChild(summary);

      const rankList = document.createElement('div');
      rankList.className = 'jp-DistributedStatusPanel-rankList';
      for (const rank of nodeStatus.ranks) {
        const rankEl = document.createElement('span');
        rankEl.className = 'jp-DistributedStatusPanel-rank';
        rankEl.textContent = `r${rank}`;
        rankList.appendChild(rankEl);
      }
      nodeEl.appendChild(rankList);
      nodesSection.appendChild(nodeEl);
    }
    node.appendChild(nodesSection);

    // GPU memory
    if (s.gpu_memory_total_mb > 0) {
      const mem = document.createElement('div');
      mem.className = 'jp-DistributedStatusPanel-memory';
      const usedGB = (s.gpu_memory_used_mb / 1024).toFixed(1);
      const totalGB = (s.gpu_memory_total_mb / 1024).toFixed(1);
      const memPct = (s.gpu_memory_used_mb / s.gpu_memory_total_mb) * 100;
      mem.innerHTML = `
        <div>GPU Memory: ${usedGB}/${totalGB} GB</div>
        <div class="jp-DistributedStatusPanel-progress">
          <div class="jp-DistributedStatusPanel-progressBar
            ${memPct > 90 ? 'jp-mod-warning' : ''}"
            style="width: ${memPct}%"></div>
        </div>
      `;
      node.appendChild(mem);
    }
  }
}
```

**Step 2: Register the sidebar plugin**

Update `packages/distributed-extension/src/index.ts` to add the sidebar plugin (follow `running-extension/src/index.ts:79-129` pattern):

```typescript
// Add to packages/distributed-extension/src/index.ts
import { ILabShell, ILayoutRestorer } from '@jupyterlab/application';
import { ITranslator } from '@jupyterlab/translation';
import { DistributedStatusPanel } from './panel';

const sidebarPlugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyterlab/distributed-extension:sidebar',
  description: 'Provides distributed cluster status sidebar.',
  autoStart: true,
  requires: [ITranslator],
  optional: [ILayoutRestorer],
  activate: (
    app: JupyterFrontEnd,
    translator: ITranslator,
    restorer: ILayoutRestorer | null
  ): void => {
    const panel = new DistributedStatusPanel();
    panel.title.caption = 'Distributed Cluster';

    if (restorer) {
      restorer.add(panel, 'distributed-status');
    }
    app.shell.add(panel, 'left', { rank: 300, type: 'Distributed' });
  }
};
```

Add `sidebarPlugin` to the default export array.

**Step 3: Add panel CSS to `style/index.css`**

```css
/* Append to packages/distributed-extension/style/index.css */

.jp-DistributedStatusPanel {
  padding: 8px;
  overflow-y: auto;
}

.jp-DistributedStatusPanel-empty {
  padding: 16px;
  text-align: center;
  color: var(--jp-ui-font-color2);
}

.jp-DistributedStatusPanel-header {
  margin-bottom: 8px;
}

.jp-DistributedStatusPanel-row {
  display: flex;
  justify-content: space-between;
  padding: 2px 0;
}

.jp-DistributedStatusPanel-progress {
  height: 6px;
  background: var(--jp-layout-color3);
  border-radius: 3px;
  margin: 4px 0 8px;
  overflow: hidden;
}

.jp-DistributedStatusPanel-progressBar {
  height: 100%;
  background: var(--jp-brand-color1);
  border-radius: 3px;
  transition: width 0.3s;
}

.jp-DistributedStatusPanel-progressBar.jp-mod-warning {
  background: var(--jp-warn-color1);
}

.jp-DistributedStatusPanel-nodes {
  margin: 8px 0;
}

.jp-DistributedStatusPanel-nodes details {
  margin: 4px 0;
}

.jp-DistributedStatusPanel-nodes summary {
  cursor: pointer;
  font-size: 12px;
  padding: 2px 0;
}

.jp-DistributedStatusPanel-rankList {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  padding: 4px 16px;
}

.jp-DistributedStatusPanel-rank {
  font-family: var(--jp-code-font-family);
  font-size: 10px;
  padding: 1px 4px;
  background: var(--jp-layout-color2);
  border-radius: 2px;
  border: 1px solid var(--jp-border-color2);
}

.jp-DistributedStatusPanel-memory {
  margin-top: 8px;
  font-size: 12px;
}
```

**Step 4: Verify compilation**

Run: `cd packages/distributed-extension && npx tsc -b --noEmit`

**Step 5: Commit**

```bash
git add packages/distributed-extension/
git commit -m "feat(distributed): add distributed cluster status sidebar panel"
```

---

## Phase 4-6: Hardening, Integration, Polish

### Task 12: Integration Test — End-to-End Local

**Files:**
- Create: `tests/test_distributed/test_integration.py`

Write an integration test that starts a gateway, two workers (in-process), and the distributed kernel, executes a cell, and verifies per-rank output. This test runs locally with no SLURM or torchrun.

```python
# tests/test_distributed/test_integration.py
import asyncio
import json
import pytest
from jupyterlab_distributed.gateway import Gateway
from jupyterlab_distributed.worker import Worker


class TestIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_two_workers(self):
        """Full flow: gateway + 2 workers + broadcast + collect."""
        gw = Gateway(port=0, auth_token="e2e-token", expected_workers=2)
        await gw.start()
        url = f"ws://localhost:{gw.port}/ws"

        w1 = Worker(rank=1, server_url=url, auth_token="e2e-token")
        w2 = Worker(rank=2, server_url=url, auth_token="e2e-token")
        t1 = asyncio.create_task(w1.run())
        t2 = asyncio.create_task(w2.run())

        try:
            # Wait for registration
            for _ in range(100):
                if gw.all_workers_registered():
                    break
                await asyncio.sleep(0.1)
            assert gw.all_workers_registered()

            # Execute a cell
            msg_id = gw.broadcast_execute(
                "import os; print(f'hello from worker')", "cell-e2e"
            )
            results = await asyncio.wait_for(
                gw.collect_results(msg_id), timeout=15
            )

            # Both workers should report success
            assert 1 in results
            assert 2 in results
            assert results[1]["status"] == "ok"
            assert results[2]["status"] == "ok"

            # Both should have stdout output
            for rank in [1, 2]:
                stdout = [
                    o for o in results[rank]["outputs"]
                    if o.get("type") == "stream" and o.get("name") == "stdout"
                ]
                assert any("hello from worker" in o.get("text", "") for o in stdout)

        finally:
            w1.shutdown()
            w2.shutdown()
            t1.cancel()
            t2.cancel()
            await gw.stop()
            for t in [t1, t2]:
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_reset_clears_all_workers(self):
        """Soft restart: reset clears namespace on all workers."""
        gw = Gateway(port=0, auth_token="rst-token", expected_workers=2)
        await gw.start()
        url = f"ws://localhost:{gw.port}/ws"

        w1 = Worker(rank=1, server_url=url, auth_token="rst-token")
        w2 = Worker(rank=2, server_url=url, auth_token="rst-token")
        t1 = asyncio.create_task(w1.run())
        t2 = asyncio.create_task(w2.run())

        try:
            for _ in range(100):
                if gw.all_workers_registered():
                    break
                await asyncio.sleep(0.1)

            # Define variable
            msg_id = gw.broadcast_execute("test_var = 42", "cell-rst-1")
            await asyncio.wait_for(gw.collect_results(msg_id), timeout=10)

            # Reset
            gw.broadcast_reset()
            await asyncio.sleep(1)

            # Variable should be gone
            msg_id = gw.broadcast_execute("print(test_var)", "cell-rst-2")
            results = await asyncio.wait_for(
                gw.collect_results(msg_id), timeout=10
            )
            assert results[1]["status"] == "error"
            assert results[2]["status"] == "error"
        finally:
            w1.shutdown()
            w2.shutdown()
            t1.cancel()
            t2.cancel()
            await gw.stop()
            for t in [t1, t2]:
                try:
                    await t
                except asyncio.CancelledError:
                    pass
```

Run: `python -m pytest tests/test_distributed/test_integration.py -v`

**Commit:**
```bash
git add tests/test_distributed/test_integration.py
git commit -m "test(distributed): add end-to-end integration tests"
```

---

### Task 13: Kernel Spec Registration

**Files:**
- Create: `jupyterlab_distributed/kernelspec/kernel.json`
- Modify: `jupyterlab_distributed/__init__.py`

Create the kernel spec so "Distributed Python" appears in the kernel selector.

```json
{
  "argv": [
    "python", "-m", "ipykernel_launcher", "-f", "{connection_file}"
  ],
  "display_name": "Distributed Python",
  "language": "python",
  "metadata": {
    "kernel_provisioner": {
      "provisioner_name": "distributed-provisioner"
    }
  }
}
```

Register the provisioner via entry point in `pyproject.toml` or `setup.cfg`:

```toml
[project.entry-points."jupyter_client.kernel_provisioners"]
distributed-provisioner = "jupyterlab_distributed.provisioner:DistributedProvisioner"
```

**Commit:**
```bash
git add jupyterlab_distributed/kernelspec/ pyproject.toml
git commit -m "feat(distributed): add kernel spec and provisioner entry point"
```

---

### Summary of All Tasks

| # | Task | Phase | Files |
|---|------|-------|-------|
| 1 | Session Config Module | 1 | `config.py`, tests |
| 2 | WebSocket Gateway | 1 | `gateway.py`, tests |
| 3 | Worker Daemon | 1 | `worker.py`, tests |
| 4 | Distributed Kernel | 1 | `kernel.py`, tests |
| 5 | Launcher Entry Point | 1 | `launcher.py`, tests |
| 6 | Magic Commands | 1 | `magics.py`, tests |
| 7 | Kernel Provisioner | 2 | `provisioner.py`, tests |
| 8 | Crash Log Handler | 2 | `handlers.py`, tests |
| 9 | TS Package Scaffolding | 3 | `distributed/`, `distributed-extension/` |
| 10 | MIME Renderer + Rank Selector | 3 | `mimeRenderer.ts`, CSS |
| 11 | Status Sidebar Panel | 3 | `panel.ts`, CSS |
| 12 | Integration Tests | 4 | `test_integration.py` |
| 13 | Kernel Spec Registration | 4 | `kernel.json`, entry points |
