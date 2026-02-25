"""Tests for the WebSocket Gateway."""

import asyncio
import json

import pytest
import pytest_asyncio
import websockets

from jupyterlab_distributed.gateway import Gateway


@pytest_asyncio.fixture
async def gateway():
    gw = Gateway(port=0, auth_token="test-secret", expected_workers=2)
    await gw.start()
    yield gw
    await gw.stop()


@pytest.fixture
def gateway_url(gateway):
    return f"ws://localhost:{gateway.port}/ws"


async def _register_worker(gateway_url, rank, token="test-secret"):
    """Helper: connect and register a worker, return the websocket and response."""
    ws = await websockets.connect(gateway_url)
    register_msg = json.dumps({
        "type": "register",
        "rank": rank,
        "hostname": f"host-{rank}",
        "gpu_id": rank,
        "pid": 1000 + rank,
        "token": token,
    })
    await ws.send(register_msg)
    response = json.loads(await ws.recv())
    return ws, response


class TestGateway:
    @pytest.mark.asyncio
    async def test_rejects_bad_auth(self, gateway, gateway_url):
        """Connect with wrong token, expect error response with 'auth' in message."""
        ws = await websockets.connect(gateway_url)
        register_msg = json.dumps({
            "type": "register",
            "rank": 0,
            "hostname": "bad-host",
            "gpu_id": 0,
            "pid": 9999,
            "token": "wrong-token",
        })
        await ws.send(register_msg)
        response = json.loads(await ws.recv())
        assert response["type"] == "error"
        assert "auth" in response["message"].lower()
        # Connection should be closed by server
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await ws.recv()

    @pytest.mark.asyncio
    async def test_accepts_valid_registration(self, gateway, gateway_url):
        """Connect with correct token, expect 'registered' response."""
        ws, response = await _register_worker(gateway_url, rank=0)
        try:
            assert response["type"] == "registered"
            assert response["rank"] == 0
            assert response["world_size"] == 2
            assert 0 in gateway.workers
        finally:
            await ws.close()

    @pytest.mark.asyncio
    async def test_broadcast_execute(self, gateway, gateway_url):
        """Register a worker, broadcast execute, verify worker receives message."""
        ws, _ = await _register_worker(gateway_url, rank=0)
        try:
            msg_id = await gateway.broadcast_execute("print('hello')", "cell-1")

            # Worker should receive an execute message
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            msg = json.loads(raw)
            assert msg["type"] == "execute"
            assert msg["code"] == "print('hello')"
            assert msg["cell_id"] == "cell-1"
            assert msg["msg_id"] == msg_id
        finally:
            await ws.close()

    @pytest.mark.asyncio
    async def test_collect_worker_output(self, gateway, gateway_url):
        """Register worker, broadcast, worker sends execute_complete, verify collect_results."""
        ws0, _ = await _register_worker(gateway_url, rank=0)
        ws1, _ = await _register_worker(gateway_url, rank=1)
        try:
            msg_id = await gateway.broadcast_execute("x = 1", "cell-2")

            # Both workers receive the execute message
            await asyncio.wait_for(ws0.recv(), timeout=5.0)
            await asyncio.wait_for(ws1.recv(), timeout=5.0)

            # Both workers send execute_complete
            for ws, rank in [(ws0, 0), (ws1, 1)]:
                complete_msg = json.dumps({
                    "type": "execute_complete",
                    "msg_id": msg_id,
                    "rank": rank,
                    "status": "ok",
                    "result": {"data": f"result-from-{rank}"},
                })
                await ws.send(complete_msg)

            # Collect results should return data from both workers
            results = await asyncio.wait_for(
                gateway.collect_results(msg_id), timeout=5.0
            )
            assert 0 in results
            assert 1 in results
            assert results[0]["status"] == "ok"
            assert results[1]["result"]["data"] == "result-from-1"
        finally:
            await ws0.close()
            await ws1.close()

    @pytest.mark.asyncio
    async def test_broadcast_interrupt(self, gateway, gateway_url):
        """Register worker, broadcast interrupt, verify worker receives interrupt."""
        ws, _ = await _register_worker(gateway_url, rank=0)
        try:
            await gateway.broadcast_interrupt()

            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            msg = json.loads(raw)
            assert msg["type"] == "interrupt"
        finally:
            await ws.close()

    @pytest.mark.asyncio
    async def test_all_workers_registered(self, gateway, gateway_url):
        """Verify returns False with 0/2, False with 1/2, True with 2/2 workers."""
        # No workers yet
        assert gateway.all_workers_registered() is False

        # Register one worker
        ws0, _ = await _register_worker(gateway_url, rank=0)
        try:
            assert gateway.all_workers_registered() is False

            # Register second worker
            ws1, _ = await _register_worker(gateway_url, rank=1)
            try:
                assert gateway.all_workers_registered() is True
            finally:
                await ws1.close()
        finally:
            await ws0.close()
