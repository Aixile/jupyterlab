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

    @pytest.mark.asyncio
    async def test_send_to_rank_delivers_to_single_worker(self, gateway, gateway_url):
        """send_to_rank sends an execute message only to the specified rank."""
        ws0, _ = await _register_worker(gateway_url, rank=0)
        ws1, _ = await _register_worker(gateway_url, rank=1)
        try:
            msg_id = await gateway.send_to_rank(1, "x = 42", cell_id="cell-r1")

            # Rank 1 should receive the message
            raw = await asyncio.wait_for(ws1.recv(), timeout=5.0)
            msg = json.loads(raw)
            assert msg["type"] == "execute"
            assert msg["code"] == "x = 42"
            assert msg["cell_id"] == "cell-r1"
            assert msg["msg_id"] == msg_id

            # Rank 0 should NOT receive any message (with a short timeout)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(ws0.recv(), timeout=0.3)
        finally:
            await ws0.close()
            await ws1.close()

    @pytest.mark.asyncio
    async def test_send_to_rank_raises_for_unknown_rank(self, gateway, gateway_url):
        """send_to_rank raises KeyError when the target rank is not connected."""
        ws0, _ = await _register_worker(gateway_url, rank=0)
        try:
            with pytest.raises(KeyError, match="Rank 99"):
                await gateway.send_to_rank(99, "code", cell_id="cell-x")
        finally:
            await ws0.close()

    @pytest.mark.asyncio
    async def test_send_to_rank_completion_from_single_worker(self, gateway, gateway_url):
        """send_to_rank completes when the targeted rank responds."""
        ws0, _ = await _register_worker(gateway_url, rank=0)
        ws1, _ = await _register_worker(gateway_url, rank=1)
        try:
            msg_id = await gateway.send_to_rank(1, "y = 1", cell_id="cell-s")

            # Rank 1 receives the execute message
            await asyncio.wait_for(ws1.recv(), timeout=5.0)

            # Only rank 1 responds with execute_complete
            complete_msg = json.dumps({
                "type": "execute_complete",
                "msg_id": msg_id,
                "status": "ok",
            })
            await ws1.send(complete_msg)

            # collect_results should complete (not wait for rank 0)
            results = await asyncio.wait_for(
                gateway.collect_results(msg_id), timeout=5.0
            )
            assert 1 in results
            assert results[1]["status"] == "ok"
            assert 0 not in results
        finally:
            await ws0.close()
            await ws1.close()

    @pytest.mark.asyncio
    async def test_broadcast_execute_snapshots_target_ranks(self, gateway, gateway_url):
        """Completion check uses the rank snapshot from broadcast time, not current workers."""
        ws0, _ = await _register_worker(gateway_url, rank=0)
        ws1, _ = await _register_worker(gateway_url, rank=1)
        try:
            msg_id = await gateway.broadcast_execute("z = 1", "cell-snap")

            # Verify the snapshot was taken
            assert msg_id in gateway._target_ranks
            assert gateway._target_ranks[msg_id] == {0, 1}

            # Both workers receive the execute message
            await asyncio.wait_for(ws0.recv(), timeout=5.0)
            await asyncio.wait_for(ws1.recv(), timeout=5.0)

            # Now a third worker joins (should NOT affect this execution)
            ws2, _ = await _register_worker(gateway_url, rank=2)
            try:
                # Only the original 2 workers respond
                for ws, rank in [(ws0, 0), (ws1, 1)]:
                    complete_msg = json.dumps({
                        "type": "execute_complete",
                        "msg_id": msg_id,
                        "status": "ok",
                    })
                    await ws.send(complete_msg)

                # Should complete even though there are now 3 workers
                results = await asyncio.wait_for(
                    gateway.collect_results(msg_id), timeout=5.0
                )
                assert 0 in results
                assert 1 in results
                assert 2 not in results
            finally:
                await ws2.close()
        finally:
            await ws0.close()
            await ws1.close()

    @pytest.mark.asyncio
    async def test_outputs_included_in_collect_results(self, gateway, gateway_url):
        """Buffered stream outputs are merged into results returned by collect_results."""
        ws0, _ = await _register_worker(gateway_url, rank=0)
        ws1, _ = await _register_worker(gateway_url, rank=1)
        try:
            msg_id = await gateway.broadcast_execute("print('hi')", "cell-out")

            # Both workers receive the execute message
            await asyncio.wait_for(ws0.recv(), timeout=5.0)
            await asyncio.wait_for(ws1.recv(), timeout=5.0)

            # Workers send stream output before execute_complete
            for ws, rank in [(ws0, 0), (ws1, 1)]:
                stream_msg = json.dumps({
                    "type": "stream",
                    "msg_id": msg_id,
                    "name": "stdout",
                    "text": f"output-from-{rank}",
                })
                await ws.send(stream_msg)

            # Workers send execute_complete
            for ws, rank in [(ws0, 0), (ws1, 1)]:
                complete_msg = json.dumps({
                    "type": "execute_complete",
                    "msg_id": msg_id,
                    "status": "ok",
                })
                await ws.send(complete_msg)

            results = await asyncio.wait_for(
                gateway.collect_results(msg_id), timeout=5.0
            )

            # Verify outputs are merged into results
            assert "outputs" in results[0]
            assert len(results[0]["outputs"]) == 1
            assert results[0]["outputs"][0]["text"] == "output-from-0"
            assert "outputs" in results[1]
            assert len(results[1]["outputs"]) == 1
            assert results[1]["outputs"][0]["text"] == "output-from-1"
        finally:
            await ws0.close()
            await ws1.close()
