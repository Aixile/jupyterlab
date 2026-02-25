"""Tests for the Worker daemon with embedded IPython shell."""

import asyncio
import json

import pytest
import pytest_asyncio

from jupyterlab_distributed.gateway import Gateway
from jupyterlab_distributed.worker import Worker


@pytest_asyncio.fixture
async def gateway():
    gw = Gateway(port=0, auth_token="test-token", expected_workers=1)
    await gw.start()
    yield gw
    await gw.stop()


@pytest.fixture
def gateway_url(gateway):
    return f"ws://localhost:{gateway.port}/ws"


class TestWorker:
    @pytest.mark.asyncio
    async def test_worker_connects_and_registers(self, gateway, gateway_url):
        """Verify rank 1 appears in gateway.workers after worker connects."""
        worker = Worker(rank=1, server_url=gateway_url, auth_token="test-token")
        task = asyncio.create_task(worker.run())
        try:
            # Wait for the worker to register
            for _ in range(50):
                if 1 in gateway.workers:
                    break
                await asyncio.sleep(0.05)
            assert 1 in gateway.workers
            assert gateway.workers[1].rank == 1
        finally:
            worker.shutdown()
            await asyncio.wait_for(task, timeout=5.0)

    @pytest.mark.asyncio
    async def test_worker_executes_code(self, gateway, gateway_url):
        """Broadcast 'x = 42', verify status='ok' in execute_complete."""
        worker = Worker(rank=1, server_url=gateway_url, auth_token="test-token")
        task = asyncio.create_task(worker.run())
        try:
            # Wait for registration
            for _ in range(50):
                if 1 in gateway.workers:
                    break
                await asyncio.sleep(0.05)
            assert 1 in gateway.workers

            msg_id = await gateway.broadcast_execute("x = 42", "cell-1")
            results = await asyncio.wait_for(
                gateway.collect_results(msg_id), timeout=10.0
            )
            assert 1 in results
            assert results[1]["status"] == "ok"
        finally:
            worker.shutdown()
            await asyncio.wait_for(task, timeout=5.0)

    @pytest.mark.asyncio
    async def test_worker_captures_stdout(self, gateway, gateway_url):
        """Broadcast print('hello world'), verify stdout captured in outputs."""
        worker = Worker(rank=1, server_url=gateway_url, auth_token="test-token")
        task = asyncio.create_task(worker.run())
        try:
            # Wait for registration
            for _ in range(50):
                if 1 in gateway.workers:
                    break
                await asyncio.sleep(0.05)
            assert 1 in gateway.workers

            msg_id = await gateway.broadcast_execute(
                "print('hello world')", "cell-2"
            )
            results = await asyncio.wait_for(
                gateway.collect_results(msg_id), timeout=10.0
            )
            assert results[1]["status"] == "ok"

            # Check that stdout was captured in outputs
            outputs = gateway.get_outputs(msg_id)
            assert 1 in outputs
            stdout_msgs = [
                o for o in outputs[1]
                if o["type"] == "stream" and o.get("name") == "stdout"
            ]
            assert len(stdout_msgs) > 0
            assert "hello world" in stdout_msgs[0]["text"]
        finally:
            worker.shutdown()
            await asyncio.wait_for(task, timeout=5.0)

    @pytest.mark.asyncio
    async def test_worker_captures_error(self, gateway, gateway_url):
        """Broadcast '1/0', verify status='error' and error info."""
        worker = Worker(rank=1, server_url=gateway_url, auth_token="test-token")
        task = asyncio.create_task(worker.run())
        try:
            # Wait for registration
            for _ in range(50):
                if 1 in gateway.workers:
                    break
                await asyncio.sleep(0.05)
            assert 1 in gateway.workers

            msg_id = await gateway.broadcast_execute("1/0", "cell-3")
            results = await asyncio.wait_for(
                gateway.collect_results(msg_id), timeout=10.0
            )
            assert 1 in results
            assert results[1]["status"] == "error"
            assert results[1]["ename"] == "ZeroDivisionError"
        finally:
            worker.shutdown()
            await asyncio.wait_for(task, timeout=5.0)

    @pytest.mark.asyncio
    async def test_worker_reset_clears_namespace(self, gateway, gateway_url):
        """Define a variable, reset, verify NameError when accessing it."""
        worker = Worker(rank=1, server_url=gateway_url, auth_token="test-token")
        task = asyncio.create_task(worker.run())
        try:
            # Wait for registration
            for _ in range(50):
                if 1 in gateway.workers:
                    break
                await asyncio.sleep(0.05)
            assert 1 in gateway.workers

            # Define a variable
            msg_id = await gateway.broadcast_execute(
                "reset_test_var = 999", "cell-4"
            )
            results = await asyncio.wait_for(
                gateway.collect_results(msg_id), timeout=10.0
            )
            assert results[1]["status"] == "ok"

            # Reset the namespace
            await gateway.broadcast_reset()
            # Give the worker time to process the reset
            await asyncio.sleep(0.2)

            # Try to access the variable — should get NameError
            msg_id2 = await gateway.broadcast_execute(
                "reset_test_var", "cell-5"
            )
            results2 = await asyncio.wait_for(
                gateway.collect_results(msg_id2), timeout=10.0
            )
            assert 1 in results2
            assert results2[1]["status"] == "error"
            assert results2[1]["ename"] == "NameError"
        finally:
            worker.shutdown()
            await asyncio.wait_for(task, timeout=5.0)
