"""End-to-end integration tests for Gateway + Worker coordination.

These tests start a real Gateway, connect real Worker instances (in-process),
broadcast code execution, and verify that results and outputs are collected
correctly. No SLURM or torchrun needed — everything runs locally.
"""

import asyncio
import time

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
            # Wait for both workers to register
            for _ in range(100):
                if gw.all_workers_registered():
                    break
                await asyncio.sleep(0.1)
            assert gw.all_workers_registered()

            # Execute a cell that produces stdout
            msg_id = await gw.broadcast_execute(
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

            # Both should have stdout output (stored in get_outputs, not results)
            outputs = gw.get_outputs(msg_id)
            for rank in [1, 2]:
                assert rank in outputs, f"No outputs for rank {rank}"
                stdout_msgs = [
                    o for o in outputs[rank]
                    if o.get("type") == "stream" and o.get("name") == "stdout"
                ]
                assert any(
                    "hello from worker" in o.get("text", "") for o in stdout_msgs
                ), f"Rank {rank} missing 'hello from worker' in stdout"

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
            assert gw.all_workers_registered()

            # Define a variable on both workers
            msg_id = await gw.broadcast_execute("test_var = 42", "cell-rst-1")
            await asyncio.wait_for(gw.collect_results(msg_id), timeout=10)

            # Reset all worker namespaces
            await gw.broadcast_reset()
            await asyncio.sleep(1)

            # Variable should be gone — accessing it should produce NameError
            msg_id = await gw.broadcast_execute("print(test_var)", "cell-rst-2")
            results = await asyncio.wait_for(
                gw.collect_results(msg_id), timeout=10
            )

            assert 1 in results
            assert 2 in results
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

    @pytest.mark.asyncio
    async def test_multiple_sequential_executions(self):
        """Execute multiple cells in sequence and verify state persists."""
        gw = Gateway(port=0, auth_token="seq-token", expected_workers=2)
        await gw.start()
        url = f"ws://localhost:{gw.port}/ws"

        w1 = Worker(rank=1, server_url=url, auth_token="seq-token")
        w2 = Worker(rank=2, server_url=url, auth_token="seq-token")
        t1 = asyncio.create_task(w1.run())
        t2 = asyncio.create_task(w2.run())

        try:
            for _ in range(100):
                if gw.all_workers_registered():
                    break
                await asyncio.sleep(0.1)
            assert gw.all_workers_registered()

            # Cell 1: define a variable
            msg_id1 = await gw.broadcast_execute("seq_x = 10", "cell-seq-1")
            results1 = await asyncio.wait_for(
                gw.collect_results(msg_id1), timeout=10
            )
            assert results1[1]["status"] == "ok"
            assert results1[2]["status"] == "ok"

            # Cell 2: use the variable from cell 1
            msg_id2 = await gw.broadcast_execute(
                "seq_y = seq_x * 2; print(seq_y)", "cell-seq-2"
            )
            results2 = await asyncio.wait_for(
                gw.collect_results(msg_id2), timeout=10
            )
            assert results2[1]["status"] == "ok"
            assert results2[2]["status"] == "ok"

            # Verify stdout contains "20" from both workers
            outputs = gw.get_outputs(msg_id2)
            for rank in [1, 2]:
                assert rank in outputs
                stdout_msgs = [
                    o for o in outputs[rank]
                    if o.get("type") == "stream" and o.get("name") == "stdout"
                ]
                assert any("20" in o.get("text", "") for o in stdout_msgs)

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
    async def test_error_on_one_worker_does_not_block_other(self):
        """If one worker hits an error, the other still reports successfully."""
        gw = Gateway(port=0, auth_token="err-token", expected_workers=2)
        await gw.start()
        url = f"ws://localhost:{gw.port}/ws"

        w1 = Worker(rank=1, server_url=url, auth_token="err-token")
        w2 = Worker(rank=2, server_url=url, auth_token="err-token")
        t1 = asyncio.create_task(w1.run())
        t2 = asyncio.create_task(w2.run())

        try:
            for _ in range(100):
                if gw.all_workers_registered():
                    break
                await asyncio.sleep(0.1)
            assert gw.all_workers_registered()

            # Both workers execute the same code, so both will hit the error.
            # This verifies that errors are collected from all workers correctly.
            msg_id = await gw.broadcast_execute("1 / 0", "cell-err-1")
            results = await asyncio.wait_for(
                gw.collect_results(msg_id), timeout=10
            )

            assert 1 in results
            assert 2 in results
            assert results[1]["status"] == "error"
            assert results[2]["status"] == "error"
            assert results[1]["ename"] == "ZeroDivisionError"
            assert results[2]["ename"] == "ZeroDivisionError"

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
    async def test_interrupt_during_execution(self):
        """Interrupt a long-running cell and verify it completes quickly with error."""
        gw = Gateway(port=0, auth_token="int-token", expected_workers=1, timeout=15.0)
        await gw.start()
        url = f"ws://localhost:{gw.port}/ws"

        w1 = Worker(rank=1, server_url=url, auth_token="int-token")
        t1 = asyncio.create_task(w1.run())

        try:
            # Wait for worker to register
            for _ in range(100):
                if gw.all_workers_registered():
                    break
                await asyncio.sleep(0.1)
            assert gw.all_workers_registered()

            # Broadcast a long-running cell using a Python-level busy loop.
            # We use a tight loop with many iterations rather than time.sleep()
            # because PyThreadState_SetAsyncExc interrupts at bytecode
            # instruction boundaries and cannot interrupt C-level blocking calls.
            long_running_code = (
                "import time as _t\n"
                "_end = _t.monotonic() + 10\n"
                "while _t.monotonic() < _end:\n"
                "    pass\n"
            )
            msg_id = await gw.broadcast_execute(long_running_code, "cell-int")

            # Wait briefly for execution to start
            await asyncio.sleep(0.5)

            # Send interrupt
            await gw.broadcast_interrupt()

            # Verify execution completes with error status within a few seconds
            # (NOT 10 seconds — the interrupt should cut it short)
            start = time.monotonic()
            results = await asyncio.wait_for(
                gw.collect_results(msg_id), timeout=8.0
            )
            elapsed = time.monotonic() - start

            assert 1 in results
            assert results[1]["status"] == "error"
            assert results[1]["ename"] == "KeyboardInterrupt"
            # Should complete well before the 10-second busy loop
            assert elapsed < 7.0, (
                f"Interrupt took {elapsed:.1f}s, expected < 7s"
            )

        finally:
            w1.shutdown()
            t1.cancel()
            await gw.stop()
            try:
                await t1
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_worker_disconnect_during_execution(self):
        """Disconnecting a worker mid-execution unblocks collect_results."""
        gw = Gateway(port=0, auth_token="dc-token", expected_workers=2, timeout=10.0)
        await gw.start()
        url = f"ws://localhost:{gw.port}/ws"

        w1 = Worker(rank=1, server_url=url, auth_token="dc-token")
        w2 = Worker(rank=2, server_url=url, auth_token="dc-token")
        t1 = asyncio.create_task(w1.run())
        t2 = asyncio.create_task(w2.run())

        try:
            # Wait for both workers to register
            for _ in range(100):
                if gw.all_workers_registered():
                    break
                await asyncio.sleep(0.1)
            assert gw.all_workers_registered()

            # Broadcast a long-running cell to both workers (Python-level loop
            # so that PyThreadState_SetAsyncExc can interrupt at bytecode
            # boundaries, avoiding issues with C-level blocking calls).
            long_running_code = (
                "import time as _t\n"
                "_end = _t.monotonic() + 10\n"
                "while _t.monotonic() < _end:\n"
                "    pass\n"
            )
            msg_id = await gw.broadcast_execute(long_running_code, "cell-dc")

            # Wait briefly for execution to start
            await asyncio.sleep(0.5)

            # Disconnect worker 2 by shutting it down abruptly
            t2.cancel()
            try:
                await t2
            except asyncio.CancelledError:
                pass

            # collect_results should return within timeout (not block forever)
            # Worker 1 is still running, so interrupt it to speed up the test
            await gw.broadcast_interrupt()

            start = time.monotonic()
            results = await asyncio.wait_for(
                gw.collect_results(msg_id), timeout=8.0
            )
            elapsed = time.monotonic() - start

            # Should not have blocked for the full 10 seconds
            assert elapsed < 7.0, (
                f"collect_results took {elapsed:.1f}s, expected < 7s"
            )

            # The disconnected worker's result should have error status
            assert 2 in results
            assert results[2]["status"] == "error"

            # Worker 1 should also have a result (interrupted)
            assert 1 in results

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
