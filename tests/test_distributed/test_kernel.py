"""Tests for the DistributedKernel with do_execute override."""

import asyncio
import json

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from jupyterlab_distributed.kernel import DistributedKernel, DISTRIBUTED_MIME


def _make_kernel():
    """Create a DistributedKernel instance without calling __init__."""
    with patch.object(DistributedKernel, "__init__", lambda self, **kw: None):
        k = DistributedKernel.__new__(DistributedKernel)
        k.distributed_enabled = False
        k._gateway = None
        return k


class TestDistributedKernelInit:
    def test_kernel_has_distributed_flag(self):
        """A freshly constructed kernel has distributed_enabled=False."""
        k = _make_kernel()
        assert not k.distributed_enabled

    def test_kernel_has_no_gateway(self):
        """A freshly constructed kernel has _gateway=None."""
        k = _make_kernel()
        assert k._gateway is None

    def test_set_gateway(self):
        """set_gateway stores the gateway reference."""
        k = _make_kernel()
        gw = MagicMock()
        k.set_gateway(gw)
        assert k._gateway is gw

    def test_implementation_name(self):
        """Kernel reports correct implementation name."""
        assert DistributedKernel.implementation == "distributed-python"

    def test_implementation_version(self):
        """Kernel reports correct implementation version."""
        assert DistributedKernel.implementation_version == "0.1.0"


class TestDoExecuteNonDistributed:
    @pytest.mark.asyncio
    async def test_do_execute_without_distributed_calls_parent(self):
        """When distributed is disabled, do_execute delegates to parent."""
        k = _make_kernel()
        k._parent_do_execute = AsyncMock(return_value={
            "status": "ok",
            "execution_count": 1,
            "payload": [],
            "user_expressions": {},
        })
        result = await k.do_execute("x = 1", False)
        k._parent_do_execute.assert_called_once_with(
            "x = 1", False, True, None, False
        )
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_do_execute_no_gateway_calls_parent(self):
        """When distributed is enabled but no gateway, delegates to parent."""
        k = _make_kernel()
        k.distributed_enabled = True
        k._gateway = None
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
    async def test_do_execute_workers_not_registered_calls_parent(self):
        """When distributed is enabled but workers not registered, delegates to parent."""
        k = _make_kernel()
        k.distributed_enabled = True
        k._gateway = MagicMock()
        k._gateway.all_workers_registered.return_value = False
        k._parent_do_execute = AsyncMock(return_value={
            "status": "ok",
            "execution_count": 1,
            "payload": [],
            "user_expressions": {},
        })
        result = await k.do_execute("x = 1", False)
        k._parent_do_execute.assert_called_once()
        assert result["status"] == "ok"


class TestDoExecuteDistributed:
    def _make_distributed_kernel(self):
        """Create a kernel configured for distributed execution."""
        k = _make_kernel()
        k.distributed_enabled = True
        k._gateway = MagicMock()
        k._gateway.all_workers_registered.return_value = True
        k._gateway.broadcast_execute = AsyncMock(return_value="msg-001")
        k._gateway.collect_results = AsyncMock(return_value={
            1: {"status": "ok", "outputs": [], "execution_time": 0.5},
            2: {"status": "ok", "outputs": [], "execution_time": 0.3},
        })
        k._parent_do_execute = AsyncMock(return_value={
            "status": "ok",
            "execution_count": 1,
            "payload": [],
            "user_expressions": {},
        })
        k.send_response = MagicMock()
        k.iopub_socket = MagicMock()
        return k

    @pytest.mark.asyncio
    async def test_distributed_execute_broadcasts(self):
        """When distributed, do_execute calls broadcast_execute."""
        k = self._make_distributed_kernel()
        await k.do_execute("x = 1", False)
        k._gateway.broadcast_execute.assert_called_once_with("x = 1", "")

    @pytest.mark.asyncio
    async def test_distributed_execute_collects_results(self):
        """When distributed, do_execute calls collect_results with the msg_id."""
        k = self._make_distributed_kernel()
        await k.do_execute("x = 1", False)
        k._gateway.collect_results.assert_called_once_with("msg-001")

    @pytest.mark.asyncio
    async def test_distributed_execute_publishes_outputs(self):
        """When distributed, do_execute publishes distributed display_data."""
        k = self._make_distributed_kernel()
        await k.do_execute("x = 1", False)
        k.send_response.assert_called_once()
        call_args = k.send_response.call_args
        assert call_args[0][1] == "display_data"
        content = call_args[0][2]
        assert DISTRIBUTED_MIME in content["data"]
        dist_data = content["data"][DISTRIBUTED_MIME]
        assert dist_data["type"] == "rank_outputs"
        assert "1" in dist_data["ranks"]
        assert "2" in dist_data["ranks"]

    @pytest.mark.asyncio
    async def test_distributed_execute_returns_local_result(self):
        """The return value is the local rank-0 execution result."""
        k = self._make_distributed_kernel()
        result = await k.do_execute("x = 1", False)
        assert result["status"] == "ok"
        assert result["execution_count"] == 1

    @pytest.mark.asyncio
    async def test_distributed_execute_also_runs_locally(self):
        """Distributed execute also calls _parent_do_execute for rank-0."""
        k = self._make_distributed_kernel()
        await k.do_execute("x = 1", False)
        k._parent_do_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_distributed_execute_handles_collect_error(self):
        """If collect_results raises, execution still returns local result."""
        k = self._make_distributed_kernel()
        k._gateway.collect_results = AsyncMock(
            side_effect=asyncio.TimeoutError("timed out")
        )
        result = await k.do_execute("x = 1", False)
        assert result["status"] == "ok"
        # send_response should not be called since worker_results is empty
        k.send_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_distributed_execute_empty_results_no_publish(self):
        """If no worker results, _publish_distributed_outputs does nothing."""
        k = self._make_distributed_kernel()
        k._gateway.collect_results = AsyncMock(return_value={})
        await k.do_execute("x = 1", False)
        k.send_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_distributed_execute_failed_ranks_logged(self):
        """If some ranks fail, a warning is logged but result is still returned."""
        k = self._make_distributed_kernel()
        k._gateway.collect_results = AsyncMock(return_value={
            1: {"status": "error", "outputs": [], "execution_time": 0.1},
        })
        with patch("jupyterlab_distributed.kernel.logger") as mock_logger:
            result = await k.do_execute("x = 1", False)
            mock_logger.warning.assert_called_once()
            assert "1" in str(mock_logger.warning.call_args)
        assert result["status"] == "ok"


class TestRank0Magic:
    @pytest.mark.asyncio
    async def test_rank0_magic_skips_broadcast(self):
        """%%rank0 magic causes code to be executed only locally."""
        k = _make_kernel()
        k.distributed_enabled = True
        k._gateway = MagicMock()
        k._gateway.all_workers_registered.return_value = True
        k._gateway.broadcast_execute = AsyncMock(return_value="msg-001")
        k._parent_do_execute = AsyncMock(return_value={
            "status": "ok",
            "execution_count": 1,
            "payload": [],
            "user_expressions": {},
        })
        code = "%%rank0\nimport os\nprint(os.getcwd())"
        result = await k.do_execute(code, False)
        # broadcast should NOT have been called
        k._gateway.broadcast_execute.assert_not_called()
        # parent should have been called with the code minus the magic line
        k._parent_do_execute.assert_called_once()
        call_code = k._parent_do_execute.call_args[0][0]
        assert call_code == "import os\nprint(os.getcwd())"
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_rank0_magic_with_leading_whitespace(self):
        """%%rank0 with leading whitespace is still detected."""
        k = _make_kernel()
        k.distributed_enabled = True
        k._gateway = MagicMock()
        k._gateway.all_workers_registered.return_value = True
        k._gateway.broadcast_execute = AsyncMock()
        k._parent_do_execute = AsyncMock(return_value={
            "status": "ok",
            "execution_count": 1,
            "payload": [],
            "user_expressions": {},
        })
        code = "  %%rank0\nx = 1"
        result = await k.do_execute(code, False)
        k._gateway.broadcast_execute.assert_not_called()
        assert result["status"] == "ok"


class TestPublishDistributedOutputs:
    def test_publish_with_results(self):
        """_publish_distributed_outputs sends display_data with rank info."""
        k = _make_kernel()
        k.send_response = MagicMock()
        k.iopub_socket = MagicMock()

        worker_results = {
            0: {"status": "ok", "outputs": [{"text": "hello"}], "execution_time": 1.0},
            1: {"status": "ok", "outputs": [], "execution_time": 0.5},
        }
        k._publish_distributed_outputs(worker_results)

        k.send_response.assert_called_once()
        call_args = k.send_response.call_args
        content = call_args[0][2]
        assert DISTRIBUTED_MIME in content["data"]
        ranks = content["data"][DISTRIBUTED_MIME]["ranks"]
        assert ranks["0"]["status"] == "ok"
        assert ranks["0"]["execution_time"] == 1.0
        assert ranks["1"]["outputs"] == []
        assert "text/plain" in content["data"]

    def test_publish_empty_results_is_noop(self):
        """_publish_distributed_outputs with empty dict does nothing."""
        k = _make_kernel()
        k.send_response = MagicMock()
        k._publish_distributed_outputs({})
        k.send_response.assert_not_called()

    def test_publish_handles_missing_fields(self):
        """_publish_distributed_outputs handles results with missing keys."""
        k = _make_kernel()
        k.send_response = MagicMock()
        k.iopub_socket = MagicMock()

        worker_results = {
            0: {},  # missing all optional fields
        }
        k._publish_distributed_outputs(worker_results)

        k.send_response.assert_called_once()
        content = k.send_response.call_args[0][2]
        ranks = content["data"][DISTRIBUTED_MIME]["ranks"]
        assert ranks["0"]["status"] == "unknown"
        assert ranks["0"]["execution_time"] == 0
        assert ranks["0"]["outputs"] == []
