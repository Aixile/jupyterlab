"""Tests for the DistributedProvisioner."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jupyterlab_distributed.config import SessionConfig
from jupyterlab_distributed.provisioner import DEFAULT_CONFIG_DIR, DistributedProvisioner


def _make_provisioner(tmp_path, kernel_id="test-123", world_size=4, gateway_port=9876):
    """Create a DistributedProvisioner with test-friendly defaults.

    Uses __new__ to avoid KernelProvisionerBase.__init__ which requires
    a full kernel_manager context.
    """
    prov = DistributedProvisioner.__new__(DistributedProvisioner)
    prov.kernel_id = kernel_id
    prov.config_base_dir = str(tmp_path)
    prov.world_size = world_size
    prov.gateway_port = gateway_port
    prov._session_config = None
    prov._process = None
    prov._launch_timeout = 300.0
    prov.connection_info = {}
    return prov


class TestDistributedProvisioner:
    def test_default_config_dir_is_set(self):
        """DEFAULT_CONFIG_DIR should be a sensible default string."""
        assert isinstance(DEFAULT_CONFIG_DIR, str)
        assert len(DEFAULT_CONFIG_DIR) > 0

    @pytest.mark.asyncio
    async def test_pre_launch_creates_config(self, tmp_path):
        """pre_launch should create a SessionConfig file on the filesystem."""
        prov = _make_provisioner(tmp_path)
        kwargs = await prov.pre_launch(cmd=["python", "-m", "ipykernel"])
        assert prov._session_config is not None
        assert prov._session_config.path.exists()
        # Verify the config has the right values
        data = json.loads(prov._session_config.path.read_text())
        assert data["kernel_id"] == "test-123"
        assert data["world_size"] == 4
        assert data["gateway_port"] == 9876
        assert data["status"] == "waiting"

    @pytest.mark.asyncio
    async def test_pre_launch_returns_kwargs(self, tmp_path):
        """pre_launch should return a kwargs dict (possibly updated)."""
        prov = _make_provisioner(tmp_path)
        kwargs = await prov.pre_launch(cmd=["python", "-m", "ipykernel"])
        assert isinstance(kwargs, dict)

    @pytest.mark.asyncio
    async def test_poll_returns_none_when_alive(self, tmp_path):
        """poll() should return None when the kernel is considered alive."""
        prov = _make_provisioner(tmp_path)
        result = await prov.poll()
        assert result is None

    @pytest.mark.asyncio
    async def test_send_signal_returns_none(self, tmp_path):
        """send_signal should not raise and should return None."""
        prov = _make_provisioner(tmp_path)
        result = await prov.send_signal(15)
        assert result is None

    @pytest.mark.asyncio
    async def test_kill_updates_status_to_shutdown(self, tmp_path):
        """kill() should update the session config status to 'shutdown'."""
        prov = _make_provisioner(tmp_path)
        await prov.pre_launch(cmd=["python"])
        await prov.kill(restart=False)
        # Re-read the config file to verify status was updated
        reloaded = SessionConfig.load(prov._session_config.path)
        assert reloaded.status == "shutdown"

    @pytest.mark.asyncio
    async def test_kill_without_config_is_safe(self, tmp_path):
        """kill() should not raise when session config is None."""
        prov = _make_provisioner(tmp_path)
        # _session_config is None; should not raise
        await prov.kill(restart=False)

    @pytest.mark.asyncio
    async def test_terminate_delegates_to_kill(self, tmp_path):
        """terminate() should call kill() and update status."""
        prov = _make_provisioner(tmp_path)
        await prov.pre_launch(cmd=["python"])
        await prov.terminate(restart=False)
        reloaded = SessionConfig.load(prov._session_config.path)
        assert reloaded.status == "shutdown"

    @pytest.mark.asyncio
    async def test_cleanup_removes_config_file(self, tmp_path):
        """cleanup(restart=False) should remove the session config file."""
        prov = _make_provisioner(tmp_path)
        await prov.pre_launch(cmd=["python"])
        config_path = prov._session_config.path
        assert config_path.exists()
        await prov.cleanup(restart=False)
        assert not config_path.exists()

    @pytest.mark.asyncio
    async def test_cleanup_preserves_config_on_restart(self, tmp_path):
        """cleanup(restart=True) should keep the session config file."""
        prov = _make_provisioner(tmp_path)
        await prov.pre_launch(cmd=["python"])
        config_path = prov._session_config.path
        assert config_path.exists()
        await prov.cleanup(restart=True)
        assert config_path.exists()

    @pytest.mark.asyncio
    async def test_cleanup_without_config_is_safe(self, tmp_path):
        """cleanup() should not raise when session config is None."""
        prov = _make_provisioner(tmp_path)
        await prov.cleanup(restart=False)

    def test_has_process_false_by_default(self, tmp_path):
        """has_process should be False when no process has been set."""
        prov = _make_provisioner(tmp_path)
        assert prov.has_process is False

    def test_get_provisioner_info_with_config(self, tmp_path):
        """get_provisioner_info should include config_path when config exists."""
        prov = _make_provisioner(tmp_path)
        # Manually create a session config
        prov._session_config = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-123",
            world_size=4,
            gateway_port=9876,
        )
        info = prov.get_provisioner_info()
        assert "config_path" in info
        assert info["config_path"] == str(prov._session_config.path)
        assert info["world_size"] == 4
        assert info["gateway_port"] == 9876

    def test_get_provisioner_info_without_config(self, tmp_path):
        """get_provisioner_info should return None config_path when no config."""
        prov = _make_provisioner(tmp_path)
        info = prov.get_provisioner_info()
        assert info["config_path"] is None

    def test_load_provisioner_info_restores_config(self, tmp_path):
        """load_provisioner_info should restore session config from config_path."""
        # First create a config on disk
        config = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-456",
            world_size=8,
            gateway_port=9877,
        )
        # Now create a fresh provisioner and load from info
        prov = _make_provisioner(tmp_path, kernel_id="test-456", world_size=8, gateway_port=9877)
        prov.load_provisioner_info({
            "config_path": str(config.path),
            "world_size": 8,
            "gateway_port": 9877,
        })
        assert prov._session_config is not None
        assert prov._session_config.kernel_id == "test-456"
        assert prov.world_size == 8
        assert prov.gateway_port == 9877

    def test_load_provisioner_info_handles_missing_path(self, tmp_path):
        """load_provisioner_info should handle None config_path gracefully."""
        prov = _make_provisioner(tmp_path)
        prov.load_provisioner_info({
            "config_path": None,
            "world_size": 2,
            "gateway_port": 9876,
        })
        assert prov._session_config is None

    @pytest.mark.asyncio
    async def test_launch_kernel_waits_for_running_status(self, tmp_path):
        """launch_kernel should poll session config until status is 'running'."""
        prov = _make_provisioner(tmp_path)
        await prov.pre_launch(cmd=["python"])

        # Simulate an external process updating the config to 'running'
        # after a short delay
        async def update_config_later():
            await asyncio.sleep(0.1)
            prov._session_config.update(
                status="running",
                host="compute-01",
            )

        asyncio.create_task(update_config_later())

        conn_info = await prov.launch_kernel(cmd=["python", "-m", "ipykernel"])
        assert isinstance(conn_info, dict)
        # Connection info should contain the ZMQ port channels
        assert "shell_port" in conn_info
        assert "iopub_port" in conn_info
        assert "stdin_port" in conn_info
        assert "control_port" in conn_info
        assert "hb_port" in conn_info
        assert conn_info["ip"] == "compute-01"

    @pytest.mark.asyncio
    async def test_launch_kernel_timeout(self, tmp_path):
        """launch_kernel should raise TimeoutError if config never reaches 'running'."""
        prov = _make_provisioner(tmp_path)
        prov._launch_timeout = 0.3  # Short timeout for testing
        await prov.pre_launch(cmd=["python"])

        with pytest.raises(TimeoutError, match="kernel.*start"):
            await prov.launch_kernel(cmd=["python", "-m", "ipykernel"])

    @pytest.mark.asyncio
    async def test_wait_polls_until_not_running(self, tmp_path):
        """wait() should poll until config status is no longer 'running'."""
        prov = _make_provisioner(tmp_path)
        await prov.pre_launch(cmd=["python"])
        prov._session_config.update(status="running")

        async def shutdown_later():
            await asyncio.sleep(0.1)
            prov._session_config.update(status="shutdown")

        asyncio.create_task(shutdown_later())

        result = await prov.wait()
        assert result == 0
