"""Kernel provisioner for distributed SPMD kernels.

Coordinates with a remote rank-0 process launched via torchrun by
monitoring a session config file on a shared filesystem.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from jupyter_client.provisioning import KernelProvisionerBase

from .config import SessionConfig

DEFAULT_CONFIG_DIR = os.environ.get(
    "JUPYTER_DISTRIBUTED_CONFIG_DIR",
    os.path.expanduser("~"),
)

# Default timeout waiting for the external kernel to start (seconds).
_DEFAULT_LAUNCH_TIMEOUT = 300.0

# Polling interval when waiting for config status changes (seconds).
_POLL_INTERVAL = 0.5


class DistributedProvisioner(KernelProvisionerBase):
    """Provisions a distributed kernel by coordinating with a remote
    rank-0 process launched via torchrun.

    Instead of launching a subprocess directly, this provisioner:
    1. Creates a session config on a shared filesystem (pre_launch).
    2. Waits for an external torchrun process to update the config
       status to "running" (launch_kernel).
    3. Returns ZMQ connection info so JupyterLab can communicate
       with the remote kernel.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config_base_dir: str = kwargs.get("config_base_dir", DEFAULT_CONFIG_DIR)
        self.world_size: int = kwargs.get("world_size", 1)
        self.gateway_port: int = kwargs.get("gateway_port", 9876)
        self._session_config: Optional[SessionConfig] = None
        self._process: object = None
        self._launch_timeout: float = _DEFAULT_LAUNCH_TIMEOUT

    # -- KernelProvisionerBase required properties --

    @property
    def has_process(self) -> bool:
        """Return True if a distributed kernel session is active."""
        return self._session_config is not None and self._session_config.status == "running"

    # -- Lifecycle methods --

    async def pre_launch(self, **kwargs: Any) -> Dict[str, Any]:
        """Create session config on the shared filesystem.

        This is called before launch_kernel and sets up the config file
        that the external torchrun workers will discover.
        """
        self._session_config = SessionConfig.create(
            base_dir=self.config_base_dir,
            kernel_id=self.kernel_id,
            world_size=self.world_size,
            gateway_port=self.gateway_port,
        )
        return kwargs

    async def launch_kernel(self, cmd: List[str], **kwargs: Any) -> Dict[str, Any]:
        """Wait for external torchrun to start the kernel.

        Polls the session config file until an external process updates
        the status to "running", then constructs connection_info from
        the ZMQ ports recorded in the config.

        Raises TimeoutError if the kernel does not start within the
        configured timeout.
        """
        if self._session_config is None:
            msg = "pre_launch must be called before launch_kernel"
            raise RuntimeError(msg)

        timeout = self._launch_timeout
        elapsed = 0.0

        while elapsed < timeout:
            # Re-read config from disk to pick up external updates
            config = SessionConfig.load(self._session_config.path)
            if config.status == "running":
                self._session_config = config
                self.connection_info = self._build_connection_info(config)
                return self.connection_info
            await asyncio.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL

        raise TimeoutError(
            f"Distributed kernel did not start within {timeout}s. "
            f"Ensure torchrun is running and can access "
            f"{self._session_config.path}"
        )

    async def poll(self) -> Optional[int]:
        """Check if kernel process is still running.

        Returns None if the kernel is alive (or we are still waiting
        for it to start). For distributed kernels, we treat the kernel
        as alive unless we detect a shutdown.
        """
        if self._session_config is not None and self._session_config.path.exists():
            try:
                config = SessionConfig.load(self._session_config.path)
                if config.status == "shutdown":
                    return 0
            except (OSError, ValueError):
                pass
        return None

    async def wait(self) -> Optional[int]:
        """Wait for kernel to finish.

        Polls the session config until the status is no longer 'running'.
        """
        if self._session_config is None:
            return 0

        while True:
            try:
                config = SessionConfig.load(self._session_config.path)
                if config.status != "running":
                    self._session_config = config
                    return 0
            except (OSError, ValueError):
                return 1
            await asyncio.sleep(_POLL_INTERVAL)

    async def send_signal(self, signum: int) -> None:
        """Send a signal to the kernel process.

        For distributed kernels, signals are not directly supported
        since the process runs remotely. Interrupts should go through
        the gateway's broadcast_interrupt mechanism instead.
        """
        return None

    async def kill(self, restart: bool = False) -> None:
        """Kill the kernel by updating config status to 'shutdown'.

        The remote workers monitor this status and will shut down
        when they detect the change.
        """
        if self._session_config is not None:
            self._session_config.update(status="shutdown")

    async def terminate(self, restart: bool = False) -> None:
        """Terminate the kernel gracefully.

        Delegates to kill() since distributed kernels use config-based
        signalling.
        """
        await self.kill(restart=restart)

    async def cleanup(self, restart: bool = False) -> None:
        """Clean up session config resources.

        Removes the session config file unless this is a restart, in
        which case the file is preserved for the next launch.
        """
        if self._session_config is not None and not restart:
            self._session_config.path.unlink(missing_ok=True)

    # -- Persistence methods --

    def get_provisioner_info(self) -> Dict[str, Any]:
        """Return provisioner state for persistence.

        Applications can use this to save and restore provisioner state
        for features like disaster recovery.
        """
        return {
            "config_path": str(self._session_config.path) if self._session_config else None,
            "world_size": self.world_size,
            "gateway_port": self.gateway_port,
        }

    def load_provisioner_info(self, info: Dict[str, Any]) -> None:
        """Restore provisioner state from previously saved info.

        Re-establishes the session config from the config_path if it
        still exists on disk.
        """
        config_path = info.get("config_path")
        self.world_size = info.get("world_size", self.world_size)
        self.gateway_port = info.get("gateway_port", self.gateway_port)

        if config_path is not None:
            path = Path(config_path)
            if path.exists():
                self._session_config = SessionConfig.load(path)

    # -- Internal helpers --

    @staticmethod
    def _build_connection_info(config: SessionConfig) -> Dict[str, Any]:
        """Build a KernelConnectionInfo dict from a SessionConfig."""
        ports = config.zmq_ports
        return {
            "ip": config.host or "127.0.0.1",
            "transport": "tcp",
            "shell_port": ports["shell"],
            "iopub_port": ports["iopub"],
            "stdin_port": ports["stdin"],
            "control_port": ports["control"],
            "hb_port": ports["hb"],
            "key": config.auth_token,
            "signature_scheme": "hmac-sha256",
        }
