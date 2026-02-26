"""IPython magic commands for distributed SPMD execution control.

Provides ``%distributed`` line magic for enabling/disabling distributed
mode, querying worker status, and configuring execution parameters.
Also provides ``%%rank0`` and ``%%rank N`` cell magics for targeting
execution to specific ranks.
"""

import asyncio
import logging
from typing import Any

from IPython.core.magic import Magics, magics_class, line_magic, cell_magic

logger = logging.getLogger(__name__)

VALID_FAILURE_MODES = ("fail-fast", "best-effort")


@magics_class
class DistributedMagics(Magics):
    """Magic commands for controlling distributed SPMD execution.

    After registration the kernel sets ``self.kernel`` so that the
    magic commands can inspect and mutate execution state.
    """

    def __init__(self, shell: Any = None, **kwargs: Any) -> None:
        super().__init__(shell, **kwargs)
        self.kernel: Any = None  # Set by the kernel after initialization

    # ------------------------------------------------------------------
    # Line magic: %distributed
    # ------------------------------------------------------------------

    @line_magic
    def distributed(self, line: str) -> None:
        """Control distributed execution.

        Usage:
            %distributed on             - Enable distributed execution
            %distributed off            - Disable distributed execution
            %distributed status         - Show worker status
            %distributed expect N       - Set expected world size
            %distributed timeout N      - Set timeout in seconds
            %distributed failure-mode fail-fast|best-effort
            %distributed restart hard   - Kill all worker processes
        """
        args = line.strip().split()
        if not args:
            print(self.distributed.__doc__)
            return

        command = args[0]

        if self.kernel is None:
            print("Error: No kernel attached. Distributed magics are not available.")
            return

        dispatch = {
            "on": self._cmd_on,
            "off": self._cmd_off,
            "status": self._cmd_status,
            "expect": self._cmd_expect,
            "timeout": self._cmd_timeout,
            "failure-mode": self._cmd_failure_mode,
            "restart": self._cmd_restart,
        }

        handler = dispatch.get(command)
        if handler is None:
            print(
                f"Error: Unknown command '{command}'. "
                "Usage: %distributed on|off|status|expect|timeout|failure-mode|restart"
            )
            return

        handler(args[1:])

    def _cmd_on(self, args: list[str]) -> None:
        self.kernel.distributed_enabled = True
        print("Distributed execution enabled.")

    def _cmd_off(self, args: list[str]) -> None:
        self.kernel.distributed_enabled = False
        print("Distributed execution disabled.")

    def _cmd_status(self, args: list[str]) -> None:
        gateway = self.kernel._gateway
        enabled = self.kernel.distributed_enabled

        if gateway is None:
            print("Status: No gateway connected.")
            return

        state = "enabled" if enabled else "disabled"
        workers = gateway.workers
        expected = gateway.expected_workers

        print(f"Distributed mode: {state}")
        print(f"Workers: {len(workers)} connected, {expected} expected")

        for rank, worker in sorted(workers.items()):
            hostname = getattr(worker, "hostname", "unknown")
            gpu_id = getattr(worker, "gpu_id", -1)
            pid = getattr(worker, "pid", -1)
            worker_state = getattr(worker, "state", "unknown")
            print(
                f"  Rank {rank}: host={hostname}, gpu={gpu_id}, "
                f"pid={pid}, state={worker_state}"
            )

    def _cmd_expect(self, args: list[str]) -> None:
        if not args:
            print("Error: Expected an integer argument. Usage: %distributed expect N")
            return
        try:
            n = int(args[0])
        except ValueError:
            print(f"Error: Invalid integer '{args[0]}'. Usage: %distributed expect N")
            return

        self.kernel._gateway.expected_workers = n
        print(f"Expected workers set to {n}.")

    def _cmd_timeout(self, args: list[str]) -> None:
        if not args:
            print("Error: Expected a timeout value. Usage: %distributed timeout N")
            return
        try:
            t = float(args[0])
        except ValueError:
            print(f"Error: Invalid numeric value '{args[0]}'. Usage: %distributed timeout N")
            return

        self.kernel._gateway.timeout = t
        print(f"Timeout set to {t} seconds.")

    def _cmd_failure_mode(self, args: list[str]) -> None:
        if not args:
            print(
                "Error: Expected a failure mode. "
                "Usage: %distributed failure-mode fail-fast|best-effort"
            )
            return

        mode = args[0]
        if mode not in VALID_FAILURE_MODES:
            print(
                f"Error: Invalid failure mode '{mode}'. "
                f"Must be one of: {', '.join(VALID_FAILURE_MODES)}"
            )
            return

        self.kernel._gateway.failure_mode = mode
        print(f"Failure mode set to '{mode}'.")

    def _cmd_restart(self, args: list[str]) -> None:
        if not args or args[0] != "hard":
            print("Usage: %distributed restart hard")
            return

        gateway = self.kernel._gateway
        if gateway is None:
            print("Error: No gateway connected.")
            return

        # broadcast_shutdown is a coroutine — schedule as fire-and-forget.
        # We cannot block the event loop thread waiting for the coroutine
        # (that would deadlock), and we don't need the result.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(gateway.broadcast_shutdown())
        except RuntimeError:
            asyncio.run(gateway.broadcast_shutdown())

        print("Hard restart: shutdown broadcast sent to all workers.")

    # ------------------------------------------------------------------
    # Cell magic: %%rank0
    # ------------------------------------------------------------------

    @cell_magic
    def rank0(self, line: str, cell: str) -> None:
        """Execute cell only on rank-0 (skip broadcast to workers)."""
        self.shell.run_cell(cell)

    # ------------------------------------------------------------------
    # Cell magic: %%rank N
    # ------------------------------------------------------------------

    @cell_magic
    def rank(self, line: str, cell: str) -> None:
        """Execute cell only on specified rank.

        Usage:
            %%rank N
            <code>

        If N is 0, the cell is executed locally. Otherwise the cell
        code is sent to rank N via the gateway.
        """
        line = line.strip()
        if not line:
            print("Error: Expected a rank number. Usage: %%rank N")
            return

        try:
            target_rank = int(line)
        except ValueError:
            print(f"Error: Invalid rank '{line}'. Must be an integer.")
            return

        if target_rank == 0:
            self.shell.run_cell(cell)
            return

        gateway = self.kernel._gateway if self.kernel is not None else None
        if gateway is None:
            print("Error: No gateway connected.")
            return

        # Validate rank is connected before scheduling async send
        if target_rank not in gateway.workers:
            print(f"Error: Rank {target_rank} is not connected")
            return

        # Schedule fire-and-forget. We cannot block the event loop thread
        # waiting for the coroutine (that would deadlock). The actual output
        # will stream back through the gateway → kernel → frontend pipeline.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(gateway.send_to_rank(target_rank, cell, cell_id=""))
        except RuntimeError:
            asyncio.run(gateway.send_to_rank(target_rank, cell, cell_id=""))


def load_ipython_extension(ipython):
    """Register distributed magics when loaded via %load_ext."""
    magics = DistributedMagics(shell=ipython)
    ipython.register_magics(magics)
