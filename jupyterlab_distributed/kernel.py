"""Distributed IPython kernel with do_execute override.

This module provides a DistributedKernel that subclasses IPythonKernel
to broadcast cell code to all SPMD workers via a Gateway and collect
their results. Rank-0 executes locally while remote ranks execute in
parallel through WebSocket-connected worker processes.
"""

import logging
from typing import Any

from ipykernel.ipkernel import IPythonKernel

from .gateway import Gateway

logger = logging.getLogger(__name__)

DISTRIBUTED_MIME = "application/vnd.jupyterlab.distributed+json"


class DistributedKernel(IPythonKernel):
    """IPython kernel subclass that supports distributed SPMD execution.

    When ``distributed_enabled`` is True and all workers have registered
    through the gateway, each ``do_execute`` call broadcasts the cell
    code to every worker, executes locally on rank-0, and then collects
    and publishes per-rank results as a custom MIME display_data message.
    """

    implementation = "distributed-python"
    implementation_version = "0.1.0"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.distributed_enabled: bool = False
        self._gateway: Gateway | None = None
        self._register_magics()

    def _register_magics(self) -> None:
        """Register %distributed and %%rank magics in the shell."""
        try:
            from .magics import DistributedMagics
            magics = DistributedMagics(shell=self.shell)
            magics.kernel = self
            self.shell.register_magics(magics)
        except Exception:
            logger.warning("Failed to register distributed magics", exc_info=True)

    def set_gateway(self, gateway: Gateway) -> None:
        """Attach a Gateway instance used for worker communication."""
        self._gateway = gateway

    async def do_execute(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: dict[str, str] | None = None,
        allow_stdin: bool = False,
    ) -> dict[str, Any]:
        """Execute code, optionally broadcasting to all SPMD workers.

        Handles the ``%%rank0`` cell magic which forces local-only
        execution even when distributed mode is active.
        """
        # Check for %%rank0 magic -- execute only on rank-0
        stripped = code.strip()
        if stripped.startswith("%%rank0"):
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
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: dict[str, str] | None = None,
        allow_stdin: bool = False,
    ) -> dict[str, Any]:
        """Delegate to the base IPythonKernel.do_execute."""
        return await super().do_execute(
            code, silent, store_history, user_expressions, allow_stdin
        )

    async def _distributed_execute(
        self,
        code: str,
        silent: bool,
        store_history: bool,
        user_expressions: dict[str, str] | None,
        allow_stdin: bool,
    ) -> dict[str, Any]:
        """Broadcast code to workers, execute locally, and collect results.

        Steps:
        1. Broadcast the cell code to all connected workers via the gateway.
        2. Execute the code locally on rank-0 through the parent do_execute.
        3. Collect execution results from all workers.
        4. Publish per-rank outputs as a custom display_data message.
        5. Log warnings for any failed ranks.
        """
        # 1. Broadcast to workers
        msg_id = await self._gateway.broadcast_execute(code, "")

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

        # 5. Check for failures and log warnings
        failed_ranks = [
            r for r, res in worker_results.items()
            if res.get("status") != "ok"
        ]
        if failed_ranks and result.get("status") == "ok":
            logger.warning("Ranks %s failed", failed_ranks)

        return result

    def _publish_distributed_outputs(self, worker_results: dict[int, dict[str, Any]]) -> None:
        """Publish a display_data message with per-rank execution outputs.

        Each rank's data includes its outputs list, status string, and
        execution_time in seconds. The message uses the custom MIME type
        ``application/vnd.jupyterlab.distributed+json`` so the frontend
        renderer can display a collapsible per-rank output view.
        """
        if not worker_results:
            return

        ranks_data: dict[str, dict[str, Any]] = {}
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
