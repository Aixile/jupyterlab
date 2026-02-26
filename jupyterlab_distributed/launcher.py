"""Launcher entry point for torchrun-compatible distributed SPMD execution.

This module is the main entry point invoked via ``torchrun -m
jupyterlab_distributed.launcher``.  It auto-detects the process rank
from standard environment variables (``RANK``, ``SLURM_PROCID``,
``OMPI_COMM_WORLD_RANK``) and dispatches to either the rank-0 kernel +
gateway path or the worker daemon path.
"""

import argparse
import asyncio
import logging
import os
import sys
import time

from .config import SessionConfig

logger = logging.getLogger(__name__)


def detect_rank() -> int:
    """Auto-detect rank from environment variables.

    Checks, in order: ``RANK`` (PyTorch / torchrun), ``SLURM_PROCID``
    (Slurm), ``OMPI_COMM_WORLD_RANK`` (Open MPI).  Returns 0 if none
    are set.
    """
    for var in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    return 0


def detect_world_size() -> int:
    """Auto-detect world size from environment variables.

    Checks, in order: ``WORLD_SIZE`` (PyTorch / torchrun),
    ``SLURM_NTASKS`` (Slurm), ``OMPI_COMM_WORLD_SIZE`` (Open MPI).
    Returns 1 if none are set.
    """
    for var in ("WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    return 1


def _run_rank0(config: SessionConfig, log_dir: str | None) -> None:
    """Start the Gateway and IPython kernel on rank-0.

    1. Starts the WebSocket gateway so workers can connect.
    2. Starts the IPython kernel (DistributedKernel) on the ZMQ ports
       specified in the session config.
    3. Updates the session config with actual host/port and marks it
       ``running`` so the provisioner can connect.
    """
    import socket as _socket
    import threading

    from .gateway import Gateway
    from .kernel import DistributedKernel

    # Use torchrun's WORLD_SIZE if available (overrides config)
    actual_world_size = detect_world_size()
    if actual_world_size > 1:
        config.world_size = actual_world_size
    expected_workers = config.world_size - 1  # rank-0 is the kernel

    logger.info(
        "Rank-0: starting gateway on port %d for kernel %s "
        "(world_size=%d, expected_workers=%d)",
        config.gateway_port,
        config.kernel_id,
        config.world_size,
        expected_workers,
    )

    # --- Start gateway in a background thread with its own event loop ---
    gateway = Gateway(
        port=config.gateway_port,
        auth_token=config.auth_token,
        expected_workers=expected_workers,
    )

    gateway_loop = asyncio.new_event_loop()

    def run_gateway():
        asyncio.set_event_loop(gateway_loop)
        gateway_loop.run_until_complete(gateway.start())
        logger.info("Rank-0: gateway started on port %d", gateway.port)
        gateway_loop.run_forever()

    gw_thread = threading.Thread(target=run_gateway, daemon=True)
    gw_thread.start()

    # Wait for gateway to be ready
    import time

    for _ in range(50):
        if gateway.port != 0:
            break
        time.sleep(0.1)

    # Update config so provisioner and workers can find us
    actual_host = _socket.gethostname()
    config.update(
        status="running",
        host=actual_host,
        gateway_port=gateway.port,
    )
    logger.info(
        "Rank-0: config updated — host=%s, gateway_port=%d, zmq_ports=%s",
        actual_host,
        gateway.port,
        config.zmq_ports,
    )

    # --- Start IPython kernel on the ZMQ ports from session config ---
    from ipykernel.kernelapp import IPKernelApp

    # Build argv for IPKernelApp with the specific ZMQ ports
    ports = config.zmq_ports
    kernel_argv = [
        "--IPKernelApp.transport=tcp",
        f"--IPKernelApp.ip={actual_host}",
        f"--IPKernelApp.shell_port={ports['shell']}",
        f"--IPKernelApp.iopub_port={ports['iopub']}",
        f"--IPKernelApp.stdin_port={ports['stdin']}",
        f"--IPKernelApp.control_port={ports['control']}",
        f"--IPKernelApp.hb_port={ports['hb']}",
        f"--Session.key={config.auth_token}",
        "--IPKernelApp.no_stdout=False",
        "--IPKernelApp.no_stderr=False",
    ]

    logger.info("Rank-0: starting DistributedKernel on ZMQ ports %s", ports)

    app = IPKernelApp.instance(kernel_class=DistributedKernel)
    app.initialize(kernel_argv)

    # Wire up the gateway to the kernel
    kernel = app.kernel
    kernel.set_gateway(gateway)
    kernel.distributed_enabled = True
    logger.info("Rank-0: kernel ready, distributed mode enabled")

    # Start the kernel (blocks forever)
    try:
        app.start()
    except KeyboardInterrupt:
        logger.info("Rank-0: interrupted, shutting down")
    finally:
        gateway_loop.call_soon_threadsafe(gateway_loop.stop)
        gw_thread.join(timeout=5)


def _run_worker(rank: int, config: SessionConfig, log_dir: str | None) -> None:
    """Wait for rank-0 to be running, then start the Worker daemon.

    Polls the session config file until its status becomes ``running``,
    then connects to rank-0's gateway as a Worker.
    """
    from .worker import Worker

    logger.info(
        "Rank-%d: waiting for rank-0 to start (config=%s)",
        rank,
        config.path,
    )

    # Poll until rank-0 marks the session as running
    timeout = 120.0
    poll_interval = 0.5
    start = time.monotonic()

    while time.monotonic() - start < timeout:
        try:
            refreshed = SessionConfig.load(config.path)
            if refreshed.status == "running":
                config = refreshed
                break
        except Exception:
            pass
        time.sleep(poll_interval)
    else:
        logger.error(
            "Rank-%d: timed out waiting for rank-0 after %.0fs",
            rank,
            timeout,
        )
        sys.exit(1)

    server_url = f"ws://{config.host or 'localhost'}:{config.gateway_port}/ws"
    logger.info("Rank-%d: connecting to gateway at %s", rank, server_url)

    worker = Worker(
        rank=rank,
        server_url=server_url,
        auth_token=config.auth_token,
        log_dir=log_dir,
    )
    asyncio.run(worker.run())


def main() -> None:
    """CLI entry point for the distributed launcher.

    Parses command-line arguments, detects the process rank (from
    ``--rank`` or environment variables), loads the session config,
    and dispatches to rank-0 or worker code paths.
    """
    parser = argparse.ArgumentParser(
        description="JupyterLab Distributed SPMD Launcher",
    )
    parser.add_argument(
        "--session-config",
        required=True,
        help="Path to session config JSON file",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Override rank (default: auto-detect from environment)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for log files",
    )

    args = parser.parse_args()

    # Detect rank: explicit flag overrides environment
    rank = args.rank if args.rank is not None else detect_rank()

    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [rank-{rank}] %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    logger.info("Launcher starting: rank=%d, config=%s", rank, args.session_config)

    # Load session config
    config = SessionConfig.load(args.session_config)

    # Dispatch based on rank
    if rank == 0:
        _run_rank0(config, args.log_dir)
    else:
        _run_worker(rank, config, args.log_dir)

if __name__ == "__main__":
    main()
