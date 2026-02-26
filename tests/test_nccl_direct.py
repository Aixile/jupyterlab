"""Direct NCCL test: launch torchrun, connect to kernel ZMQ ports, execute cells.

Bypasses JupyterLab entirely — connects directly to the rank-0 kernel
via ZMQ (like jupyter console --existing).
"""
import asyncio
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/home/jinyh/jupyterlab")

PYTHON = "/home/jinyh/miniconda3/envs/test_distributed_jupyterlab/bin/python"
TORCHRUN = "/home/jinyh/miniconda3/envs/test_distributed_jupyterlab/bin/torchrun"
REPO = "/home/jinyh/jupyterlab"


def main():
    from jupyterlab_distributed.config import SessionConfig

    # Create session config
    # world_size must match torchrun's --nproc_per_node (total processes)
    # rank-0 = kernel, ranks 1-(N-1) = workers, so expected_workers = N-1
    config = SessionConfig.create(
        base_dir=os.path.expanduser("~"),
        kernel_id="nccl-direct-test",
        world_size=4,  # matches --nproc_per_node 4
        gateway_port=0,
    )
    print(f"Session config: {config.path}")
    print(f"ZMQ ports: {config.zmq_ports}")

    # Launch torchrun with 4 processes (rank-0 = kernel, ranks 1-3 = workers)
    print("\nLaunching torchrun with 4 processes...")
    env = {**os.environ, "PYTHONPATH": REPO}
    torchrun = subprocess.Popen(
        [TORCHRUN, "--nnodes", "1", "--nproc_per_node", "4",
         "--standalone",
         "-m", "jupyterlab_distributed.launcher",
         "--session-config", str(config.path)],
        env=env,
    )

    # Wait for config to become "running"
    print("Waiting for rank-0 to start...")
    for _ in range(60):
        try:
            c = SessionConfig.load(config.path)
            if c.status == "running":
                config = c
                break
        except:
            pass
        time.sleep(1)
    else:
        torchrun.terminate()
        raise RuntimeError("Rank-0 never started")

    print(f"Rank-0 running on {config.host}:{config.gateway_port}")
    print(f"ZMQ ports: {config.zmq_ports}")
    time.sleep(3)  # Let workers connect

    # Connect to the kernel via jupyter_client
    from jupyter_client import BlockingKernelClient

    kc = BlockingKernelClient()
    kc.transport = "tcp"
    kc.ip = config.host
    kc.shell_port = config.zmq_ports["shell"]
    kc.iopub_port = config.zmq_ports["iopub"]
    kc.stdin_port = config.zmq_ports["stdin"]
    kc.control_port = config.zmq_ports["control"]
    kc.hb_port = config.zmq_ports["hb"]
    kc.session.key = config.auth_token.encode()
    kc.start_channels()

    # Wait for kernel to be ready
    print("Connecting to kernel...")
    try:
        kc.wait_for_ready(timeout=30)
        print("Kernel ready!")
    except Exception as e:
        print(f"Kernel not ready: {e}")
        torchrun.terminate()
        return

    def run_cell(code, timeout=60):
        """Execute code and collect output."""
        msg_id = kc.execute(code)
        outputs = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                msg = kc.get_iopub_msg(timeout=min(5, deadline - time.monotonic()))
            except:
                break
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            mt = msg["msg_type"]
            if mt == "stream":
                outputs.append(msg["content"]["text"])
            elif mt == "error":
                outputs.append(f"ERROR: {msg['content']['ename']}: {msg['content']['evalue']}\n")
                for tb in msg["content"].get("traceback", []):
                    outputs.append(tb + "\n")
            elif mt == "display_data":
                data = msg["content"].get("data", {})
                dist = data.get("application/vnd.jupyterlab.distributed+json")
                if dist and dist.get("type") == "rank_outputs":
                    for rank_str in sorted(dist["ranks"].keys(), key=int):
                        rank_data = dist["ranks"][rank_str]
                        for o in rank_data.get("outputs", []):
                            if o.get("type") == "stream":
                                outputs.append(f"[rank {rank_str}] {o.get('text', '')}")
                            elif o.get("type") == "error":
                                outputs.append(f"[rank {rank_str}] ERROR: {o.get('ename')}: {o.get('evalue')}\n")
                elif "text/plain" in data:
                    outputs.append(data["text/plain"] + "\n")
            elif mt == "status" and msg["content"]["execution_state"] == "idle":
                break
        return "".join(outputs)

    try:
        # === Cell 1: Check distributed status ===
        print("\n=== Cell 1: %distributed status ===")
        out = run_cell("%distributed status")
        print(out)

        # === Cell 2: Init torch.distributed ===
        print("\n=== Cell 2: Init torch.distributed with NCCL ===")
        out = run_cell("""
import os, torch, torch.distributed as dist
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
torch.cuda.set_device(local_rank)
if not dist.is_initialized():
    dist.init_process_group(backend='nccl')
rank = dist.get_rank()
ws = dist.get_world_size()
print(f"rank={rank}, world_size={ws}, device=cuda:{local_rank}, gpu={torch.cuda.get_device_name(local_rank)}")
""", timeout=30)
        print(out)

        # === Cell 3: NCCL allreduce ===
        print("\n=== Cell 3: NCCL allreduce ===")
        out = run_cell("""
import torch, torch.distributed as dist
rank = dist.get_rank()
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
t = torch.ones(4, device=f'cuda:{local_rank}') * (rank + 1)
print(f"rank={rank}: before allreduce: {t.tolist()}")
dist.all_reduce(t, op=dist.ReduceOp.SUM)
expected = float(sum(range(1, dist.get_world_size() + 1)))
print(f"rank={rank}: after allreduce: {t.tolist()} (expected={expected})")
assert t[0].item() == expected, f"FAIL: expected {expected}, got {t[0].item()}"
print(f"rank={rank}: NCCL allreduce PASSED!")
""", timeout=30)
        print(out)

        # === Cell 4: NCCL allgather ===
        print("\n=== Cell 4: NCCL allgather ===")
        out = run_cell("""
import torch, torch.distributed as dist
rank = dist.get_rank()
ws = dist.get_world_size()
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
tensor = torch.tensor([rank * 10 + 1, rank * 10 + 2], device=f'cuda:{local_rank}', dtype=torch.float32)
gathered = [torch.zeros(2, device=f'cuda:{local_rank}') for _ in range(ws)]
dist.all_gather(gathered, tensor)
result = [g.tolist() for g in gathered]
print(f"rank={rank}: input={tensor.tolist()}, allgather={result}")
print(f"rank={rank}: NCCL allgather PASSED!")
""", timeout=30)
        print(out)

        print("\n=== ALL NCCL TESTS COMPLETE ===")

    finally:
        kc.stop_channels()
        torchrun.terminate()
        torchrun.wait(timeout=10)
        config.path.unlink(missing_ok=True)
        print("Cleanup done")


if __name__ == "__main__":
    main()
