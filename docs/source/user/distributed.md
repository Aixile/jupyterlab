# Distributed SPMD Cell Execution

Run notebook cells interactively across multiple machines in SPMD (Single Program Multiple Data) fashion. Designed for PyTorch DDP training on SLURM HPC clusters.

## How It Works

```
JupyterLab (head node)
    │
    ▼
Rank-0 Kernel (compute node, via torchrun)
    ├── Real IPython kernel (autocomplete, inspect work)
    ├── WebSocket gateway (coordinates workers)
    ├── torch.distributed rank 0
    │
    ├──► Worker rank 1 (embedded IPython shell)
    ├──► Worker rank 2
    └──► Worker rank N
```

When you run a cell, it executes on **all ranks simultaneously**. Rank-0 is the real Jupyter kernel — autocomplete, variable inspector, and inspection all show rank-0's live objects (real DDP models, real distributed tensors).

## Installation

### Quick Install (recommended for testing)

Uses the official pre-built JupyterLab frontend assets with your fork's Python code. The distributed backend (gateway, worker, kernel, magics) all work. The only things skipped are the custom MIME renderer and sidebar panel (not wired up yet).

```bash
# Create a fresh conda env
conda create -n distributed-jlab python=3.10 -y
conda activate distributed-jlab

# Install pre-built JupyterLab + dependencies
pip install jupyterlab websockets
pip install torch --index-url https://download.pytorch.org/whl/cu128  # or cpu

# Clone and install the fork (Python code only)
git clone https://github.com/Aixile/jupyterlab.git
cd jupyterlab
pip install -e . --no-deps --no-build-isolation

# Start from any directory
jupyter lab --ip=0.0.0.0
```

### Full Build from Source (includes custom frontend)

Builds the complete JupyterLab frontend including the distributed extension packages. Takes ~10 minutes the first time. Requires Node.js 20+.

```bash
conda create -n distributed-jlab python=3.10 -y
conda activate distributed-jlab

git clone https://github.com/Aixile/jupyterlab.git
cd jupyterlab

# Install Python package in dev mode
pip install -e ".[dev]"
pip install websockets torch --index-url https://download.pytorch.org/whl/cu128

# Install JS dependencies and build frontend
jlpm install          # ~2 min
jlpm run build        # ~5-10 min
jupyter lab build     # ~2 min

# Start from any directory
jupyter lab --ip=0.0.0.0
```

> **Note:** After the initial build, you only need to re-run `jlpm run build`
> if you change TypeScript files. Python changes take effect immediately
> with the `-e` (editable) install.

### Troubleshooting Install

**"JupyterLab application assets not found"** — You used `pip install -e .`
without building the frontend. Either run `jlpm install && jlpm run build && jupyter lab build`, or use the Quick Install method above.

**"No module named 'jupyterlab_distributed'"** — Add the repo to your
PYTHONPATH: `PYTHONPATH=/path/to/jupyterlab jupyter lab`

---

## Single-Node Setup (Development / Testing)

Test distributed execution on a single machine without SLURM or torchrun. Useful for development, debugging, and verifying your code before scaling to a cluster.

### 2. Option A: Script-based (no JupyterLab UI needed)

The simplest way to test distributed execution — runs entirely from the command line:

```python
# test_local.py
import asyncio, sys
sys.path.insert(0, '/path/to/jupyterlab')

from jupyterlab_distributed.gateway import Gateway
from jupyterlab_distributed.worker import Worker

async def main():
    # Start gateway
    gw = Gateway(port=0, auth_token='test', expected_workers=2)
    await gw.start()
    url = f'ws://localhost:{gw.port}/ws'
    print(f'Gateway on port {gw.port}')

    # Start 2 workers
    w1 = Worker(rank=1, server_url=url, auth_token='test')
    w2 = Worker(rank=2, server_url=url, auth_token='test')
    t1 = asyncio.create_task(w1.run())
    t2 = asyncio.create_task(w2.run())

    # Wait for registration
    while not gw.all_workers_registered():
        await asyncio.sleep(0.1)
    print(f'Workers: {sorted(gw.workers.keys())}')

    # Execute on all workers
    msg_id = await gw.broadcast_execute(
        "import os; print(f'Hello from PID {os.getpid()}')", 'cell-1'
    )
    results = await gw.collect_results(msg_id)
    for rank in sorted(results.keys()):
        r = results[rank]
        outs = [o['text'] for o in r.get('outputs', [])
                if o.get('type') == 'stream']
        print(f"Rank {rank}: {''.join(outs).strip()}")

    # Cleanup
    await gw.broadcast_shutdown()
    await gw.stop()

asyncio.run(main())
```

```bash
PYTHONPATH=/path/to/jupyterlab python test_local.py
```

### 3. Option B: Subprocess workers (closer to production)

Launch workers as real separate processes, same as torchrun would:

```python
# test_subprocess.py
import asyncio, json, os, subprocess, sys
sys.path.insert(0, '/path/to/jupyterlab')

from jupyterlab_distributed.gateway import Gateway

PYTHON = sys.executable  # or path to your conda env python

WORKER_SCRIPT = '''
import asyncio, sys
sys.path.insert(0, '/path/to/jupyterlab')
from jupyterlab_distributed.worker import Worker
w = Worker(rank=int(sys.argv[1]), server_url=sys.argv[2], auth_token=sys.argv[3])
asyncio.run(w.run())
'''

async def main():
    gw = Gateway(port=0, auth_token='test', expected_workers=4)
    await gw.start()
    url = f'ws://localhost:{gw.port}/ws'
    print(f'Gateway on port {gw.port}')

    # Launch 4 worker processes (simulating 4 GPUs)
    procs = []
    for rank in range(1, 5):
        p = subprocess.Popen(
            [PYTHON, '-c', WORKER_SCRIPT, str(rank), url, 'test'],
            env={**os.environ, 'PYTHONPATH': '/path/to/jupyterlab'},
        )
        procs.append(p)

    while not gw.all_workers_registered():
        await asyncio.sleep(0.1)
    print(f'Workers: {sorted(gw.workers.keys())}')

    # Broadcast torch import
    msg_id = await gw.broadcast_execute(
        "import torch; print(f'torch {torch.__version__}, CUDA={torch.cuda.is_available()}')",
        'torch-test'
    )
    results = await gw.collect_results(msg_id)
    for rank in sorted(results.keys()):
        r = results[rank]
        outs = [o['text'] for o in r.get('outputs', []) if o.get('type') == 'stream']
        print(f"Rank {rank}: status={r['status']}, output={''.join(outs).strip()}")

    await gw.broadcast_shutdown()
    await gw.stop()
    for p in procs:
        p.wait(timeout=5)

asyncio.run(main())
```

```bash
PYTHONPATH=/path/to/jupyterlab python test_subprocess.py
```

Expected output:
```
Gateway on port 41541
Workers: [1, 2, 3, 4]
Rank 1: status=ok, output=torch 2.8.0+cu128, CUDA=True
Rank 2: status=ok, output=torch 2.8.0+cu128, CUDA=True
Rank 3: status=ok, output=torch 2.8.0+cu128, CUDA=True
Rank 4: status=ok, output=torch 2.8.0+cu128, CUDA=True
```

### 4. Option C: Inside JupyterLab notebook

Start JupyterLab and use the gateway API directly from notebook cells:

```bash
PYTHONPATH=/path/to/jupyterlab jupyter lab
```

In a notebook (use a kernel that has torch installed):

```python
# Cell 1: Start gateway and workers
import sys, asyncio
sys.path.insert(0, '/path/to/jupyterlab')
from jupyterlab_distributed.gateway import Gateway
from jupyterlab_distributed.worker import Worker

gw = Gateway(port=0, auth_token='demo', expected_workers=3)
await gw.start()
print(f'Gateway on port {gw.port}')

workers, tasks = [], []
for i in range(1, 4):
    w = Worker(rank=i, server_url=f'ws://localhost:{gw.port}/ws', auth_token='demo')
    workers.append(w)
    tasks.append(asyncio.create_task(w.run()))

for _ in range(50):
    if gw.all_workers_registered(): break
    await asyncio.sleep(0.1)
print(f'Workers: {sorted(gw.workers.keys())}')
```

```python
# Cell 2: Broadcast execution
msg_id = await gw.broadcast_execute("x = 42; print(f'result: {x * 2}')", 'test')
results = await gw.collect_results(msg_id)
for rank in sorted(results.keys()):
    r = results[rank]
    outs = [o['text'] for o in r.get('outputs', []) if o.get('type') == 'stream']
    print(f"Rank {rank}: {''.join(outs).strip()}")
```

> **Note:** When running workers in-process (Option C), `print()` output from
> subsequent cells may not display in the notebook UI due to stdout capture
> by the worker threads. This is an in-process limitation only. With real
> subprocess workers (Options B or torchrun), each worker has its own stdout
> and output works correctly.

### 5. Run the test suite

```bash
# Unit + integration tests (126 tests, no GPU needed)
python -m pytest tests/test_distributed/ -v --noconftest

# Subprocess-based verification (needs torch)
PYTHONPATH=. python tests/test_distributed_subprocess.py
```

---

## Multi-Node Setup (SLURM / HPC)

### 1. Install

```bash
pip install -e ./jupyterlab_distributed
```

### 2. Start JupyterLab on the head node

```bash
jupyter lab --ip=0.0.0.0
```

### 3. Open a notebook and select the "Distributed Python" kernel

The kernel provisioner creates a session config file and waits for the distributed processes to start.

Check the session config path in the kernel startup log:

```
Session config written to /home/user/.jupyter/distributed/abc-123.json
```

### 4. Launch workers via torchrun on compute nodes

Submit a SLURM job or run interactively:

```bash
# Example: 4 nodes, 8 GPUs per node
torchrun --nnodes 4 \
    --nproc_per_node 8 \
    --master-addr $MASTER_ADDR \
    --master-port 29501 \
    --rdzv_backend static \
    --node_rank $NODE_RANK \
    -m jupyterlab_distributed.launcher \
    --session-config /home/user/.jupyter/distributed/abc-123.json
```

Or in an sbatch script:

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

srun torchrun \
    --nnodes $SLURM_JOB_NUM_NODES \
    --nproc_per_node 8 \
    --rdzv_backend static \
    --node_rank $SLURM_NODEID \
    -m jupyterlab_distributed.launcher \
    --session-config /path/to/session-config.json
```

### 5. Wait for workers to connect

The JupyterLab distributed status panel (left sidebar) shows registration progress:

```
Workers: 32/32 ✓
├── compute-01: rank 0-7  ✓
├── compute-02: rank 8-15 ✓
├── compute-03: rank 16-23 ✓
└── compute-04: rank 24-31 ✓
```

### 6. Run cells normally

Every cell executes on all ranks. Rank-0 output is shown by default. Click rank tabs to view other ranks' output.

```python
import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl')
print(f"rank {dist.get_rank()}, world_size {dist.get_world_size()}")
```

Output (rank 0 shown by default):
```
rank 0, world_size 32
```

Click `[1]` `[2]` ... `[31]` tabs to see each rank's output, or `[All Ranks]` to see all.

## Magic Commands

### `%distributed on` / `%distributed off`

Toggle distributed execution. When off, cells only execute on rank-0 (the kernel).

```python
%distributed off
# This cell runs only on rank-0
local_result = expensive_local_computation()

%distributed on
# This cell runs on all ranks again
```

### `%distributed status`

Show connected workers and their health:

```python
%distributed status
```

```
Distributed mode: enabled
Workers: 32 connected, 32 expected
  Rank 0: host=compute-01, gpu=0, pid=12345, state=idle
  Rank 1: host=compute-01, gpu=1, pid=12346, state=idle
  ...
```

### `%distributed expect N`

Set the expected number of workers. Cell execution is blocked until this many workers connect.

```python
%distributed expect 16  # Only wait for 16 workers instead of 32
```

### `%distributed timeout N`

Set the timeout (seconds) for waiting for worker results after rank-0 finishes.

```python
%distributed timeout 60  # Wait up to 60 seconds
```

### `%distributed failure-mode fail-fast` / `best-effort`

Control what happens when a worker fails:

- **fail-fast** (default): Cell fails if any rank fails
- **best-effort**: Cell succeeds if rank-0 succeeds; failed ranks shown in output

```python
%distributed failure-mode best-effort
```

### `%%rank0`

Execute a cell only on rank-0 (skip broadcasting to workers). Useful for logging, checkpointing, or rank-0-only operations.

```python
%%rank0
torch.save(model.state_dict(), "checkpoint.pt")
print("Checkpoint saved (rank 0 only)")
```

### `%%rank N`

Execute a cell on a specific rank only.

```python
%%rank 3
print(f"Debug info from rank 3: {model.module.layer1.weight.mean()}")
```

### `%distributed restart hard`

Kill all worker processes. Use this when you need a full restart (CUDA OOM, corrupted NCCL state).

```python
%distributed restart hard
# Then re-launch torchrun to start fresh
```

## Restart Behavior

### Soft Restart (Kernel → Restart Kernel)

- Clears all Python namespaces on all ranks
- `torch.distributed` group stays alive
- Worker connections stay alive
- SLURM allocation preserved
- Fast (< 1 second)

### Hard Restart (Kernel → Restart Distributed Cluster)

- Kills all processes
- Frees GPU memory and CUDA contexts
- Re-launch torchrun to start fresh
- Use when: CUDA OOM, corrupted state, NCCL hangs

## Output Viewing

Each cell's output has a **rank selector** tab bar:

```
[0] [1] [2] [3] ... [31]  [All Ranks]
```

- **Click a rank number**: see that rank's stdout, stderr, errors, and display output
- **Click "All Ranks"**: expandable accordion showing every rank
- **Color coding**: green = success, red = error

## Log Files

Every rank writes logs to the shared filesystem:

```
~/.jupyter/distributed/<session-id>/rank-0.log
~/.jupyter/distributed/<session-id>/rank-1.log
...
```

If a worker crashes, a crash report is written:

```
~/.jupyter/distributed/<session-id>/rank-3.crash.json
```

These survive process death, CUDA OOM, and segfaults.

## Autoreload

Autoreload works on all ranks. Since all cell code is broadcast, running:

```python
%load_ext autoreload
%autoreload 2
```

...configures autoreload on rank-0 AND all workers. When you edit a module on the shared filesystem, all ranks pick up the changes on the next cell execution.

## Interrupt

Pressing the interrupt button (or Kernel → Interrupt) sends `KeyboardInterrupt` to all ranks. This works even during NCCL collectives (at the Python level).

## Example: DDP Training Loop

```python
# Cell 1: Setup
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
device = torch.device(f'cuda:{rank % 8}')
```

```python
# Cell 2: Model (autocomplete works for DDP methods!)
model = MyModel().to(device)
model = DDP(model, device_ids=[rank % 8])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

```python
# Cell 3: Train one batch
data = get_batch(rank)  # Each rank gets different data
loss = model(data)
loss.backward()
optimizer.step()
optimizer.zero_grad()
print(f"rank {rank}, loss={loss.item():.4f}")
```

```python
%%rank0
# Cell 4: Save checkpoint (rank 0 only)
torch.save(model.module.state_dict(), "model.pt")
print("Saved!")
```

## Troubleshooting

### Workers not connecting

- Check that compute nodes can reach rank-0's IP on the gateway port
- Verify the session config path is accessible from all nodes (shared filesystem)
- Check `--session-config` path matches what JupyterLab printed

### NCCL timeout

- Interrupt from JupyterLab (sends KeyboardInterrupt to all ranks)
- If that doesn't work: `%distributed restart hard`

### CUDA OOM

- `%distributed restart hard` to free all GPU memory
- Re-launch torchrun

### Cell hangs

- Check `%distributed status` — is a worker disconnected?
- Check rank log files for errors
- Interrupt and retry
