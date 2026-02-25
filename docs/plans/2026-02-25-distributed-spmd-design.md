# Distributed SPMD Cell Execution for JupyterLab

## Summary

Add interactive distributed cell execution to JupyterLab, enabling SPMD
(Single Program Multiple Data) workloads across multiple machines. Cells
execute simultaneously on all ranks. Rank-0 is the real Jupyter kernel,
providing native autocomplete, inspection, and variable explorer for DDP
objects. Other ranks run as lightweight worker daemons with embedded IPython
shells.

Target runtime: PyTorch DDP on SLURM HPC clusters via torchrun.

## Goals

- Run notebook code cells interactively across multiple machines in SPMD
  fashion.
- Rank-0 IS the real IPython kernel: autocomplete, inspection, variable
  explorer all show real distributed objects (DDP models, distributed
  tensors).
- Keep UX notebook-native: "Run Cell" works as before.
- Show per-rank outputs with an interactive rank selector.
- Support autoreload on all ranks (shared filesystem).
- Preserve logs and crash reports on hard failures.

## Non-Goals (v1)

- Multi-rank synchronized debugging (step debugger across ranks).
- SLURM job submission from within JupyterLab (user manages their own jobs).
- ipywidgets on worker ranks.
- `input()` on worker ranks.
- Auto-scaling (dynamic rank count changes mid-session).

## Architecture

```
HEAD NODE                              COMPUTE NODES
+--------------------+                 +----------------------------------+
|  JupyterLab        |                 |  torchrun launches:              |
|  (browser + server) |                |                                  |
|                    |                 |  Rank-0 Launcher Process:        |
|  Kernel Provisioner|--ZMQ over TCP-->|    torch.distributed (rank 0)    |
|  (manages lifecycle)|                |    IPython Kernel (restartable)  |
|                    |                 |    WebSocket Gateway (sep thread) |
+--------------------+                 |                                  |
                                       |  Rank 1..N Launcher Processes:   |
                                       |    torch.distributed (rank 1..N) |
                                       |    Worker daemon                 |
                                       |    Embedded IPython shell        |
                                       |    --WebSocket--> Rank-0 Gateway |
                                       +----------------------------------+
```

### Component Overview

1. **Distributed Kernel Provisioner** - Jupyter kernel provisioner that
   coordinates with a remote kernel launched via torchrun.
2. **Rank-0 Launcher** - Persistent process on rank-0 that hosts the IPython
   kernel, WebSocket gateway, and torch.distributed membership.
3. **Worker Daemon** - Lightweight process on ranks 1-N with embedded IPython
   shell, connecting to rank-0 via WebSocket.
4. **Frontend Plugin** - JupyterLab extension for the rank selector output UI,
   distributed status sidebar panel, and toolbar controls.

## Component Details

### 1. Distributed Kernel Provisioner

A custom `KernelProvisionerBase` subclass (Python, server-side).

**Responsibilities:**
- Creates session config on shared filesystem when "Distributed Python"
  kernel is selected.
- Waits for the remote kernel (rank-0) to come alive by polling the session
  config file for status updates.
- Once rank-0 reports "running" with its hostname and ZMQ ports, connects
  the Jupyter server to the remote kernel.
- Handles kernel lifecycle: soft restart (namespace reset) and hard restart
  (process kill).

**Session config file** (`<shared_dir>/.jupyter/distributed/<kernel_id>.json`):
```json
{
  "kernel_id": "abc-123",
  "zmq_ports": {
    "shell": 50001, "iopub": 50002, "stdin": 50003,
    "control": 50004, "hb": 50005
  },
  "gateway_port": 9876,
  "world_size": 32,
  "auth_token": "<random-secret>",
  "status": "waiting",
  "host": null,
  "created_at": "2026-02-25T10:00:00Z",
  "ttl_seconds": 3600
}
```

**Security:**
- File created with `0700` permissions.
- Contains a random auth token that workers must present on WebSocket connect.
- Atomic write via write-to-temp + rename.
- TTL-based cleanup of stale config files.

### 2. Rank-0 Launcher Process

Entry point: `python -m jupyterlab_distributed.launcher --session-config <path>`

Launched as rank 0 by torchrun. This is a persistent process that survives
kernel restarts.

**Startup sequence:**
1. Read session config from shared filesystem.
2. Call `torch.distributed.init_process_group()` (real rank 0).
3. Start IPython kernel on the ZMQ ports specified in config.
4. Start WebSocket gateway on `gateway_port` (separate thread, bounded
   queues).
5. Update session config: `"status": "running", "host": "<hostname>"`.
6. Enter kernel event loop.

**Kernel execution override (`do_execute`):**
Instead of IPython hooks (`pre_execute`/`post_execute`), the kernel subclass
overrides `do_execute` to provide a proper execution transaction boundary:

```python
async def do_execute(self, code, silent, ...):
    if self.distributed_enabled and self.all_workers_registered():
        # 1. Broadcast code to all workers
        msg_id = self.broadcast_execute(code)
        # 2. Execute locally on rank-0 (standard IPython execution)
        result = await super().do_execute(code, silent, ...)
        # 3. Wait for all workers to complete (with timeout)
        worker_results = await self.collect_worker_results(msg_id)
        # 4. Publish per-rank outputs as custom display_data
        self.publish_distributed_outputs(worker_results)
        return result
    else:
        return await super().do_execute(code, silent, ...)
```

**WebSocket gateway (separate thread):**
- Accepts worker connections on `ws://<host>:<gateway_port>/ws`.
- Validates auth token on connect.
- Tracks registered workers by rank.
- Routes execution messages and output streams.
- Bounded output queues with backpressure (prevents OOM).
- Heartbeat pings every 10 seconds.

**Magic commands:**
- `%distributed on/off` - Toggle distributed execution.
- `%distributed status` - Show connected workers, ranks, health.
- `%distributed expect <N>` - Set expected world size.
- `%distributed timeout <seconds>` - Set worker completion timeout.
- `%distributed failure-mode <fail-fast|best-effort>` - Set failure policy.
- `%distributed restart hard` - Kill all processes (hard restart).
- `%%rank0` - Execute cell only on rank-0 (skip broadcast).
- `%%rank <N>` - Execute cell only on rank N.

### 3. Worker Daemon

Entry point: Same launcher module, but detects rank != 0.

**Startup sequence:**
1. Read session config from shared filesystem.
2. Call `torch.distributed.init_process_group()` (real rank 1..N).
3. Create embedded `IPython.core.interactiveshell.InteractiveShell`.
4. Load autoreload extension by default.
5. Connect to rank-0 WebSocket gateway, present auth token and rank.
6. Enter execution loop.

**Execution loop:**
- Receives `{"type": "execute", "msg_id": "...", "code": "..."}`.
- Redirects stdout/stderr capture.
- Calls `shell.run_cell(code)` on the embedded IPython shell.
- Streams output messages back in real-time:
  - `{"type": "stream", "msg_id": "...", "name": "stdout", "text": "..."}`
  - `{"type": "display_data", "msg_id": "...", "data": {...}}`
  - `{"type": "error", "msg_id": "...", "ename": "...", "evalue": "...", "traceback": [...]}`
  - `{"type": "execute_complete", "msg_id": "...", "status": "ok|error", "execution_time": 1.23}`
- Receives `{"type": "interrupt"}` -> raises KeyboardInterrupt in execution thread.
- Receives `{"type": "reset"}` -> clears InteractiveShell namespace (soft restart).
- Receives `{"type": "shutdown"}` -> exits gracefully.

**Health reporting:**
- Periodic heartbeat includes: GPU memory usage, process RSS, execution state.
- Written to log file regardless of WebSocket state.

**Unsupported features (explicit, fail-fast):**
- `input()` -> raises error: "input() not supported on worker ranks"
- Debugger -> not available
- ipywidgets comms -> not available
- Some IPython magics may behave differently (documented)

### 4. Frontend Plugin

A JupyterLab extension package with three components:

#### 4a. Rank Selector Output Widget

Custom MIME renderer for `application/vnd.jupyterlab.distributed+json`.

For each cell output, replaces the default output area with:
- A rank tab bar: clickable rank numbers (0, 1, 2, ..., N-1).
- Default: rank 0 selected, showing its output inline.
- Click another rank: instantly switch to that rank's buffered output.
- "All Ranks" button: accordion view with all ranks.
- Color coding: green (ok), red (error), yellow (slow/warning), gray (no output).
- Search/filter for large rank counts (256+).
- Diff view (optional): highlight differences between selected rank and rank 0.

Per-rank outputs are stored in a per-cell, per-rank output model. The kernel
sends all rank outputs via the custom MIME type in display_data messages.

#### 4b. Distributed Status Sidebar Panel

A Lumino sidebar widget registered in the left or right sidebar.

Sections:
- **Session info**: session ID, connection status, kernel host.
- **Worker registration progress**: progress bar (12/32), per-node breakdown
  of connected vs. missing ranks. "Start anyway" button for partial workers.
- **Nodes tree**: collapsible tree view: nodes -> ranks, showing GPU memory,
  health status, PID per rank.
- **Execution progress**: when a cell is running, per-rank completion
  progress, straggler identification (slowest/fastest rank).
- **Resource summary**: aggregate GPU memory usage bar.
- **Action buttons**: soft restart, hard restart, view logs.

Updated via custom IOPub messages from the kernel's distributed extension
(status updates, health data, execution progress).

#### 4c. Toolbar and Menu Items

- **Toolbar**: worker count indicator ("32/32 workers"), execution mode
  indicator ("Distributed" / "Local").
- **Kernel menu**: "Restart Distributed Cluster" (hard restart).
- **Kernel start dialog**: fields for expected world size and session config
  path when "Distributed Python" kernel is selected.

### 5. Log Preservation and Crash Recovery

#### Persistent Log Files

Every rank writes stdout/stderr to the shared filesystem in real-time:
`<shared_dir>/.jupyter/distributed/<session_id>/rank-<N>.log`

These survive process death, CUDA OOM, and segfaults.

#### Structured Crash Reports

Each rank's launcher wrapper catches crashes and writes:
`<shared_dir>/.jupyter/distributed/<session_id>/rank-<N>.crash.json`

```json
{
  "rank": 3,
  "hostname": "compute-node-02",
  "signal": "SIGSEGV",
  "last_cell_id": "abc-123",
  "last_cell_code": "loss.backward()",
  "gpu_memory_mb": 79800,
  "traceback": "...",
  "timestamp": "2026-02-25T14:32:01Z"
}
```

#### Frontend Crash Recovery

When the kernel connection drops, the frontend:
1. Shows "Distributed kernel crashed" dialog.
2. Reads crash reports from shared filesystem (via server-side API).
3. Displays per-rank crash status: which ranks died, last error, GPU memory
   at time of crash.

#### Proactive Health Warnings

Workers report GPU memory in heartbeats. Frontend shows warnings when GPU
memory exceeds 90% on any rank.

## Execution Flow

### Normal Cell Execution

1. User clicks "Run Cell".
2. JupyterLab sends `execute_request` to rank-0 kernel (standard protocol).
3. Kernel's `do_execute` override:
   a. Broadcasts cell code to all workers via WebSocket gateway.
   b. Workers begin executing in their embedded IPython shells.
   c. Rank-0 executes locally (standard IPython execution).
   d. Workers stream outputs back to gateway in real-time.
   e. `do_execute` waits for all workers to complete (configurable timeout).
   f. Publishes per-rank outputs as `display_data` with custom MIME type.
4. Frontend renders rank-0 output inline, other ranks in rank selector.

### Worker Registration

1. Kernel provisioner writes session config with `world_size` and auth token.
2. User launches `torchrun` on compute nodes.
3. Workers connect to rank-0 gateway, present auth token and rank.
4. Gateway tracks registration: broadcasts progress via IOPub.
5. Frontend status panel shows registration progress with per-node breakdown.
6. Cell execution is blocked until all expected workers are registered
   (or user clicks "Start anyway").

### Soft Restart (Restart Kernel)

1. Kernel receives `shutdown_request`.
2. Launcher process catches it (does NOT exit).
3. Clears IPython kernel namespace (`shell.reset()`).
4. Sends `{"type": "reset"}` to all workers.
5. Workers clear their InteractiveShell namespaces.
6. Kernel reports ready. torch.distributed group and worker connections
   survive.

### Hard Restart (Restart Distributed Cluster)

1. User selects "Restart Distributed Cluster" from menu.
2. Kernel sends `{"type": "shutdown"}` to all workers.
3. Workers exit gracefully.
4. Launcher process exits (kills torch.distributed group).
5. SLURM allocation may still be alive.
6. Kernel provisioner goes to "waiting for distributed kernel" state.
7. User re-launches torchrun to start fresh.

### Interrupt

1. User presses interrupt in JupyterLab.
2. Standard kernel interrupt on rank-0.
3. Distributed extension broadcasts `{"type": "interrupt"}` to all workers.
4. Workers raise KeyboardInterrupt in execution threads.
5. If workers don't respond within 5 seconds: SIGKILL escalation.

## Error Handling

| Scenario | Behavior |
|---|---|
| Worker crashes mid-execution | Gateway detects disconnect, marks rank failed. Default: fail-fast (cell fails). Optional: best-effort (continue with remaining ranks). Frontend shows error on rank tab. |
| Worker hangs | Timeout after configurable seconds. Mark as timed-out. SIGKILL escalation. |
| Rank-0 kernel crashes | Standard kernel death. Workers detect WebSocket close, exit. Crash reports written to shared FS. Frontend shows crash dialog. |
| NCCL collective hangs | All ranks hang. User interrupts. Broadcast interrupt -> 5s grace -> SIGKILL. |
| Partial worker registration | Frontend shows progress. User can start with partial workers or wait. |
| Output explosion (large rank count) | Per-rank output size limit (configurable, default 1MB). Bounded queues with backpressure in gateway thread. |
| Network partition | Heartbeat detects missing workers. Frontend shows disconnected ranks. |
| Stale session config | TTL-based cleanup. Provisioner checks timestamps. |

Default failure mode: **fail-fast**. Configurable to best-effort via
`%distributed failure-mode best-effort`.

## Wire Protocol (WebSocket Messages)

### Worker -> Gateway

```
{"type": "register", "rank": N, "hostname": "...", "gpu_id": 0, "pid": 12345, "token": "..."}
{"type": "heartbeat", "rank": N, "gpu_memory_mb": 45200, "rss_mb": 12000, "state": "idle|executing"}
{"type": "stream", "msg_id": "...", "name": "stdout|stderr", "text": "..."}
{"type": "display_data", "msg_id": "...", "data": {...}, "metadata": {...}}
{"type": "execute_result", "msg_id": "...", "data": {...}, "execution_count": N}
{"type": "error", "msg_id": "...", "ename": "...", "evalue": "...", "traceback": [...]}
{"type": "execute_complete", "msg_id": "...", "status": "ok|error", "execution_time": 1.23}
```

### Gateway -> Worker

```
{"type": "registered", "rank": N, "world_size": 32}
{"type": "execute", "msg_id": "...", "code": "...", "cell_id": "..."}
{"type": "interrupt"}
{"type": "reset"}
{"type": "shutdown"}
```

### Kernel -> Frontend (via IOPub display_data)

```json
{
  "data": {
    "application/vnd.jupyterlab.distributed+json": {
      "type": "rank_outputs",
      "msg_id": "...",
      "ranks": {
        "0": {"outputs": [...], "status": "ok", "execution_time": 0.8},
        "1": {"outputs": [...], "status": "ok", "execution_time": 0.9},
        "16": {"outputs": [...], "status": "error", "execution_time": 1.2}
      }
    }
  }
}
```

```json
{
  "data": {
    "application/vnd.jupyterlab.distributed+json": {
      "type": "cluster_status",
      "registered": 32,
      "expected": 32,
      "nodes": {
        "compute-01": {"ranks": [0,1,2,3,4,5,6,7], "status": "healthy"},
        "compute-02": {"ranks": [8,9,10,11,12,13,14,15], "status": "healthy"}
      },
      "gpu_memory_total_mb": 1400000,
      "gpu_memory_used_mb": 784000
    }
  }
}
```

## Package Structure

New packages added to the JupyterLab fork:

```
packages/
  distributed/                    # Shared types and interfaces
    src/
      tokens.ts                   # IDistributedSession, IDistributedStatusModel
      types.ts                    # Wire protocol message types
  distributed-extension/          # Frontend plugin
    src/
      index.ts                    # Plugin registration
      panel.ts                    # Sidebar status panel widget
      rankSelector.ts             # Rank selector output widget
      mimeRenderer.ts             # Custom MIME renderer
      toolbar.ts                  # Toolbar widgets

jupyterlab_distributed/           # Python package (server-side)
  __init__.py
  provisioner.py                  # DistributedKernelProvisioner
  launcher.py                     # Entry point for torchrun
  kernel.py                       # Subclassed IPython kernel with do_execute override
  gateway.py                      # WebSocket gateway (runs in rank-0 process)
  worker.py                       # Worker daemon with embedded InteractiveShell
  magics.py                       # %distributed and %%rank magic commands
  handlers.py                     # Server API for crash log retrieval
  config.py                       # Session config read/write with atomic ops
```

## Testing Strategy

### Unit Tests (Python)

- Provisioner lifecycle: create config, detect kernel alive, handle timeout.
- Gateway: worker registration, message routing, bounded queues, auth.
- Kernel do_execute override: broadcast, collect, timeout, error propagation.
- Worker: execution, output capture, interrupt handling, reset.
- Config: atomic write, TTL cleanup, permission checks.
- Magics: all magic commands parse and execute correctly.

### Unit Tests (TypeScript)

- Rank selector widget: renders outputs, switches ranks, handles errors.
- Status panel: displays registration progress, node tree, execution state.
- MIME renderer: parses distributed JSON, creates correct widgets.
- Toolbar: worker count display, mode indicator.

### Integration Tests

- End-to-end: start distributed kernel, connect workers, execute cell,
  verify per-rank output.
- Soft restart: verify namespace clears, workers survive, re-execute works.
- Hard restart: verify all processes die, provisioner resets.
- Interrupt: verify broadcast to workers, SIGKILL escalation.
- Failure modes: worker crash (fail-fast and best-effort), timeout, OOM.
- Log preservation: verify log files and crash reports on worker death.
- Large rank count: test with 64+ mock workers for output rendering
  performance.

## Rollout Plan

1. **Phase 1: Core infrastructure** - Launcher, worker daemon, WebSocket
   gateway, kernel subclass with do_execute override. Test with 2 local
   processes.
2. **Phase 2: Kernel provisioner** - Session config, provisioner, connection
   management. Test with real SLURM + torchrun.
3. **Phase 3: Frontend - output rendering** - Custom MIME renderer, rank
   selector widget.
4. **Phase 4: Frontend - status panel** - Sidebar panel, toolbar widgets,
   registration progress UI.
5. **Phase 5: Hardening** - Log preservation, crash reports, SIGKILL
   escalation, output size limits, security (auth tokens).
6. **Phase 6: Polish** - Diff view between ranks, straggler detection, GPU
   memory warnings, documentation.

## Assumptions

- Shared filesystem accessible from all nodes (standard HPC).
- Users manage their own SLURM job submissions.
- PyTorch DDP with torchrun is the primary launch mechanism.
- Compute nodes can reach each other via TCP (for WebSocket and ZMQ).
- Single JupyterLab instance per distributed session.
