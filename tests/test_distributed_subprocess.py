"""Test distributed SPMD with real subprocess workers.

This avoids the in-process stdout capture issue by running workers
as actual separate processes, which is how the real system works.
"""
import asyncio
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, '/home/jinyh/jupyterlab')
PYTHON = '/home/jinyh/miniconda3/envs/hgen-dev/bin/python'

from jupyterlab_distributed.gateway import Gateway


# Worker script that runs as a separate process
WORKER_SCRIPT = '''
import asyncio, json, os, sys, socket
sys.path.insert(0, '/home/jinyh/jupyterlab')
from jupyterlab_distributed.worker import Worker

rank = int(sys.argv[1])
url = sys.argv[2]
token = sys.argv[3]

w = Worker(rank=rank, server_url=url, auth_token=token)
asyncio.run(w.run())
'''


async def main():
    # Start gateway
    gw = Gateway(port=0, auth_token='subprocess-test', expected_workers=3)
    await gw.start()
    url = f'ws://localhost:{gw.port}/ws'
    print(f'Gateway started on port {gw.port}')

    # Launch 3 worker subprocesses
    procs = []
    for rank in range(1, 4):
        p = subprocess.Popen(
            [PYTHON, '-c', WORKER_SCRIPT, str(rank), url, 'subprocess-test'],
            env={**os.environ, 'PYTHONPATH': '/home/jinyh/jupyterlab'},
        )
        procs.append(p)
    print(f'Launched {len(procs)} worker processes')

    # Wait for registration
    for i in range(100):
        if gw.all_workers_registered():
            break
        await asyncio.sleep(0.1)
    print(f'Workers registered: {sorted(gw.workers.keys())}')
    assert gw.all_workers_registered(), "Not all workers registered"

    # === Test 1: Simple broadcast ===
    print('\n=== Test 1: Broadcast "Hello" to all workers ===')
    msg_id = await gw.broadcast_execute(
        "import os; print(f'Hello from PID {os.getpid()}, rank reporting in!')",
        'hello'
    )
    results = await gw.collect_results(msg_id)
    for rank in sorted(results.keys()):
        r = results[rank]
        outs = [o['text'] for o in r.get('outputs', []) if o.get('type') == 'stream']
        print(f'  Rank {rank}: status={r["status"]}, output={"".join(outs).strip()}')
    assert all(r['status'] == 'ok' for r in results.values()), "Some workers failed"
    print('PASS')

    # === Test 2: Computation on all workers ===
    print('\n=== Test 2: Distributed computation (sum on each rank) ===')
    msg_id = await gw.broadcast_execute(
        "import os; rank = os.getpid() % 1000; result = sum(range(rank)); print(f'rank_id={rank}, sum={result}')",
        'compute'
    )
    results = await gw.collect_results(msg_id)
    for rank in sorted(results.keys()):
        r = results[rank]
        outs = [o['text'] for o in r.get('outputs', []) if o.get('type') == 'stream']
        print(f'  Rank {rank}: status={r["status"]}, output={"".join(outs).strip()}')
    assert all(r['status'] == 'ok' for r in results.values())
    print('PASS')

    # === Test 3: Import torch on all workers ===
    print('\n=== Test 3: Import torch on all workers ===')
    msg_id = await gw.broadcast_execute(
        "import torch; print(f'torch {torch.__version__}, CUDA={torch.cuda.is_available()}')",
        'torch'
    )
    results = await gw.collect_results(msg_id)
    for rank in sorted(results.keys()):
        r = results[rank]
        outs = [o['text'] for o in r.get('outputs', []) if o.get('type') == 'stream']
        print(f'  Rank {rank}: status={r["status"]}, output={"".join(outs).strip()}')
    assert all(r['status'] == 'ok' for r in results.values())
    print('PASS')

    # === Test 4: Error handling ===
    print('\n=== Test 4: Error handling (1/0 on all workers) ===')
    msg_id = await gw.broadcast_execute("1/0", 'error')
    results = await gw.collect_results(msg_id)
    for rank in sorted(results.keys()):
        r = results[rank]
        print(f'  Rank {rank}: status={r["status"]}, error={r.get("ename", "")}')
    assert all(r['status'] == 'error' for r in results.values())
    print('PASS')

    # === Test 5: State persists across cells ===
    print('\n=== Test 5: State persistence across cells ===')
    msg_id = await gw.broadcast_execute("shared_var = 42", 'set')
    await gw.collect_results(msg_id)
    msg_id = await gw.broadcast_execute("print(f'shared_var = {shared_var}')", 'get')
    results = await gw.collect_results(msg_id)
    for rank in sorted(results.keys()):
        r = results[rank]
        outs = [o['text'] for o in r.get('outputs', []) if o.get('type') == 'stream']
        print(f'  Rank {rank}: status={r["status"]}, output={"".join(outs).strip()}')
    assert all(r['status'] == 'ok' for r in results.values())
    assert all('shared_var = 42' in ''.join(o['text'] for o in r.get('outputs', []) if o.get('type') == 'stream') for r in results.values())
    print('PASS')

    # Cleanup
    print('\n=== Cleanup ===')
    await gw.broadcast_shutdown()
    await gw.stop()
    for p in procs:
        p.wait(timeout=5)
    print('All workers stopped')

    print('\n=============================')
    print('ALL 5 TESTS PASSED')
    print('=============================')


asyncio.run(main())
