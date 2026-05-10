# Chapter 54 — Persistent Kernels: The Server Pattern

## What Is a Persistent Kernel?

On real CUDA hardware, a **persistent kernel** is a long-running kernel that occupies GPU SMs continuously, polling an input queue for work rather than returning to the host between each task. The pattern eliminates kernel-launch overhead — which costs several microseconds on real hardware — for workloads that need to dispatch many short tasks sequentially.

In gpusim, Phase 14 introduces two new types that model this pattern:

```python
from gpusim.persistent.queue import WorkQueue
from gpusim.persistent.kernel import PersistentKernel
```

`WorkQueue` is a FIFO queue that the host pushes work items into. `PersistentKernel` pulls items one by one, launching the underlying PTX kernel once per item and collecting results until the queue is stopped and empty.

The core data model:

```python
from collections import deque
from dataclasses import dataclass, field

@dataclass
class WorkQueue:
    items: deque = field(default_factory=deque)
    stopped: bool = False

    def push(self, item) -> None:
        if self.stopped:
            raise RuntimeError("queue stopped; cannot push")
        self.items.append(item)

    def pop(self):
        if not self.items:
            return None
        return self.items.popleft()

    def stop(self) -> None:
        self.stopped = True
```

Once `stop()` is called, no further items can be pushed. The `PersistentKernel.start()` call blocks until the queue is stopped and fully drained.

## The persistent_kernel_server Demo

The `persistent_kernel_server` example creates five output buffers, pushes them as work items, stops the queue, then starts a persistent kernel to process all five in sequence:

```python
import numpy as np, pathlib
from gpusim.persistent.queue import WorkQueue
from gpusim.persistent.kernel import PersistentKernel
from gpusim.config.loader import load_default

cfg = load_default()
ptx = pathlib.Path("examples/persistent_kernel_server/kernel.ptx").read_text()

queue = WorkQueue()
out_bufs = []
for _ in range(5):
    ob = np.zeros(32, dtype=np.uint32)
    out_bufs.append(ob)
    queue.push({"OUT": ob})
queue.stop()

pk = PersistentKernel(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                      params_template={}, work_queue=queue,
                      kernel_name="server")
results = pk.start(cfg)

print(f"Processed {len(results)} items")   # 5
for ob in out_bufs:
    print(ob.sum())   # 32 — each of 32 threads wrote 1
```

Run it:

```bash
python examples/persistent_kernel_server/run.py
```

The kernel (`kernel.ptx`) writes the value `1` into every element of the output buffer. Each work item provides a different `OUT` buffer via the params dict, so all five buffers are filled independently.

The key sequence:

1. `queue.push({"OUT": ob})` enqueues a dict that overrides `params_template` per item.
2. `queue.stop()` signals that no more items will arrive.
3. `pk.start(cfg)` loops: `pop()` → `Stream.launch()` → `synchronize()` until `pop()` returns `None`.
4. Returns a list of one `Result` per processed item.

## 看模拟器

**用 Recorder 追踪 is_persistent 事件：**

Pass a `Recorder` to `pk.start()` to capture KernelLaunch events for every work item:

```python
from gpusim.trace.recorder import Recorder

rec = Recorder()
pk.start(cfg, recorder=rec)

persistent_events = [e for e in rec.kernel_launch_events if e.is_persistent]
print(f"Persistent launches: {len(persistent_events)}")   # 5
print(f"All have parent_kernel_id=-1: {all(e.parent_kernel_id == -1 for e in persistent_events)}")
```

Each iteration records a `KernelLaunch` event with `is_persistent=True` and `parent_kernel_id=-1` (not a child of another kernel). The new KernelLaunch fields added in Phase 14:

```python
@dataclass(frozen=True)
class KernelLaunch:
    # ... existing fields ...
    parent_kernel_id: int = -1    # -1 = top-level (not a child)
    is_persistent: bool = False   # True if emitted by PersistentKernel
```

Use the `persistent_kernel_throughput` metric to measure iterations per 1000 cycles:

```python
from gpusim.analysis.metrics import persistent_kernel_throughput
import pandas as pd

df = pd.DataFrame([
    {"is_persistent": e.is_persistent, "stream_id": i, "parent_kernel_id": e.parent_kernel_id}
    for i, e in enumerate(rec.kernel_launch_events)
])
rate = persistent_kernel_throughput(df, total_cycles=5000)
print(f"Throughput: {rate:.2f} iters/1000 cycles")
```

## 改一改

**改变 work item 数量，观察 results 长度线性增长：**

Replace the fixed loop of 5 with a variable count:

```python
N = 10
queue = WorkQueue()
out_bufs = [np.zeros(32, dtype=np.uint32) for _ in range(N)]
for ob in out_bufs:
    queue.push({"OUT": ob})
queue.stop()

pk = PersistentKernel(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                      params_template={}, work_queue=queue)
results = pk.start(cfg)
assert len(results) == N   # always N
```

Try stopping the queue before pushing any items:

```python
q = WorkQueue()
q.stop()
pk = PersistentKernel(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                      params_template={}, work_queue=q)
r = pk.start(cfg)
assert r == []   # empty — queue was already drained (empty) at stop
```

Try pushing after stop to see the guard:

```python
q = WorkQueue()
q.stop()
try:
    q.push({"OUT": np.zeros(32)})
except RuntimeError as exc:
    print(exc)   # "queue stopped; cannot push"
```

## 真机对照

On real CUDA, a persistent kernel occupies SMs indefinitely and polls a lock-free queue in device global memory:

```c
__global__ void server_kernel(WorkItem* queue, int* head, int* tail,
                              unsigned int* outputs) {
    while (true) {
        int idx = atomicAdd(head, 1);
        if (idx >= *tail) break;    // queue exhausted
        WorkItem item = queue[idx];
        // Process item — write result to outputs[idx * blockDim.x + threadIdx.x]
        outputs[idx * blockDim.x + threadIdx.x] = 1u;
    }
}
```

| Aspect | `gpusim` | CUDA runtime |
|---|---|---|
| **Queue** | `WorkQueue` (Python deque, host-side) | Lock-free ring buffer in device global memory |
| **Pop semantics** | Host calls `pop()` between Python launches | Atomic `atomicAdd` on device-side head pointer |
| **Kernel re-launch** | Separate `Stream.launch()` per item (no true persistency) | Single `<<<...>>>` launch; kernel loops internally on device |
| **Stop signal** | `queue.stop()` + empty check in `start()` loop | Atomic flag or sentinel item in device queue |
| **Throughput** | `persistent_kernel_throughput` metric (iterations/1000 cycles) | `ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed` |

The main modeling difference: in gpusim, each work item becomes a separate `Stream.launch()` call, so SM occupancy drops to zero between items. Real persistent kernels maintain 100% SM occupancy across all items, which is their primary advantage. Phase 14 models the semantic behavior (per-item results, is_persistent tracing) without simulating intra-SM idle suppression.

Chapter 55 shows how to grow the queue dynamically while the kernel is processing.
