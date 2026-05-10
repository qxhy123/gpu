# Chapter 56 — Dynamic Parallelism: Parent Triggers Child

## What Is Dynamic Parallelism?

On real CUDA hardware, **dynamic parallelism** lets a kernel running on the GPU launch additional child kernels without returning to the CPU. The parent kernel calls `cudaLaunchKernel()` from device code, the CUDA runtime queues the child kernel, and the child executes concurrently (or sequentially, depending on synchronization) with the tail of the parent.

In gpusim, Phase 14 models this with a host-side callback API:

```python
from gpusim.persistent.dynamic import (
    device_launch,
    drain_pending_child_launches,
    reset_pending_child_launches,
)
```

`device_launch()` records a pending child kernel with a reference to its parent's stream ID. `drain_pending_child_launches()` processes the queue of pending children, launching each one on a fresh `Stream` and returning the list of results. The two-step separation models the real-hardware sequencing: parent retires, then the CUDA runtime processes its child launch requests.

```python
# Parent kernel ran on stream s (stream_id = s.stream_id)
device_launch(
    parent_kernel_id=s.stream_id,
    ptx_src=ptx,
    grid=(1,1,1), block=(32,1,1),
    params={"OUT": out_b},
    kernel_name="child",
)

# Process all pending children (one launch each)
child_results = drain_pending_child_launches(cfg)
```

## The dynamic_parallelism_recursive Demo

The `dynamic_parallelism_recursive` example demonstrates a three-level chain: parent → child → grandchild.

```python
import numpy as np, pathlib
from gpusim.persistent.dynamic import (
    device_launch, drain_pending_child_launches, reset_pending_child_launches,
)
from gpusim.api import Stream, synchronize
from gpusim.config.loader import load_default

cfg = load_default()
ptx = pathlib.Path("examples/dynamic_parallelism_recursive/kernel.ptx").read_text()

reset_pending_child_launches()   # clear any leftover state from previous runs

out_a = np.zeros(32, dtype=np.uint32)
out_b = np.zeros(32, dtype=np.uint32)
out_c = np.zeros(32, dtype=np.uint32)

# Level 0: parent kernel
s = Stream()
s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
         params={"OUT": out_a}, kernel_name="parent", config=cfg)
synchronize(streams=[s], config=cfg)
parent_id = s.stream_id
print(f"Parent ran: out_a sum = {out_a.sum()}")   # 32

# Level 1: child (triggered by parent)
device_launch(parent_kernel_id=parent_id, ptx_src=ptx,
              grid=(1,1,1), block=(32,1,1),
              params={"OUT": out_b}, kernel_name="child")
child_results = drain_pending_child_launches(cfg)
print(f"Child ran: out_b sum = {out_b.sum()}")   # 32

# Level 2: grandchild (triggered by child)
device_launch(parent_kernel_id=parent_id + 1, ptx_src=ptx,
              grid=(1,1,1), block=(32,1,1),
              params={"OUT": out_c}, kernel_name="grandchild")
drain_pending_child_launches(cfg)
print(f"Grandchild ran: out_c sum = {out_c.sum()}")   # 32
```

Run it:

```bash
python examples/dynamic_parallelism_recursive/run.py
```

Each level writes `1` into every thread's output slot. After all three levels, `out_a`, `out_b`, and `out_c` each sum to 32.

The key data structure: `_pending_child_launches` is a module-level list in `gpusim.persistent.dynamic`. Each `device_launch()` call appends a dict:

```python
{
    "parent_kernel_id": parent_id,
    "ptx_src": ptx,
    "grid": (1,1,1),
    "block": (32,1,1),
    "params": {"OUT": out_b},
    "kernel_name": "child",
}
```

`drain_pending_child_launches()` pops entries front-to-back and executes them as independent `Stream` launches.

## 看模拟器

**用 dynamic_parallelism_depth 和 fanout 分析调用图：**

Build a `KernelLaunch` dataframe from a `Recorder` and pass it to the Phase 14 metrics:

```python
import pandas as pd
from gpusim.trace.recorder import Recorder
from gpusim.analysis.metrics import dynamic_parallelism_depth, dynamic_parallelism_fanout

# Build a synthetic trace matching the demo chain: 0 → 1 → 2
df = pd.DataFrame([
    {"stream_id": 0, "parent_kernel_id": -1},   # parent (top-level)
    {"stream_id": 1, "parent_kernel_id": 0},   # child of parent
    {"stream_id": 2, "parent_kernel_id": 1},   # grandchild of child
])

depth = dynamic_parallelism_depth(df)
fanout = dynamic_parallelism_fanout(df)

print(f"Max depth: {depth}")   # 3 (parent → child → grandchild)
print(f"Fanout: {fanout}")     # {0: 1, 1: 1} — each parent has 1 child
```

`dynamic_parallelism_depth()` walks the parent→child relationships and returns the maximum chain length. `dynamic_parallelism_fanout()` returns a dict mapping each parent stream_id to the count of its direct children.

For a tree where the parent spawns two children:

```python
df2 = pd.DataFrame([
    {"stream_id": 0, "parent_kernel_id": -1},
    {"stream_id": 1, "parent_kernel_id": 0},
    {"stream_id": 2, "parent_kernel_id": 0},   # second child
])
assert dynamic_parallelism_fanout(df2)[0] == 2   # two children
```

**用 Recorder 捕获 parent_kernel_id 字段：**

Pass a recorder to `drain_pending_child_launches`:

```python
from gpusim.trace.recorder import Recorder

rec = Recorder()
reset_pending_child_launches()

device_launch(parent_kernel_id=0, ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": out_b}, kernel_name="child")
drain_pending_child_launches(cfg, recorder=rec)

# The child's KernelLaunch event has parent_kernel_id=0
child_ev = rec.kernel_launch_events[-1]
assert child_ev.parent_kernel_id == 0
assert child_ev.is_persistent is False
```

## 改一改

**添加第三个孙子内核 — 观察 depth 递增：**

```python
# Four levels: root → L1 → L2 → L3
reset_pending_child_launches()

s = Stream()
s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
         params={"OUT": out_a}, kernel_name="root", config=cfg)
synchronize(streams=[s], config=cfg)
root_id = s.stream_id

# L1 child
device_launch(parent_kernel_id=root_id, ptx_src=ptx,
              grid=(1,1,1), block=(32,1,1),
              params={"OUT": out_b}, kernel_name="L1")
drain_pending_child_launches(cfg)

# L2 child
device_launch(parent_kernel_id=root_id + 1, ptx_src=ptx,
              grid=(1,1,1), block=(32,1,1),
              params={"OUT": out_c}, kernel_name="L2")
drain_pending_child_launches(cfg)

# Manually build the trace
df = pd.DataFrame([
    {"stream_id": root_id, "parent_kernel_id": -1},
    {"stream_id": root_id + 1, "parent_kernel_id": root_id},
    {"stream_id": root_id + 2, "parent_kernel_id": root_id + 1},
])
assert dynamic_parallelism_depth(df) >= 2
```

**验证 reset_pending_child_launches 清空状态：**

`reset_pending_child_launches()` is a test helper that clears the module-level list. Always call it at the start of any test or script that uses `device_launch` to avoid cross-test contamination:

```python
from gpusim.persistent.dynamic import _pending_child_launches, reset_pending_child_launches

device_launch(parent_kernel_id=0, ptx_src="x", grid=(1,1,1), block=(32,1,1),
              params={}, kernel_name="orphan")
assert len(_pending_child_launches) == 1

reset_pending_child_launches()
assert len(_pending_child_launches) == 0
```

## 真机对照

On real CUDA, dynamic parallelism uses `cudaLaunchKernel` from device code:

```c
__global__ void parent_kernel(float* out) {
    // ... parent work ...

    // Launch child kernel from device
    dim3 grid(1), block(32);
    child_kernel<<<grid, block, 0, cudaStreamFireAndForget>>>(out + blockDim.x);
    cudaDeviceSynchronize();   // wait for child to complete
}
```

| Aspect | `gpusim` | CUDA runtime |
|---|---|---|
| **Launch point** | Host-side `device_launch()` call | Device-side `cudaLaunchKernel()` or `<<<>>>` syntax |
| **Pending queue** | `_pending_child_launches` list (module-level) | Device-side launch buffer in SM-local queue |
| **Drain** | Explicit `drain_pending_child_launches(cfg)` | Automatic: CUDA runtime drains after parent grid retires |
| **Parent ID** | `parent_kernel_id` int (stream_id proxy) | Built-in dependency via parent→child grid ordering |
| **Child stream** | Fresh `Stream()` per child | `cudaStreamFireAndForget` or inherited parent stream |
| **Depth metric** | `dynamic_parallelism_depth(df)` | `ncu --metrics launch__child_grid_depth` |
| **Fanout metric** | `dynamic_parallelism_fanout(df)` | `ncu --metrics launch__child_grid_count` |

The main modeling difference: real dynamic parallelism runs the child grid **on the same GPU** concurrently with the tail of the parent. In gpusim, child kernels execute after `drain_pending_child_launches()` is called from the host, so there is no true concurrency between parent and child. The semantic behavior — chained launches, parent→child ID tracing, depth/fanout metrics — is fully modeled.

Chapter 57 brings everything together: a persistent producer kernel that feeds a persistent consumer via a shared `WorkQueue`, demonstrating the capstone producer-consumer pipeline pattern.
