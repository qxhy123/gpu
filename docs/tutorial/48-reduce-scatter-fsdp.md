# Chapter 48 — Reduce-Scatter for FSDP Gradient Reduction

## The FSDP Gradient Reduction Pattern

Fully Sharded Data Parallel (FSDP) training works by distributing model parameters and gradients across ranks. During the backward pass, each rank has accumulated a local gradient tensor. The goal is to reduce all gradients across ranks and shard the result — each rank should own exactly 1/N of the fully-reduced gradient. This is precisely what reduce_scatter accomplishes in a single collective operation.

The reduce_scatter operation combines a reduce (element-wise sum across all ranks) and a scatter (distributing non-overlapping chunks of the result back to each rank). If you have N ranks and a gradient buffer of K elements, each rank ends up holding K/N elements of the fully-reduced gradient. The NCCL function `ncclReduceScatter` implements this on real hardware.

## The Ring Algorithm

The simulator implements reduce_scatter using a ring algorithm. In a ring of N ranks, each rank makes exactly N-1 point-to-point transfers over NVLink to complete the reduce_scatter. Here is the pattern:

1. The N elements of the input buffer are divided into N chunks.
2. In step 1, each rank sends its chunk to the next rank in the ring and receives a chunk from the previous rank.
3. The received chunk is accumulated into a running sum.
4. After N-1 steps, each rank's designated output slot holds the fully-reduced sum.

The N-1 transfer count per rank is important: it sets a firm lower bound on the number of NVLink transfers needed to implement a correct ring reduce_scatter for any N. You can measure this directly in the simulator.

## The reduce_scatter_fsdp Demo

The `reduce_scatter_fsdp` example simulates 4-GPU FSDP gradient reduction:

```python
from gpusim.comm.comm import Comm
from gpusim.comm.system import MultiGpuSystem
from gpusim.config.loader import load_default
import numpy as np

cfg = load_default()
cfg.n_gpus = 4
sys = MultiGpuSystem.from_config(cfg)

# Rank 0 holds a 64-element gradient buffer (all 1.0)
comm = Comm(rank=0, world_size=4, system=sys)
grads = np.full(64, 1.0, dtype=np.float32)   # simulates per-rank local gradient
my_chunk = np.zeros(16, dtype=np.float32)     # output: rank 0 gets elements 0..15

cycles = comm.reduce_scatter(grads, my_chunk, op="sum")
print(f"Reduce_scatter: {cycles} cycles")
print(f"Rank 0 chunk[0:4] = {list(my_chunk[0:4])}")   # → [4.0, 4.0, 4.0, 4.0]
```

Run it:

```bash
python examples/reduce_scatter_fsdp/run.py
```

With 4 GPUs all holding `1.0` gradients, the reduced sum is `4.0`. Rank 0's output chunk covers elements 0 through 15 of the fully-reduced result — a 16-element slice because 64 / 4 = 16. Each of the other ranks would own a different 16-element slice if you ran `Comm(rank=1, ...)` through `Comm(rank=3, ...)`.

## 看模拟器

**观察 `reduce_scatter_step_count` 指标验证 N-1 传输次数：**

You can attach a `Recorder` to count exactly how many NVLink transfers the ring algorithm issues:

```python
from gpusim.trace.recorder import Recorder
from gpusim.analysis.metrics import reduce_scatter_step_count

rec = Recorder()

# Hook the recorder into the fabric so every transfer is captured
original_transfer = sys.nvlink_fabric.transfer
def traced_transfer(*args, **kwargs):
    kwargs["recorder"] = rec
    return original_transfer(*args, **kwargs)
sys.nvlink_fabric.transfer = traced_transfer

comm = Comm(rank=0, world_size=4, system=sys)
comm._recorder = rec
comm.reduce_scatter(grads, my_chunk, op="sum")

# NVLink transfers: exactly N-1 = 3 for world_size=4
print("NVLink transfers:", len(rec.nvlink_transfer_events))

# Collective events carry n_steps
step_counts = reduce_scatter_step_count(rec.to_collective_df())
print("Step count distribution:", step_counts)   # → {3: 1}
```

The `reduce_scatter_step_count` metric returns a dict mapping `n_steps → call_count`. For a single reduce_scatter on 4 GPUs you should see `{3: 1}`. If you ran 3 reduce_scatter calls on an 8-GPU ring you would see `{7: 3}`.

The NVLink transfer count matches the theoretical minimum for the ring algorithm and serves as a correctness check: if you ever see more than N-1 transfers, something in the scheduling is wrong.

## 改一改

**把 `n_gpus` 改成 8，观察传输次数翻倍：**

Change `cfg.n_gpus = 8` and increase the buffer sizes to match:

```python
cfg.n_gpus = 8
sys = MultiGpuSystem.from_config(cfg)
comm = Comm(rank=0, world_size=8, system=sys)

grads = np.full(128, 1.0, dtype=np.float32)   # 128 elements / 8 ranks = 16 per rank
my_chunk = np.zeros(16, dtype=np.float32)

comm.reduce_scatter(grads, my_chunk, op="sum")
# NVLink transfers: N-1 = 7 for world_size=8
```

You should observe 7 NVLink transfer events in the recorder — one per step of the ring. The cycle count also increases because you need more hops to complete the reduction, even though each individual transfer carries the same payload size.

You can also try different buffer sizes. The number of ring steps stays fixed at N-1 regardless of buffer size (the payload per step changes, but not the step count). This illustrates that reduce_scatter has a latency floor of N-1 serial NVLink hops even if bandwidth is infinite.

## 真机对照

On real Hopper hardware, `ncclReduceScatter` implements the same ring algorithm at the hardware level:

```cpp
// NCCL C++ — reduce_scatter on real GPU cluster
ncclReduceScatter(
    send_buff,     // input: full gradient tensor on this rank
    recv_buff,     // output: 1/N chunk for this rank
    count / world_size,
    ncclFloat,
    ncclSum,
    comm,
    stream
);
```

| Behavior | Simulator (`Comm.reduce_scatter`) | Real NCCL (`ncclReduceScatter`) |
|---|---|---|
| **Algorithm** | Ring (N-1 steps) | Ring (default for large messages) |
| **Transfers per rank** | N-1 NVLink sends | N-1 NVLink sends |
| **Output** | 1/N chunk, reduced sum | 1/N chunk, reduced sum |
| **Recorder** | `Recorder.nvlink_transfer_events` | `ncu` profile counter `nvlink_total_data_transmitted` |
| **FSDP integration** | Manual `Comm` usage | `torch.distributed.ReduceScatter` wraps `ncclReduceScatter` |

In PyTorch FSDP, the gradient reduce_scatter happens automatically at the end of the backward pass — you do not call it manually. The simulator makes the algorithm visible by requiring you to call `comm.reduce_scatter()` explicitly, which is pedagogically useful: you see exactly which transfer primitives fire and can count them.

## Summary

Chapter 48 covered:

- **reduce_scatter ring algorithm:** N-1 transfers per rank, each rank receives 1/N of the reduced output.
- **FSDP gradient reduction:** reduce_scatter is the core collective in FSDP's backward pass.
- **`reduce_scatter_step_count` metric:** counts the number of ring steps per collective call; verifies the N-1 lower bound.
- **Scaling to 8 GPUs:** step count grows linearly with world_size (N-1 steps).
- **Real hardware:** `ncclReduceScatter` maps 1:1 to the ring algorithm in NCCL for large messages.

Chapter 49 covers blocking point-to-point `send`/`recv` and pipeline parallelism.
