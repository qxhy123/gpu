# Chapter 41 — Ring Allreduce: Bandwidth-Optimal Collective

## Why Allreduce?

Distributed data-parallel training splits a minibatch across N GPUs. Each GPU computes gradients for its shard independently. Before the optimizer step, all ranks must arrive at the same gradient tensor — the element-wise sum across all ranks. This is allreduce.

A naive implementation (GPU 0 collects from all, reduces, then broadcasts) bottlenecks at GPU 0: it must receive N-1 tensors and transmit N-1 copies, making its link utilization N-1 times higher than everyone else's. The ring algorithm eliminates this asymmetry.

## The Ring Algorithm: 2*(N-1) Steps

Ring allreduce proceeds in two symmetric phases. Arrange the N GPUs in a ring: 0→1→2→…→(N-1)→0.

**Scatter-reduce phase** (N-1 steps): each GPU holds a chunk array of size `buf / N`. In each step every GPU sends its current chunk to the right neighbor and receives a chunk from the left neighbor, accumulating (summing) the received values into a local partial sum. After N-1 steps, every GPU holds the globally-reduced value for exactly one chunk.

**Allgather phase** (N-1 more steps): each GPU now propagates its fully-reduced chunk around the ring. After another N-1 steps every GPU holds the full reduced buffer.

Total transfers per GPU: 2*(N-1) chunks, each of size `buf/N` bytes. Total bytes sent per GPU = 2*(N-1)/N × buf_bytes, which approaches 2 × buf_bytes as N grows. This is optimal: every byte of the output must traverse at least two hops in any collective that achieves all-to-all consistency.

The simulator's `Comm._allreduce_ring` implements this directly:

```python
n = self.world_size
chunk_size_bytes = max(1, send_buf.nbytes // n)
cycle = 0
for step in range(2 * (n - 1)):
    dst = (self.rank + 1) % n
    cycle = self.system.nvlink_fabric.transfer(
        src_gpu=self.rank, dst_gpu=dst,
        n_bytes=chunk_size_bytes, arrival_cycle=cycle,
    )
```

For N=4, this executes 6 transfers in sequence from rank 0's perspective, each of size `buf/4` bytes.

## Running the Ring Demo

```bash
python examples/ring_allreduce/run.py
```

The demo creates a 4-GPU system and performs allreduce on a 256-element float32 array (1024 bytes, above the 4096-byte threshold — wait, 256×4 = 1024 bytes is below threshold. Try 1024 elements for ring path):

```python
cfg = load_default()
cfg.n_gpus = 4
sys = MultiGpuSystem.from_config(cfg)
comm = Comm(rank=0, world_size=4, system=sys)
send = np.full(256, 1.0, dtype=np.float32)   # 1024 bytes
recv = np.zeros(256, dtype=np.float32)
cycles = comm.allreduce(send, recv, op="sum")
print(f"Ring allreduce: {cycles} cycles")
```

Note: 256 floats = 1024 bytes. Since the threshold is 4096 bytes, this actually takes the tree path. To force the ring path, use 1024 or more floats (4096+ bytes).

## 看模拟器

**使用 collective_op_breakdown 分析步骤分布：**

The recorder captures a `CollectiveOp` event for each allreduce call. Inspect it to verify step counts and algorithm selection:

```python
from gpusim.trace.recorder import Recorder
from gpusim.comm.comm import Comm
from gpusim.comm.system import MultiGpuSystem
from gpusim.config.loader import load_default
import numpy as np

cfg = load_default()
cfg.n_gpus = 4
sys = MultiGpuSystem.from_config(cfg)
rec = Recorder()
comm = Comm(rank=0, world_size=4, system=sys)
comm._recorder = rec

send = np.full(1024, 1.0, dtype=np.float32)  # 4096 bytes — ring path
recv = np.zeros(1024, dtype=np.float32)
comm.allreduce(send, recv, op="sum")

ev = rec.collective_events[-1]
print(f"Algorithm: {ev.algorithm}")     # "ring"
print(f"Steps: {ev.n_steps}")           # 6  (= 2*(4-1))
print(f"Total bytes: {ev.n_bytes}")     # 4096
print(f"Cycles: {ev.end_cycle}")
```

The `n_steps` field matches 2*(N-1). The `n_bytes` field is the full tensor size — chunk slicing happens inside the algorithm.

## 改一改

**比较不同 N 下的延迟缩放：**

Ring allreduce latency scales as 2*(N-1)/N × T_link, where T_link is the per-chunk transfer time. For large N this approaches 2 × T_link, meaning latency plateaus rather than growing linearly. Test this:

```python
for n_gpus in [2, 4, 8]:
    cfg = load_default()
    cfg.n_gpus = n_gpus
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=n_gpus, system=sys)
    send = np.full(1024, 1.0, dtype=np.float32)  # 4096 bytes, ring path
    recv = np.zeros(1024, dtype=np.float32)
    cycles = comm.allreduce(send, recv, op="sum")
    print(f"n={n_gpus}: {cycles} cycles, steps={2*(n_gpus-1)}")
```

You should see latency grow sub-linearly because each step transfers a smaller chunk (4096/N bytes) even though there are 2*(N-1) steps. The product 2*(N-1) × (4096/N) approaches 8192 bytes total transferred regardless of N — the bandwidth-optimal property.

## 真机对照

NCCL (NVIDIA Collective Communications Library) uses ring allreduce as its default algorithm for messages larger than 256 KB. Below that threshold it switches to a tree-based algorithm (Chapter 42) to minimize latency at the cost of some bandwidth efficiency.

| Property | Simulator | NCCL (H100 DGX) |
|---|---|---|
| **Algorithm trigger** | `n_bytes >= 4096` | `n_bytes >= 256 KB` (tuned per topology) |
| **Step count** | 2*(N-1) | 2*(N-1) scatter-reduce + allgather |
| **Chunk size** | `buf / N` | `buf / N` with pipeline pipelining |
| **BW efficiency** | 2*(N-1)/N | same asymptotically |

The simulator's 4096-byte threshold is deliberately lower than NCCL's 256 KB crossover for demonstration purposes — it lets you observe both algorithms without needing large arrays. On real H100 hardware, ring allreduce over NVLink at 900 GB/s aggregate takes roughly `2 * buf_bytes / (900e9 / N)` seconds. For a 1 GB gradient tensor across 8 GPUs, that is approximately 1.8 ms — fast enough that communication is rarely the bottleneck for large model training when overlapped with compute.
