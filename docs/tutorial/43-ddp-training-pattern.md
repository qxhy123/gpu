# Chapter 43 — DDP Training Pattern

## Distributed Data Parallel at a Glance

Distributed Data Parallel (DDP) is the dominant multi-GPU training pattern in PyTorch. The idea is straightforward: split the minibatch equally across N GPUs, run the same forward and backward pass on each rank independently, synchronize gradients with allreduce, and apply the same optimizer update on all ranks so that weights remain identical.

Phase 10's capstone example, `ddp_training_step`, demonstrates the full communication pattern: per-rank forward compute, gradient allreduce, and weight broadcast — all within a single simulated training step.

## The Three Communication Events in One Step

A DDP training step produces exactly three communication events per optimizer cycle:

1. **Forward pass**: Each rank runs its shard of the minibatch through the model. No communication needed — each GPU has a full copy of the weights and only needs its own data shard. This is pure compute.

2. **Gradient allreduce**: After backward, each rank holds locally computed gradients. These must be summed across all ranks so every GPU sees the globally averaged gradient. This is the expensive communication step; it blocks the optimizer until complete.

3. **Weight broadcast**: After the optimizer updates weights on rank 0, the new weights must be pushed to all other ranks to maintain parameter synchronicity. In practice PyTorch DDP uses allreduce (rather than broadcast) to update weights in-place, but the effect is the same: all ranks end with identical parameter tensors.

The simulator example runs all three phases sequentially for clarity:

```python
# Forward: each rank computes gradients via its kernel
for rank in range(4):
    s = Stream()
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": grads},
              kernel_name=f"compute_rank{rank}", config=cfg)
    sys.gpus[rank].run_streams([s])

# Gradient allreduce (auto-picks ring or tree)
comm0 = Comm(rank=0, world_size=4, system=sys)
grads_reduced = np.zeros(n, dtype=np.float32)
cycles_ar = comm0.allreduce(grads, grads_reduced, op="sum")

# Weight broadcast from rank 0 to all others
cycles_bc = comm0.broadcast(weights, root=0)
```

## Communication Volume Analysis

For a model with P parameters (each float32 = 4 bytes), one DDP step sends:

- **Allreduce**: 2*(N-1)/N × 4P bytes of NVLink traffic per GPU (ring algorithm, large P). For N=4 and P=10^9 (a 4B parameter model): ≈6 GB of NVLink traffic per step.
- **Broadcast**: (N-1) × 4P bytes from rank 0. For the same scale: ≈12 GB.

Total NVLink traffic per DDP step ≈ 2*(N-1)/N × 4P + (N-1) × 4P bytes. At 900 GB/s aggregate NVLink bandwidth on H100, a 1B-parameter DDP step over 8 GPUs takes approximately 6.6 ms of communication — fast enough to overlap with compute for typical batch sizes.

The simulator captures this via `per_rank_communication_volume`:

```python
from gpusim.trace.recorder import Recorder
comm = Comm(rank=0, world_size=4, system=sys)
comm._recorder = rec
# ... run allreduce + broadcast ...
for ev in rec.collective_events:
    volume_bytes = ev.n_bytes * ev.n_steps
    print(f"{ev.op_name} ({ev.algorithm}): {volume_bytes} bytes, {ev.n_steps} steps")
```

## Running the DDP Demo

```bash
python examples/ddp_training_step/run.py
```

Expected output:
```
Allreduce cycles: <N>
Broadcast cycles: <M>
Reduced gradients[0:4]: [<values>]
```

The allreduce cycles reflect the ring or tree path depending on gradient tensor size. The broadcast cycles reflect a linear fan-out from rank 0.

## 看模拟器

**对每个通信阶段分别计时：**

Attach a recorder before each collective to isolate timing per phase:

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

grads = np.ones(128, dtype=np.float32)  # 512 bytes — tree path
weights = np.zeros(128, dtype=np.float32)
grads_reduced = np.zeros(128, dtype=np.float32)

cycles_ar = comm.allreduce(grads, grads_reduced, op="sum")
cycles_bc = comm.broadcast(weights, root=0)

ar_ev = rec.collective_events[0]
bc_ev = rec.collective_events[1]
total = cycles_bc  # broadcast starts after allreduce

print(f"Allreduce: {ar_ev.end_cycle} cycles, algo={ar_ev.algorithm}")
print(f"Broadcast: {bc_ev.end_cycle - ar_ev.end_cycle} cycles")
print(f"Allreduce fraction: {ar_ev.end_cycle / total:.1%}")
```

For small gradient tensors (below 4096 bytes), allreduce uses tree (4 steps, log2(4)=2 rounds). Broadcast always uses a linear fan-out (N-1 steps). As gradient size grows, allreduce dominates total communication time and ring becomes more efficient.

## 改一改

**测试 PyTorch DDP 的梯度分桶效果：**

PyTorch DDP groups parameters into buckets (default 25 MB) and overlaps allreduce of early buckets with backward computation of later layers. Simulate this overlap by interleaving compute and communication:

```python
n_buckets = 4
bucket_size = 64  # elements per bucket

for bucket_id in range(n_buckets):
    # Simulate backward for this bucket (compute phase)
    grad_bucket = np.ones(bucket_size, dtype=np.float32)
    recv_bucket = np.zeros(bucket_size, dtype=np.float32)
    
    # Allreduce this bucket while next bucket's backward runs
    cycles = comm.allreduce(grad_bucket, recv_bucket, op="sum")
    print(f"Bucket {bucket_id}: allreduce done at cycle {cycles}")
```

In this model, buckets overlap serially (the simulator does not yet model true concurrent streams for NVLink). On real H100 hardware with PyTorch DDP, buckets overlap with computation through CUDA stream concurrency, achieving nearly zero communication overhead for large models where compute time exceeds communication time per bucket.

## 真机对照

PyTorch DDP on real H100 hardware works as follows:

```python
# PyTorch DDP — one training step
model = DDP(model, device_ids=[rank])
output = model(input)        # forward — no communication
loss = criterion(output, label)
loss.backward()               # backward — triggers async allreduce per bucket
optimizer.step()              # optimizer — uses already-reduced gradients
```

Key differences from the simulator's sequential model:

| Behavior | Simulator | PyTorch DDP (H100) |
|---|---|---|
| **Allreduce trigger** | Explicit `comm.allreduce()` call | Autograd hook fires when bucket ready |
| **Overlap** | Sequential compute then comm | Allreduce overlaps with backward |
| **Bucketing** | Single tensor | 25 MB buckets (configurable) |
| **Algorithm** | Ring (>4096B) or tree | NCCL: ring (>256KB) or tree |
| **Weight sync** | Explicit `broadcast` | No broadcast needed (allreduce keeps weights in sync) |

The simulator's `ddp_training_step` example faithfully models the communication volume and algorithm selection, even though it runs phases sequentially. This is sufficient for understanding where time is spent; the overlap optimization is a scheduling concern addressed in future simulator phases.

## Phase 10 Feature Summary

Chapters 40–43 have covered Phase 10's multi-GPU additions:

- **Chapter 40**: `MultiGpuSystem` + `NvlinkFabric` — N-GPU system modeling with all-to-all NVLink topology.
- **Chapter 41**: `Comm._allreduce_ring` — bandwidth-optimal 2*(N-1)-step ring allreduce for large messages.
- **Chapter 42**: `Comm._allreduce_tree` — latency-optimal 2*log2(N)-step tree allreduce for small messages, auto-selected below 4096 bytes.
- **Chapter 43**: `ddp_training_step` — the complete DDP communication pattern: forward compute, gradient allreduce, and weight broadcast.

Phase 10 completes the simulator's collective communication layer. Combined with the trace recorder (Chapters 33–38) and event timing (Chapter 39), you can now measure, visualize, and reason about the communication component of real distributed training workloads.
