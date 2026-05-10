# Chapter 49 — Point-to-Point Send/Recv and Pipeline Parallelism

## Blocking P2P Communication

Not every distributed training pattern requires a collective operation. Pipeline parallelism, for example, requires each pipeline stage to pass activations to the next stage in a strict order. The right primitive here is point-to-point: rank N sends a tensor to rank N+1, and rank N+1 receives it before proceeding.

The simulator implements `Comm.send` and `Comm.recv` as blocking P2P operations:

- `comm.send(buf, dst_rank)` — issues a single NVLink transfer from the current rank to `dst_rank` and returns the completion cycle. The call blocks until the transfer is simulated.
- `comm.recv(buf, src_rank)` — in the simulator, the sender's NVLink transfer already accounts for the data movement. `recv` is therefore a semantic no-op that returns 0 cycles. It is provided for API symmetry with real `torch.distributed.recv`.

This model captures the right structure for pipeline parallelism simulation: the latency of a forward-pass activation transfer is determined entirely by the sender's NVLink transfer, which is cycle-accurate.

## The Pipeline Forward Pass Pattern

In pipeline parallelism (GPipe, PipeDream), the model is split across N pipeline stages. Each stage runs on one or more GPUs. During the forward pass:

1. Stage 0 computes its portion of the model on the minibatch.
2. Stage 0 sends the intermediate activations to Stage 1.
3. Stage 1 receives the activations, runs its layers, then sends to Stage 2.
4. This chain continues to the final stage.

The NVLink transfers between stages are the communication bottleneck — minimizing their latency is crucial for efficient pipeline utilization.

## The send_recv_pipeline_parallel Demo

The `send_recv_pipeline_parallel` example simulates a 4-stage forward pass:

```python
from gpusim.comm.comm import Comm
from gpusim.comm.system import MultiGpuSystem
from gpusim.config.loader import load_default
import numpy as np

cfg = load_default()
cfg.n_gpus = 4
sys = MultiGpuSystem.from_config(cfg)

buf = np.arange(32, dtype=np.float32)   # simulates an activation tensor

print("Pipeline forward pass:")
for rank in range(3):   # 3 transfers: 0→1, 1→2, 2→3
    comm = Comm(rank=rank, world_size=4, system=sys)
    cycles = comm.send(buf, dst_rank=rank + 1)
    print(f"  Rank {rank} → Rank {rank+1}: {cycles} cycles")

    comm_recv = Comm(rank=rank + 1, world_size=4, system=sys)
    comm_recv.recv(buf, src_rank=rank)
```

Run it:

```bash
python examples/send_recv_pipeline_parallel/run.py
```

You will see 3 lines of output, one per pipeline stage boundary. The cycle counts reflect NVLink latency for a 128-byte payload (32 float32 elements × 4 bytes). Each rank-to-rank transfer is independent — the simulator serializes them sequentially in this demo, mimicking the serial blocking behavior of a standard GPipe forward pass.

## 看模拟器

**通过 Recorder 观察每个 rank 的 NVLink 传输事件：**

Attach a `Recorder` to capture the NVLink transfer events for each send:

```python
from gpusim.trace.recorder import Recorder

rec = Recorder()
comm = Comm(rank=0, world_size=4, system=sys)
comm._recorder = rec

buf = np.arange(32, dtype=np.float32)
comm.send(buf, dst_rank=1)

# Exactly 1 NVLink transfer for a single send
print("Transfer count:", len(rec.nvlink_transfer_events))   # → 1
ev = rec.nvlink_transfer_events[0]
print(f"src={ev.src_gpu}, dst={ev.dst_gpu}, n_bytes={ev.n_bytes}")
print(f"op_name={ev.op_name}")   # → "send"
```

The recorder stores one `NvlinkTransferEvent` per `send` call. The `op_name` field distinguishes P2P sends from collective-internal transfers (which are also routed through NVLink). If you run all 3 pipeline sends with the same recorder, you see 3 transfer events — one per stage boundary.

For a longer pipeline with 8 stages you would see 7 transfer events (rank 0→1, 1→2, ..., 6→7). Each event has its own `start_cycle` / `end_cycle` so you can measure the latency of each individual hop independently.

## 改一改

**把流水线扩展到 8 个 stage，观察传输延迟累积：**

Change `cfg.n_gpus = 8` and run a longer pipeline:

```python
cfg.n_gpus = 8
sys = MultiGpuSystem.from_config(cfg)

rec = Recorder()
total_cycles = 0

for rank in range(7):   # 7 stage boundaries
    comm = Comm(rank=rank, world_size=8, system=sys)
    comm._recorder = rec
    cycles = comm.send(buf, dst_rank=rank + 1)
    total_cycles += cycles
    comm_recv = Comm(rank=rank + 1, world_size=8, system=sys)
    comm_recv.recv(buf, src_rank=rank)

print(f"Total pipeline latency: {total_cycles} cycles")
print(f"NVLink transfers: {len(rec.nvlink_transfer_events)}")   # → 7
```

The total pipeline latency scales linearly with the number of stages — each additional stage boundary adds one NVLink hop's worth of latency. This illustrates the fundamental pipeline parallelism tradeoff: more stages means less computation per GPU (good for compute-to-memory ratio) but more inter-stage communication latency (bad for pipeline throughput). Real pipeline schedulers like PipeDream's flush-free schedule overlap communication with computation from the next micro-batch to hide this latency.

You can also experiment with different activation buffer sizes to see how NVLink bandwidth caps the maximum transfer throughput per stage.

## 真机对照

On real hardware, PyTorch uses `torch.distributed.send` and `torch.distributed.recv` which map to NCCL point-to-point operations:

```python
import torch
import torch.distributed as dist

# Real pipeline parallelism send (blocking)
if rank < world_size - 1:
    dist.send(activation_tensor, dst=rank + 1)

# Real pipeline parallelism recv (blocking)
if rank > 0:
    dist.recv(activation_tensor, src=rank - 1)
```

| Behavior | Simulator (`Comm.send/recv`) | Real PyTorch (`dist.send/recv`) |
|---|---|---|
| **Blocking semantics** | `send` returns completion cycle; `recv` is no-op | Both block until transfer completes |
| **Latency model** | NVLink fabric: bandwidth + base latency | NVLink hardware: wire bandwidth |
| **Transfer event** | `NvlinkTransferEvent` with `op_name="send"` | `ncu` trace: `nvlink_*` counters |
| **Recv cost** | 0 cycles (sender accounts for it) | Receiver pays DMA + kernel overhead |
| **Pipeline framework** | Manual `send`/`recv` in a loop | GPipe's `GpipeSchedule` or PipeDream |

GPipe's implementation breaks each minibatch into micro-batches and pipelines them: while stage K processes micro-batch M, stage K-1 is already processing micro-batch M+1. The simulator lets you study the communication side of this in isolation by running only the send/recv chain without any compute kernels — useful for understanding when NVLink bandwidth, not compute, is the bottleneck.

PipeDream's 1F1B (one-forward, one-backward) schedule is more complex: it interleaves forward and backward passes to overlap communication with computation. The simulator's `Comm.send` / `Comm.recv` primitives provide the building blocks for modeling either schedule.

## Summary

Chapter 49 covered:

- **Blocking P2P semantics:** `Comm.send` issues a real NVLink transfer and returns the completion cycle; `Comm.recv` is a no-op (sender accounts for the data movement).
- **Pipeline forward pass:** 3 sends for a 4-stage pipeline (rank 0→1→2→3).
- **NVLink transfer events per rank:** each send produces one `NvlinkTransferEvent` in the recorder.
- **8-stage extension:** 7 transfer events; total latency scales linearly with pipeline depth.
- **Real hardware:** `torch.distributed.send/recv` maps to NCCL P2P; GPipe and PipeDream schedule these transfers to overlap with compute.

Chapter 50 covers the `gpusim.dist` module — a PyTorch-distributed-compatible API adapter.
