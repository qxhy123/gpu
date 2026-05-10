# Chapter 50 — PyTorch Distributed API Wrapper (gpusim.dist)

## The gpusim.dist Module

Phase 12 ships `gpusim.dist` — a module that mirrors the `torch.distributed` API surface, letting you write distributed training code in the simulator using the exact same function names you would use with real PyTorch. The goal is pedagogical: you can study what each collective operation does and how much NVLink bandwidth it consumes without needing a GPU cluster.

The module is **numpy-first**: all collective functions accept `numpy.ndarray` directly. If PyTorch is installed, the same functions also accept `torch.Tensor` via a lazy import fallback — the module never imports `torch` at the top level, so the simulator runs correctly even on machines where torch is not installed.

The full API mirrors `torch.distributed`:

| `gpusim.dist` | `torch.distributed` |
|---|---|
| `init_process_group(world_size, rank)` | `dist.init_process_group(backend, rank, world_size)` |
| `destroy_process_group()` | `dist.destroy_process_group()` |
| `get_rank()` | `dist.get_rank()` |
| `get_world_size()` | `dist.get_world_size()` |
| `all_reduce(tensor, op)` | `dist.all_reduce(tensor, op)` |
| `all_gather(tensor_list, tensor)` | `dist.all_gather(tensor_list, tensor)` |
| `reduce_scatter(output, input_list, op)` | `dist.reduce_scatter(output, input_list, op)` |
| `broadcast(tensor, src)` | `dist.broadcast(tensor, src)` |
| `send(tensor, dst)` | `dist.send(tensor, dst)` |
| `recv(tensor, src)` | `dist.recv(tensor, src)` |
| `barrier()` | `dist.barrier()` |

## The pytorch_dist_simple Demo

The `pytorch_dist_simple` example demonstrates the complete lifecycle: init → collective → barrier → destroy:

```python
import numpy as np
import gpusim.dist as dist

dist.init_process_group(world_size=4, rank=0)

loss = np.full(8, 1.0, dtype=np.float32)
print(f"Before all_reduce: loss[0:4] = {list(loss[0:4])}")

dist.all_reduce(loss, op="sum")
print(f"After all_reduce:  loss[0:4] = {list(loss[0:4])}")   # → [4.0, 4.0, 4.0, 4.0]

dist.barrier()
dist.destroy_process_group()
```

Run it:

```bash
python examples/pytorch_dist_simple/run.py
```

Under the hood, `init_process_group` creates a `MultiGpuSystem` with `n_gpus=world_size`, instantiates a `Comm` object at the requested rank, and stores both in module-level state. All subsequent `dist.*` calls delegate to that `Comm` instance. `destroy_process_group` resets everything to zero.

The module state is intentionally global — exactly like `torch.distributed`'s process group state — so you can call `dist.all_reduce(...)` from anywhere in your script without threading the `comm` object through your call stack.

## 看模拟器

**通过 `dist_api_call_breakdown` 统计各个 collective 的调用频次：**

After running a workload that calls multiple collective operations, you can use the `dist_api_call_breakdown` metric to see the distribution of collective types:

```python
import numpy as np
import gpusim.dist as dist
from gpusim.trace.recorder import Recorder
from gpusim.analysis.metrics import dist_api_call_breakdown

rec = Recorder()

dist.init_process_group(world_size=4, rank=0)
dist._comm._recorder = rec   # attach recorder to the internal Comm object

# Run a training-step sequence
loss = np.full(16, 1.0, dtype=np.float32)
dist.all_reduce(loss, op="sum")              # gradient sync

grads = np.full(64, 1.0, dtype=np.float32)
output = np.zeros(16, dtype=np.float32)
input_list = [np.full(16, 1.0, dtype=np.float32) for _ in range(4)]
dist.reduce_scatter(output, input_list, op="sum")   # FSDP gradient shard

params = np.arange(32, dtype=np.float32)
dist.broadcast(params, src=0)               # parameter sync

dist.barrier()
dist.destroy_process_group()

# Analyze collective call distribution
breakdown = dist_api_call_breakdown(rec.to_collective_df())
print("Collective breakdown:", breakdown)
# → {"allreduce": 1, "reduce_scatter": 1, "broadcast": 1}
```

The `dist_api_call_breakdown` metric returns a dict of `op_name → call_count`. In a full training step with FSDP you would typically see many `reduce_scatter` calls (one per parameter group) and a smaller number of `broadcast` calls (for parameter re-synchronization after optimizer updates). This metric helps you identify which collective type dominates your training step's communication budget.

## 改一改

**尝试用 torch.Tensor 代替 numpy.ndarray（如果已安装 torch）：**

If PyTorch is installed on your machine, `gpusim.dist` accepts `torch.Tensor` directly via the lazy import path:

```python
try:
    import torch
    import gpusim.dist as dist

    dist.init_process_group(world_size=4, rank=0)
    t = torch.full((16,), 1.0, dtype=torch.float32)
    print(f"Before all_reduce: {t[0:4].tolist()}")

    dist.all_reduce(t, op="sum")
    print(f"After all_reduce:  {t[0:4].tolist()}")   # → [4.0, 4.0, 4.0, 4.0]

    dist.destroy_process_group()

except ImportError:
    print("torch not installed — skipping torch tensor path")
```

Internally, `gpusim.dist._to_numpy(t)` detects the tensor type and calls `.numpy()` to convert it before passing to the `Comm` layer. After the collective completes, `_copy_back` calls `t.copy_(torch.from_numpy(result))` to write results back to the original torch tensor. The simulator never imports `torch` unless your code does, keeping the import chain clean.

Try passing a Python list to see the guard in action:

```python
dist.init_process_group(world_size=4, rank=0)
try:
    dist.all_reduce([1.0, 2.0, 3.0], op="sum")   # should raise TypeError
except TypeError as e:
    print(f"Caught expected error: {e}")
dist.destroy_process_group()
```

The module explicitly rejects plain Python lists and any other non-array type with a clear `TypeError`, matching PyTorch's behavior.

## 真机对照

The `gpusim.dist` API is designed to be a drop-in stub for `torch.distributed` in educational contexts:

```python
# Real PyTorch distributed training — identical pattern
import torch
import torch.distributed as dist

dist.init_process_group(
    backend="nccl",       # NCCL backend for GPU-to-GPU communication
    rank=rank,
    world_size=world_size,
)

loss = torch.full((8,), 1.0, dtype=torch.float32, device="cuda")
dist.all_reduce(loss, op=dist.ReduceOp.SUM)   # NCCL allreduce over NCCL communicator

dist.barrier()
dist.destroy_process_group()
```

| Aspect | `gpusim.dist` | `torch.distributed` (NCCL backend) |
|---|---|---|
| **Backend** | `MultiGpuSystem` + `Comm` + NVLink fabric | NCCL over NVLink / InfiniBand |
| **Process model** | Single process, simulated ranks | One process per rank (multi-process launch) |
| **all_reduce algorithm** | Delegates to `Comm.allreduce` (ring or tree) | NCCL auto-selects ring / tree / NVLS based on message size |
| **Tensor type** | `numpy.ndarray` (primary), `torch.Tensor` (optional) | `torch.Tensor` on CUDA device |
| **init_process_group** | Sets rank/world_size in module state | Forks N processes, opens NCCL communicators |
| **barrier** | No-op (single process; no real synchronization needed) | Global barrier: all ranks must arrive before any proceeds |
| **Measurement** | `dist_api_call_breakdown` metric via `Recorder` | `ncu --metrics` or `torch.profiler` |

The most important conceptual difference is the process model: in real distributed training you launch N separate processes (one per GPU), and `torch.distributed` coordinates between them using NCCL over NVLink or InfiniBand. In the simulator, everything runs in a single Python process and the "communication" is modeled as NVLink transfer latency simulation in the `NvlinkFabric`. The API surface stays the same so your code structure transfers directly.

When you graduate from the simulator to real hardware, you will:

1. Replace `import gpusim.dist as dist` with `import torch.distributed as dist`.
2. Add `device = torch.device("cuda", rank)` and move tensors to GPU.
3. Replace `dist.init_process_group(world_size=..., rank=...)` with the full `torch.distributed.init_process_group(backend="nccl", init_method="env://", ...)` call.
4. Launch with `torchrun --nproc_per_node=N your_script.py` instead of a single `python` invocation.

Everything else — the `all_reduce`, `reduce_scatter`, `broadcast`, `send`, `recv`, `barrier`, `destroy_process_group` calls — stays byte-for-byte identical.

## Phase 12 Summary

Chapters 48–50 complete Phase 12's NCCL completion:

- **Chapter 48**: `Comm.reduce_scatter` (ring algorithm) — N-1 transfers per rank; FSDP gradient sharding; `reduce_scatter_step_count` metric.
- **Chapter 49**: `Comm.send` / `Comm.recv` (blocking P2P) — pipeline parallel activation transfer; NVLink transfer events; GPipe / PipeDream patterns.
- **Chapter 50** ⭐: `gpusim.dist` module — PyTorch-distributed-compatible API; numpy-first with optional torch; `dist_api_call_breakdown` metric; drop-in pattern for real `torch.distributed` code.

Together with Chapters 40–43 (multi-GPU setup, ring allreduce, tree allreduce, DDP) and Chapters 44–47 (CUDA Graphs), gpusim now covers the full distributed training communication stack from raw NVLink transfers up to PyTorch-compatible collective APIs.
