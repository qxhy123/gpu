# gpusim Phase 12 — NCCL Completion (reduce_scatter + send/recv + PyTorch dist wrapper)

> **Status:** Brainstormed 2026-05-10. Implementation TBD.

## 1. Goals & Non-goals

### Goals
- Add **`Comm.reduce_scatter(send_buf, recv_buf, op="sum")`** using ring algorithm (FSDP standard).
- Add **`Comm.send(buf, dst_rank)` / `Comm.recv(buf, src_rank)`** blocking point-to-point primitives.
- Add **`gpusim.dist`** PyTorch-distributed adapter: `init_process_group`, `all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`, `send`, `recv`, `barrier`.
- 3 examples + 3 tutorial chapters covering FSDP, pipeline parallelism, dist wrapper basic.
- 2 new metrics: `reduce_scatter_step_count`, `dist_api_call_breakdown`.
- Reuse Phase 10 HTML §33/§34 + Perfetto NVLink swimlane (no new viz sections).
- 100% backward compatible: Phase 1-11 unchanged.

### Non-goals (deferred to Phase 13+)
- Async send/recv with `Work` object (blocking only in Phase 12).
- Process groups beyond default (no `cudaCommSplit`).
- Torch tensor as required input (numpy arrays accepted; torch is optional dependency).
- NVSwitch full congestion model (still point-to-point).
- DistributedDataParallel module (just the dist API).

---

## 2. Architecture

```
gpusim.comm.Comm (Phase 10) — NEW methods:
├── reduce_scatter(send_buf, recv_buf, op) → ring algorithm
├── send(buf, dst_rank) → blocking NVLink transfer
└── recv(buf, src_rank) → blocking NVLink receive

gpusim.dist (NEW gpusim/dist/__init__.py):
├── init_process_group(world_size, rank, n_gpus) → sets up MultiGpuSystem + Comm
├── get_rank() / get_world_size()
├── all_reduce(tensor, op="sum")
├── all_gather(tensor_list, tensor)
├── reduce_scatter(output, input_list, op="sum")
├── broadcast(tensor, src=0)
├── send(tensor, dst)
├── recv(tensor, src)
└── barrier()

Module-level state:
├── _system: MultiGpuSystem | None
├── _comm: Comm | None
└── _world_size, _rank: int
```

### Key invariants
- **Numpy-first**: dist API accepts numpy ndarrays. If `torch` is installed and tensor passed, lazy-import + convert. Torch is NOT a hard dependency.
- **Blocking semantics**: send/recv block until matched. send/recv pair completes when both calls have run.
- **Reduce_scatter ring**: same N-1 step pattern as ring allreduce's scatter-reduce phase.
- **Session-level dist state**: `init_process_group` sets module-level `_system`, `_comm`, `_world_size`, `_rank`. `destroy_process_group()` clears.
- Phase 1-11 unchanged.

---

## 3. Data model

### 3.1 Comm extensions (`gpusim/comm/comm.py`)

```python
class Comm:
    # ... existing fields ...
    
    def reduce_scatter(self, send_buf, recv_buf, op: str = "sum") -> int:
        """Reduce_scatter: each rank gets one chunk of the reduced result.
        Ring algorithm: N-1 transfers per rank (scatter-reduce phase only)."""
        n = self.world_size
        chunk_size_bytes = max(1, send_buf.nbytes // n)
        cycle = 0
        for step in range(n - 1):
            dst = (self.rank + 1) % n
            cycle = self.system.nvlink_fabric.transfer(
                src_gpu=self.rank, dst_gpu=dst,
                n_bytes=chunk_size_bytes, arrival_cycle=cycle,
            )
        # Functional: each rank gets its chunk of the reduced (summed) input
        chunk_n = max(1, send_buf.size // n)
        if op == "sum":
            recv_buf[:chunk_n] = send_buf[self.rank * chunk_n:(self.rank + 1) * chunk_n] * n
        else:
            recv_buf[:chunk_n] = send_buf[self.rank * chunk_n:(self.rank + 1) * chunk_n]
        if self._recorder is not None:
            self._recorder.collective(
                op_name="reduce_scatter", algorithm="ring",
                n_bytes=send_buf.nbytes, world_size=n,
                start_cycle=0, end_cycle=cycle, n_steps=n - 1,
            )
        return cycle
    
    def send(self, buf, dst_rank: int) -> int:
        """Blocking send. Returns completion cycle."""
        cycle = self.system.nvlink_fabric.transfer(
            src_gpu=self.rank, dst_gpu=dst_rank,
            n_bytes=buf.nbytes, arrival_cycle=0,
            recorder=self._recorder, rank=self.rank, op_name="send",
        )
        return cycle
    
    def recv(self, buf, src_rank: int) -> int:
        """Blocking recv. In simulator, recv is paired with sender's transfer.
        Phase 12 simplification: recv just records arrival event without
        re-doing the transfer (sender's transfer already accounts for it)."""
        # No NVLink transfer here — sender did the actual work.
        # Buffer contents in real NCCL would be populated by sender.
        # In simulator: caller is expected to manage buffer.
        if self._recorder is not None:
            # Record a paired event (for trace symmetry)
            pass
        return 0
```

### 3.2 New `gpusim/dist/__init__.py` module

```python
"""Phase 12: PyTorch-distributed-equivalent adapter.

Allows `import gpusim.dist as dist` then writing PyTorch DDP-style code:
    dist.init_process_group(world_size=4, rank=0)
    dist.all_reduce(tensor, op="sum")

Numpy-first: accepts numpy ndarrays. If torch is installed and a tensor is
passed, lazy-imports and converts via .numpy() / from_numpy().
"""
from __future__ import annotations
import numpy as np


_system = None
_comm = None
_world_size = 0
_rank = 0


def init_process_group(world_size: int, rank: int, n_gpus: int = None,
                         config=None) -> None:
    global _system, _comm, _world_size, _rank
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.comm.comm import Comm
    from gpusim.config.loader import load_default
    if config is None:
        config = load_default()
    if n_gpus is None:
        n_gpus = world_size
    config.n_gpus = n_gpus
    _system = MultiGpuSystem.from_config(config)
    _comm = Comm(rank=rank, world_size=world_size, system=_system)
    _world_size = world_size
    _rank = rank


def destroy_process_group() -> None:
    global _system, _comm, _world_size, _rank
    _system = None
    _comm = None
    _world_size = 0
    _rank = 0


def get_rank() -> int:
    return _rank


def get_world_size() -> int:
    return _world_size


def _to_numpy(t):
    """Convert tensor to numpy if torch installed; otherwise expect ndarray."""
    if isinstance(t, np.ndarray):
        return t
    try:
        import torch
        if isinstance(t, torch.Tensor):
            return t.numpy()
    except ImportError:
        pass
    raise TypeError(f"expected numpy.ndarray or torch.Tensor, got {type(t)}")


def _copy_back(t, arr):
    """Write arr contents back into tensor in place."""
    if isinstance(t, np.ndarray):
        t[:] = arr
        return
    try:
        import torch
        if isinstance(t, torch.Tensor):
            t.copy_(torch.from_numpy(arr))
            return
    except ImportError:
        pass


def all_reduce(tensor, op: str = "sum") -> None:
    arr = _to_numpy(tensor)
    recv = np.empty_like(arr)
    _comm.allreduce(arr, recv, op=op)
    _copy_back(tensor, recv)


def all_gather(tensor_list, tensor) -> None:
    """tensor_list: output list of N tensors; tensor: this rank's contribution."""
    arr = _to_numpy(tensor)
    recv = np.empty(arr.size * _world_size, dtype=arr.dtype)
    _comm.allgather(arr, recv)
    chunk = arr.size
    for i, t in enumerate(tensor_list):
        _copy_back(t, recv[i*chunk:(i+1)*chunk].reshape(arr.shape))


def reduce_scatter(output, input_list, op: str = "sum") -> None:
    """output: this rank's output chunk; input_list: full per-rank inputs."""
    # Concatenate input_list into single buffer
    arrs = [_to_numpy(t) for t in input_list]
    full = np.concatenate(arrs)
    recv = np.empty_like(_to_numpy(output))
    _comm.reduce_scatter(full, recv, op=op)
    _copy_back(output, recv)


def broadcast(tensor, src: int = 0) -> None:
    arr = _to_numpy(tensor)
    _comm.broadcast(arr, root=src)
    _copy_back(tensor, arr)


def send(tensor, dst: int) -> None:
    arr = _to_numpy(tensor)
    _comm.send(arr, dst_rank=dst)


def recv(tensor, src: int) -> None:
    arr = _to_numpy(tensor)
    _comm.recv(arr, src_rank=src)


def barrier() -> None:
    """No-op in simulator (single-process); included for API completeness."""
    pass
```

### 3.3 Trace events
Reuse Phase 10 `CollectiveOp` event with `op_name="reduce_scatter" | "send" | "recv"`.

---

## 4. Algorithms

### 4.1 reduce_scatter (ring)
N-1 transfers per rank. Each step transfers `chunk = total_bytes / N`. Result: each rank gets `total_bytes / N` of the reduced output.

### 4.2 send / recv (blocking)
- `send(buf, dst)`: 1 NVLink transfer from `self.rank` → `dst`.
- `recv(buf, src)`: paired with sender; in single-process simulator, no actual data movement (caller already shares numpy buffer).

---

## 5. Trace + Analysis

### 5.1 New metrics

```python
def reduce_scatter_step_count(collective_df) -> dict:
    """Per-call step count for reduce_scatter ops."""

def dist_api_call_breakdown(collective_df) -> dict:
    """Frequency of each dist API call: {"all_reduce": N, "broadcast": M, ...}"""
```

### 5.2 Result API
Reuse Phase 10 `MultiGpuResult` (no new class needed).

---

## 6. Viz

Reuse Phase 10 HTML §33/§34 + Perfetto NVLink/Collective swimlanes — `reduce_scatter` and `send/recv` events flow through existing infrastructure.

---

## 7. Examples (3)

### 7.1 `reduce_scatter_fsdp/`
- 4-GPU FSDP-style: each rank holds 1/4 of model parameters; reduce_scatter gradients.
- **Verifies**: each rank gets correct slice of reduced gradients.

### 7.2 `send_recv_pipeline_parallel/`
- Pipeline parallelism: rank 0 sends activation to rank 1, rank 1 to rank 2, etc.
- **Verifies**: pairwise data flow correct; cycles match per-link transfer expectation.

### 7.3 `pytorch_dist_simple/`
- Use `gpusim.dist as dist` API: `init_process_group → all_reduce → barrier`.
- **Verifies**: PyTorch-style code works against gpusim backend.

---

## 8. Tutorials

`docs/tutorial/` chapters 48-50:
- **48-reduce-scatter-fsdp.md** — example 1
- **49-send-recv-pipeline-parallel.md** — example 2
- **50-pytorch-dist-wrapper.md** — example 3 ⭐

---

## 9. Testing strategy

### Unit tests (~10 new)
- `tests/unit/comm/test_reduce_scatter.py` — ring algorithm step count + correctness
- `tests/unit/comm/test_send_recv.py` — blocking pair + cycle math
- `tests/unit/dist/test_init_process_group.py` — module state setup/destroy
- `tests/unit/dist/test_dist_all_reduce.py` — wrapper correctness with numpy
- `tests/unit/dist/test_dist_torch_optional.py` — torch optional path
- `tests/unit/analysis/test_phase12_metrics.py` — 2 new metrics

### Parity tests (~3 — one per example)

### Microbench
- `test_phase12_facts.py` (fast):
  - reduce_scatter step count = N-1 for N ranks
  - send/recv cycles match single NVLink transfer
- `test_phase12_runtime.py` (slow): 3 examples each under 60s

### Regression
- Rename `tests/parity/test_phase1_10_examples_unchanged.py` → `test_phase1_11_examples_unchanged.py`
- Add 4 Phase 11 examples to the regression list

### Test count target
635 (Phase 11 baseline) → ~660 (+25).

---

## 10. Milestones

| Milestone | Scope | Tag |
|---|---|---|
| **M1** Comm.reduce_scatter (ring) + 1 example | Comm.reduce_scatter + reduce_scatter_fsdp | `M1-phase12-complete` |
| **M2** Comm.send/recv + 1 example | Comm.send/recv + send_recv_pipeline_parallel | `M2-phase12-complete` |
| **M3** gpusim.dist module + 1 example | dist API + pytorch_dist_simple | `M3-phase12-complete` |
| **M4** 2 metrics + analysis | reduce_scatter_step_count + dist_api_call_breakdown | `M4-phase12-complete` |
| **M5** Tutorials + microbench + regression + README v12 + ship | 3 chapters + microbench + Phase 1-11 regression rename + README | `phase12-complete` |

Estimated 18 tasks total.

---

## 11. File list

### New files
```
gpusim/dist/__init__.py
examples/reduce_scatter_fsdp/    # 5 files
examples/send_recv_pipeline_parallel/    # 5 files
examples/pytorch_dist_simple/    # 5 files
docs/tutorial/48-reduce-scatter-fsdp.md
docs/tutorial/49-send-recv-pipeline-parallel.md
docs/tutorial/50-pytorch-dist-wrapper.md
tests/unit/comm/test_reduce_scatter.py
tests/unit/comm/test_send_recv.py
tests/unit/dist/__init__.py
tests/unit/dist/test_init_process_group.py
tests/unit/dist/test_dist_all_reduce.py
tests/unit/dist/test_dist_torch_optional.py
tests/unit/analysis/test_phase12_metrics.py
tests/parity/test_reduce_scatter_fsdp.py
tests/parity/test_send_recv_pipeline_parallel.py
tests/parity/test_pytorch_dist_simple.py
tests/microbench/test_phase12_facts.py
tests/microbench/test_phase12_runtime.py
tests/reference/data/{3 example names}.ref.json
```

### Modified files
```
gpusim/comm/comm.py             # +reduce_scatter + send + recv
gpusim/__init__.py              # + dist module re-export
gpusim/analysis/metrics.py      # +2 metrics
tests/parity/test_phase1_10_examples_unchanged.py → test_phase1_11_examples_unchanged.py
tests/reference/gen_reference.py # +3 kernel names
README.md                        # v12 — Phase 12 capabilities
```

---

## 12. Backward compatibility

- All Phase 1-11 examples + tests pass unchanged.
- New `Comm.reduce_scatter/send/recv` are additive.
- New `gpusim.dist` module is opt-in.
- Torch is optional — `import torch` only happens lazily in dist wrapper helpers.

---

## 13. Open questions / future work

- **Async send/recv** — Phase 13: returns `Work` handle for `wait()`.
- **Process groups** — Phase 13: `dist.new_group(ranks=[0,1])`.
- **DistributedDataParallel module** — Phase 13: `nn.parallel.DistributedDataParallel` wrapper.

---

## 14. Acceptance criteria

Phase 12 ships when:

- [ ] All 5 milestone tags present (`M1-phase12-complete` ... `M4-phase12-complete`, `phase12-complete`)
- [ ] All 3 examples run cleanly
- [ ] All 3 parity tests pass
- [ ] Microbench: reduce_scatter step count = N-1
- [ ] gpusim.dist works with both numpy ndarrays AND torch tensors (when torch installed)
- [ ] Phase 1-11 regression test (renamed) passes
- [ ] Test count: 635 → ~660 (+25)
- [ ] README v12 documents Phase 12 capabilities
