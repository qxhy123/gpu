# gpusim Phase 12 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Implement Phase 12 per `docs/superpowers/specs/2026-05-10-gpusim-phase12-design.md` — NCCL completion (reduce_scatter + send/recv + PyTorch dist wrapper).

**Architecture:** Extend `Comm` with `reduce_scatter`/`send`/`recv`. New `gpusim.dist` module (numpy-first PyTorch-distributed adapter; torch optional via lazy import). Reuse Phase 10 trace events + viz.

**Tech Stack:** Python 3.11+. Optional torch (lazy import).

**Execution note:** Plan has 5 milestones (M1–M5) with 18 tasks. Tags after each: `M{1..5}-phase12-complete`.

---

## Phase 1+2+3+4+5+6+7+8+9+10+11 prerequisites

```bash
cd /Users/yangyang/ai_projs/gpu
git tag | grep phase
.venv/bin/pytest -q -m "not slow"
```
Expected: ~635 passed (Phase 11 baseline).

---

## File structure

```
gpusim/
├── comm/comm.py                MODIFY: + reduce_scatter + send + recv
├── dist/                       NEW (M3)
│   └── __init__.py
├── analysis/metrics.py         MODIFY (M4): + 2 metrics
└── __init__.py                 MODIFY: + dist re-export

examples/
├── reduce_scatter_fsdp/                NEW (M1)
├── send_recv_pipeline_parallel/        NEW (M2)
└── pytorch_dist_simple/                NEW (M3)

tests/unit/
├── comm/test_reduce_scatter.py NEW (M1)
├── comm/test_send_recv.py      NEW (M2)
├── dist/__init__.py            NEW (M3)
├── dist/test_init_process_group.py     NEW (M3)
├── dist/test_dist_all_reduce.py        NEW (M3)
├── dist/test_dist_torch_optional.py    NEW (M3)
└── analysis/test_phase12_metrics.py    NEW (M4)

tests/parity/test_phase1_11_examples_unchanged.py    RENAME from phase1_10 (M5)
tests/microbench/test_phase12_facts.py               NEW (M5)
tests/microbench/test_phase12_runtime.py             NEW (M5, slow)
tests/reference/data/{3 names}.ref.json              NEW (M5)
docs/tutorial/{48,49,50}-*.md                        NEW (M5)
README.md                                            MODIFY (M5): v12
```

---

## Milestones

| Milestone | Tasks | Tag |
|---|---|---|
| **M1** Comm.reduce_scatter + reduce_scatter_fsdp example | T1–T3 | `M1-phase12-complete` |
| **M2** Comm.send/recv + send_recv_pipeline_parallel example | T4–T6 | `M2-phase12-complete` |
| **M3** gpusim.dist module + pytorch_dist_simple example | T7–T11 | `M3-phase12-complete` |
| **M4** 2 metrics | T12–T13 | `M4-phase12-complete` |
| **M5** Tutorials + microbench + regression rename + README v12 + ship | T14–T18 | `phase12-complete` |

---

## Milestone M1: Comm.reduce_scatter + 1 example

### Task 1: Comm.reduce_scatter (ring algorithm)

**Files:**
- Modify: `gpusim/comm/comm.py`
- Test: `tests/unit/comm/test_reduce_scatter.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_reduce_scatter_step_count():
    """Ring reduce_scatter: N-1 transfers per rank."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    # 32-element input → each rank gets 8 elements
    send = np.full(32, 1.0, dtype=np.float32)
    recv = np.zeros(8, dtype=np.float32)
    cycles = comm.reduce_scatter(send, recv, op="sum")
    assert cycles > 0


def test_reduce_scatter_correctness_sum():
    """Each rank gets its chunk of the sum (1.0 * 4 = 4.0)."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    send = np.full(32, 1.0, dtype=np.float32)
    recv = np.zeros(8, dtype=np.float32)
    comm.reduce_scatter(send, recv, op="sum")
    np.testing.assert_array_equal(recv, np.full(8, 4.0, dtype=np.float32))


def test_reduce_scatter_records_collective_event():
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    rec = Recorder()
    comm = Comm(rank=0, world_size=4, system=sys)
    comm._recorder = rec
    send = np.full(32, 1.0, dtype=np.float32)
    recv = np.zeros(8, dtype=np.float32)
    comm.reduce_scatter(send, recv, op="sum")
    assert rec.collective_events[-1].op_name == "reduce_scatter"
    assert rec.collective_events[-1].algorithm == "ring"
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add reduce_scatter to Comm:**

```python
    def reduce_scatter(self, send_buf, recv_buf, op: str = "sum") -> int:
        """Reduce_scatter: each rank gets one chunk of reduced result. Ring algorithm."""
        n = self.world_size
        chunk_size_bytes = max(1, send_buf.nbytes // n)
        cycle = 0
        for step in range(n - 1):
            dst = (self.rank + 1) % n
            cycle = self.system.nvlink_fabric.transfer(
                src_gpu=self.rank, dst_gpu=dst,
                n_bytes=chunk_size_bytes, arrival_cycle=cycle,
            )
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
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/comm/test_reduce_scatter.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/comm/comm.py tests/unit/comm/test_reduce_scatter.py
git commit -m "feat(comm): Comm.reduce_scatter (ring) for FSDP"
```

---

### Task 2: Example reduce_scatter_fsdp

**Files:**
- Create: `examples/reduce_scatter_fsdp/{reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_reduce_scatter_fsdp.py`

(No kernel.ptx — collective is host-side simulation.)

- [ ] **Step 1: Parity test**

```python
def test_reduce_scatter_fsdp_correctness():
    """4-GPU FSDP: each rank gets 1/4 of reduced gradients."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    grads = np.full(64, 1.0, dtype=np.float32)
    my_chunk = np.zeros(16, dtype=np.float32)
    cycles = comm.reduce_scatter(grads, my_chunk, op="sum")
    np.testing.assert_array_equal(my_chunk, np.full(16, 4.0, dtype=np.float32))
    assert cycles > 0
```

- [ ] **Step 2: reference.py + run.py + README.md + __init__.py**

`reference.py`:
```python
import numpy as np
def reference(grads, world_size, rank):
    chunk = grads.size // world_size
    return grads[rank*chunk:(rank+1)*chunk] * world_size
```

`run.py`:
```python
import numpy as np
from gpusim.comm.comm import Comm
from gpusim.comm.system import MultiGpuSystem
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    grads = np.full(64, 1.0, dtype=np.float32)
    my_chunk = np.zeros(16, dtype=np.float32)
    cycles = comm.reduce_scatter(grads, my_chunk, op="sum")
    print(f"Reduce_scatter: {cycles} cycles")
    print(f"Rank 0 chunk[0:4] = {list(my_chunk[0:4])} (expected [4.0, 4.0, 4.0, 4.0])")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# reduce_scatter_fsdp

Phase 12 demo: 4-GPU FSDP-style reduce_scatter on 256-byte grads.
Each rank gets 1/4 of reduced result. Demonstrates N-1 = 3 NVLink transfers per rank.

## Run
```
python examples/reduce_scatter_fsdp/run.py
```

## Tutorial
docs/tutorial/48-reduce-scatter-fsdp.md
```

`__init__.py` (empty).

- [ ] **Step 3: Run + commit**

```
.venv/bin/pytest tests/parity/test_reduce_scatter_fsdp.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/reduce_scatter_fsdp/ tests/parity/test_reduce_scatter_fsdp.py
git commit -m "feat(examples): reduce_scatter_fsdp — 4-GPU FSDP demo"
```

---

### Task 3: Tag M1

```bash
.venv/bin/pytest -q -m "not slow"
git tag M1-phase12-complete
```

---

## Milestone M2: Comm.send/recv + 1 example

### Task 4: Comm.send + Comm.recv

**Files:**
- Modify: `gpusim/comm/comm.py`
- Test: `tests/unit/comm/test_send_recv.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_send_returns_cycles():
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 2
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=2, system=sys)
    buf = np.arange(64, dtype=np.float32)
    cycles = comm.send(buf, dst_rank=1)
    assert cycles > 0


def test_recv_returns_zero_in_simulator():
    """In simulator, recv is no-op (sender's transfer accounts for it)."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 2
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=1, world_size=2, system=sys)
    buf = np.zeros(64, dtype=np.float32)
    cycles = comm.recv(buf, src_rank=0)
    assert cycles == 0


def test_send_records_nvlink_transfer():
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    cfg = load_default()
    cfg.n_gpus = 2
    sys = MultiGpuSystem.from_config(cfg)
    rec = Recorder()
    comm = Comm(rank=0, world_size=2, system=sys)
    comm._recorder = rec
    buf = np.arange(64, dtype=np.float32)
    comm.send(buf, dst_rank=1)
    assert len(rec.nvlink_transfer_events) == 1
    assert rec.nvlink_transfer_events[0].op_name == "send"
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add send + recv to Comm:**

```python
    def send(self, buf, dst_rank: int) -> int:
        """Blocking send. Returns completion cycle."""
        cycle = self.system.nvlink_fabric.transfer(
            src_gpu=self.rank, dst_gpu=dst_rank,
            n_bytes=buf.nbytes, arrival_cycle=0,
            recorder=self._recorder, rank=self.rank, op_name="send",
        )
        return cycle
    
    def recv(self, buf, src_rank: int) -> int:
        """Blocking recv. In simulator, paired send already did the work."""
        return 0
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/comm/test_send_recv.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/comm/comm.py tests/unit/comm/test_send_recv.py
git commit -m "feat(comm): Comm.send + Comm.recv (blocking P2P)"
```

---

### Task 5: Example send_recv_pipeline_parallel

**Files:**
- Create: `examples/send_recv_pipeline_parallel/{reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_send_recv_pipeline_parallel.py`

- [ ] **Step 1: Parity test**

```python
def test_send_recv_pipeline_parallel_correctness():
    """Pipeline parallelism: rank N sends to rank N+1 in chain."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    
    # Each rank sends activation to next; verify all transfers complete with cycles > 0
    buf = np.arange(32, dtype=np.float32)
    total_cycles = 0
    for rank in range(3):  # 0→1, 1→2, 2→3
        comm = Comm(rank=rank, world_size=4, system=sys)
        cycles = comm.send(buf, dst_rank=rank + 1)
        total_cycles += cycles
        # Receive at next rank (no-op in simulator)
        comm_recv = Comm(rank=rank + 1, world_size=4, system=sys)
        comm_recv.recv(buf, src_rank=rank)
    
    assert total_cycles > 0
```

- [ ] **Step 2: reference.py + run.py + README.md + __init__.py**

`reference.py`:
```python
def reference(): return "pipeline forward pass complete"
```

`run.py`:
```python
import numpy as np
from gpusim.comm.comm import Comm
from gpusim.comm.system import MultiGpuSystem
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    buf = np.arange(32, dtype=np.float32)
    print("Pipeline forward pass:")
    for rank in range(3):
        comm = Comm(rank=rank, world_size=4, system=sys)
        cycles = comm.send(buf, dst_rank=rank + 1)
        print(f"  Rank {rank} → Rank {rank+1}: {cycles} cycles")
        comm_recv = Comm(rank=rank + 1, world_size=4, system=sys)
        comm_recv.recv(buf, src_rank=rank)


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# send_recv_pipeline_parallel

Phase 12 demo: 4-GPU pipeline parallelism. Each rank sends activation to next
in a forward-pass chain (rank 0→1→2→3).

## Run
```
python examples/send_recv_pipeline_parallel/run.py
```

## Tutorial
docs/tutorial/49-send-recv-pipeline-parallel.md
```

`__init__.py` (empty).

- [ ] **Step 3: Run + commit**

```
.venv/bin/pytest tests/parity/test_send_recv_pipeline_parallel.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/send_recv_pipeline_parallel/ tests/parity/test_send_recv_pipeline_parallel.py
git commit -m "feat(examples): send_recv_pipeline_parallel — 4-GPU pipeline forward pass"
```

---

### Task 6: Tag M2

```bash
.venv/bin/pytest -q -m "not slow"
git tag M2-phase12-complete
```

---

## Milestone M3: gpusim.dist module + 1 example

### Task 7: Create gpusim/dist/__init__.py with init_process_group + module state

**Files:**
- Create: `gpusim/dist/__init__.py`
- Create: `tests/unit/dist/__init__.py` (empty)
- Create: `tests/unit/dist/test_init_process_group.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_init_process_group_sets_state():
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)
    assert dist.get_rank() == 0
    assert dist.get_world_size() == 4
    dist.destroy_process_group()
    assert dist.get_world_size() == 0


def test_init_process_group_with_rank_2():
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=2)
    assert dist.get_rank() == 2
    dist.destroy_process_group()
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Create gpusim/dist/__init__.py:**

```python
"""Phase 12: PyTorch-distributed-equivalent adapter."""
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


def barrier() -> None:
    """No-op in simulator (single-process)."""
    pass
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/dist/test_init_process_group.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/dist/ tests/unit/dist/
git commit -m "feat(dist): init_process_group + module state + barrier"
```

---

### Task 8: gpusim.dist.all_reduce

**Files:**
- Modify: `gpusim/dist/__init__.py`
- Test: `tests/unit/dist/test_dist_all_reduce.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_dist_all_reduce_numpy():
    import numpy as np
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)
    t = np.full(16, 1.0, dtype=np.float32)
    dist.all_reduce(t, op="sum")
    np.testing.assert_array_equal(t, np.full(16, 4.0, dtype=np.float32))
    dist.destroy_process_group()


def test_dist_all_gather_numpy():
    import numpy as np
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)
    t = np.full(8, 1.0, dtype=np.float32)
    tensor_list = [np.zeros(8, dtype=np.float32) for _ in range(4)]
    dist.all_gather(tensor_list, t)
    # All gathered tensors should be 1.0 (uniform)
    for tl in tensor_list:
        np.testing.assert_array_equal(tl, np.full(8, 1.0, dtype=np.float32))
    dist.destroy_process_group()


def test_dist_broadcast_numpy():
    import numpy as np
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)
    t = np.arange(8, dtype=np.float32)
    dist.broadcast(t, src=0)
    # broadcast — buffer unchanged at root
    np.testing.assert_array_equal(t, np.arange(8, dtype=np.float32))
    dist.destroy_process_group()


def test_dist_reduce_scatter_numpy():
    import numpy as np
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)
    output = np.zeros(8, dtype=np.float32)
    input_list = [np.full(8, 1.0, dtype=np.float32) for _ in range(4)]
    dist.reduce_scatter(output, input_list, op="sum")
    np.testing.assert_array_equal(output, np.full(8, 4.0, dtype=np.float32))
    dist.destroy_process_group()


def test_dist_send_recv_numpy():
    import numpy as np
    import gpusim.dist as dist
    dist.init_process_group(world_size=2, rank=0)
    t = np.arange(16, dtype=np.float32)
    dist.send(t, dst=1)
    dist.recv(t, src=1)   # dummy paired recv
    dist.destroy_process_group()


def test_dist_barrier_noop():
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)
    dist.barrier()   # should not raise
    dist.destroy_process_group()
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add dist API methods to gpusim/dist/__init__.py:**

```python
def all_reduce(tensor, op: str = "sum") -> None:
    arr = _to_numpy(tensor)
    recv = np.empty_like(arr)
    _comm.allreduce(arr, recv, op=op)
    _copy_back(tensor, recv)


def all_gather(tensor_list, tensor) -> None:
    arr = _to_numpy(tensor)
    recv = np.empty(arr.size * _world_size, dtype=arr.dtype)
    _comm.allgather(arr, recv)
    chunk = arr.size
    for i, t in enumerate(tensor_list):
        _copy_back(t, recv[i*chunk:(i+1)*chunk].reshape(arr.shape))


def reduce_scatter(output, input_list, op: str = "sum") -> None:
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
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/dist/test_dist_all_reduce.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/dist/__init__.py tests/unit/dist/test_dist_all_reduce.py
git commit -m "feat(dist): all_reduce + all_gather + reduce_scatter + broadcast + send + recv"
```

---

### Task 9: Torch optional path test

**Files:**
- Test: `tests/unit/dist/test_dist_torch_optional.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_torch_tensor_path_if_available():
    """If torch is installed, dist API should accept torch tensors."""
    try:
        import torch
    except ImportError:
        import pytest
        pytest.skip("torch not installed — skipping torch-tensor path")
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)
    t = torch.full((16,), 1.0, dtype=torch.float32)
    dist.all_reduce(t, op="sum")
    expected = torch.full((16,), 4.0, dtype=torch.float32)
    assert torch.allclose(t, expected)
    dist.destroy_process_group()


def test_invalid_input_type_raises():
    """Non-numpy non-torch input should raise TypeError."""
    import gpusim.dist as dist
    import pytest
    dist.init_process_group(world_size=4, rank=0)
    with pytest.raises(TypeError):
        dist.all_reduce([1, 2, 3], op="sum")   # plain Python list
    dist.destroy_process_group()
```

- [ ] **Step 2: Run + verify** (torch test may skip if not installed; type test should pass).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/dist/test_dist_torch_optional.py
git commit -m "test(dist): torch optional path + invalid input type validation"
```

---

### Task 10: Example pytorch_dist_simple

**Files:**
- Create: `examples/pytorch_dist_simple/{reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_pytorch_dist_simple.py`

- [ ] **Step 1: Parity test**

```python
def test_pytorch_dist_simple_correctness():
    """Use gpusim.dist API: init → all_reduce → barrier."""
    import numpy as np
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)
    
    # Each rank's "loss"
    loss = np.full(8, 1.0, dtype=np.float32)
    dist.all_reduce(loss, op="sum")
    np.testing.assert_array_equal(loss, np.full(8, 4.0, dtype=np.float32))
    
    dist.barrier()
    dist.destroy_process_group()
```

- [ ] **Step 2: reference.py + run.py + README.md + __init__.py**

`reference.py`:
```python
import numpy as np
def reference(loss, world_size): return loss * world_size
```

`run.py`:
```python
import numpy as np
import gpusim.dist as dist


def main():
    dist.init_process_group(world_size=4, rank=0)
    loss = np.full(8, 1.0, dtype=np.float32)
    print(f"Before all_reduce: loss[0:4] = {list(loss[0:4])}")
    dist.all_reduce(loss, op="sum")
    print(f"After all_reduce:  loss[0:4] = {list(loss[0:4])} (expected [4.0]*4)")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# pytorch_dist_simple

Phase 12 demo: PyTorch-distributed-style code using `gpusim.dist as dist`.
init_process_group → all_reduce → barrier.

## Run
```
python examples/pytorch_dist_simple/run.py
```

## Tutorial
docs/tutorial/50-pytorch-dist-wrapper.md
```

`__init__.py` (empty).

- [ ] **Step 3: Run + commit**

```
.venv/bin/pytest tests/parity/test_pytorch_dist_simple.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/pytorch_dist_simple/ tests/parity/test_pytorch_dist_simple.py
git commit -m "feat(examples): pytorch_dist_simple — gpusim.dist API end-to-end"
```

---

### Task 11: Tag M3

```bash
.venv/bin/pytest -q -m "not slow"
git tag M3-phase12-complete
```

---

## Milestone M4: 2 metrics

### Task 12: 2 analysis metrics

**Files:**
- Modify: `gpusim/analysis/metrics.py`
- Test: `tests/unit/analysis/test_phase12_metrics.py` (NEW)

- [ ] **Step 1: Create test**

```python
import pandas as pd


def test_reduce_scatter_step_count():
    from gpusim.analysis.metrics import reduce_scatter_step_count
    df = pd.DataFrame([
        {"op_name": "reduce_scatter", "algorithm": "ring", "n_steps": 3,
          "n_bytes": 256, "world_size": 4, "start_cycle": 0, "end_cycle": 100},
        {"op_name": "reduce_scatter", "algorithm": "ring", "n_steps": 7,
          "n_bytes": 512, "world_size": 8, "start_cycle": 100, "end_cycle": 200},
    ])
    out = reduce_scatter_step_count(df)
    # Per-call step counts for reduce_scatter ops
    assert 3 in out
    assert 7 in out


def test_dist_api_call_breakdown():
    from gpusim.analysis.metrics import dist_api_call_breakdown
    df = pd.DataFrame([
        {"op_name": "allreduce", "algorithm": "ring"},
        {"op_name": "allreduce", "algorithm": "tree"},
        {"op_name": "broadcast", "algorithm": "linear"},
        {"op_name": "reduce_scatter", "algorithm": "ring"},
    ])
    out = dist_api_call_breakdown(df)
    assert out["allreduce"] == 2
    assert out["broadcast"] == 1
    assert out["reduce_scatter"] == 1
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Append metrics:**

```python
def reduce_scatter_step_count(collective_df) -> dict:
    """Per-call step count for reduce_scatter ops. Returns {n_steps: count}."""
    if collective_df is None or collective_df.empty:
        return {}
    sub = collective_df[collective_df["op_name"] == "reduce_scatter"]
    if sub.empty:
        return {}
    out = {}
    for n in sub["n_steps"]:
        out[int(n)] = out.get(int(n), 0) + 1
    return out


def dist_api_call_breakdown(collective_df) -> dict:
    """Frequency of each collective op_name."""
    if collective_df is None or collective_df.empty:
        return {}
    out = {}
    for name in collective_df["op_name"]:
        out[str(name)] = out.get(str(name), 0) + 1
    return out
```

- [ ] **Step 4: Run + commit + tag M4**

```
.venv/bin/pytest tests/unit/analysis/test_phase12_metrics.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/analysis/metrics.py tests/unit/analysis/test_phase12_metrics.py
git commit -m "feat(analysis): reduce_scatter_step_count + dist_api_call_breakdown"
git tag M4-phase12-complete
```

---

### Task 13: (consolidated into T12)

(Reserved.)

---

## Milestone M5: Tutorials + microbench + ship

### Task 14: 3 tutorial chapters 48-50

**Files:**
- Create: `docs/tutorial/{48,49,50}-*.md`

- [ ] **Step 1: Read** `docs/tutorial/47-graph-iterative-training.md` for style reference.

- [ ] **Step 2: Write 3 chapters (~500-700 words each, English body + Chinese subheadings):**

**Chapter 48 — reduce-scatter-fsdp:**
- Reduce_scatter ring algorithm: N-1 transfers per rank, each rank gets 1/N of reduced output
- reduce_scatter_fsdp demo
- 看模拟器: reduce_scatter_step_count metric
- 改一改: increase n_gpus to 8
- 真机对照: NCCL ncclReduceScatter, FSDP gradient reduction

**Chapter 49 — send-recv-pipeline-parallel:**
- Blocking send/recv P2P semantics
- send_recv_pipeline_parallel demo
- 看模拟器: NVLink transfer events per rank
- 改一改: longer pipeline (8 stages)
- 真机对照: GPipe / PipeDream pipeline parallel training

**Chapter 50 — pytorch-dist-wrapper ⭐:**
- gpusim.dist API: init_process_group → all_reduce → barrier
- pytorch_dist_simple demo
- 看模拟器: dist_api_call_breakdown metric
- 改一改: try with torch.Tensor (if torch installed)
- 真机对照: torch.distributed.init_process_group + torch.distributed.all_reduce

```bash
git add docs/tutorial/48-reduce-scatter-fsdp.md \
        docs/tutorial/49-send-recv-pipeline-parallel.md \
        docs/tutorial/50-pytorch-dist-wrapper.md
git commit -m "docs(tutorial): chapters 48-50 — Phase 12 NCCL completion"
```

---

### Task 15: Phase 12 microbench + 3 ref stubs

**Files:**
- Create: `tests/microbench/test_phase12_facts.py`
- Create: `tests/microbench/test_phase12_runtime.py`
- Modify: `tests/reference/gen_reference.py`
- Create: 3 ref JSONs

- [ ] **Step 1: test_phase12_facts.py:**

```python
"""Phase 12 microbench — NCCL completion facts."""
import numpy as np


def test_reduce_scatter_step_count_n_minus_1():
    """Ring reduce_scatter: N-1 NVLink transfers."""
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    rec = Recorder()
    # Hook recorder into fabric
    original_transfer = sys.nvlink_fabric.transfer
    def traced_transfer(*args, **kwargs):
        kwargs["recorder"] = rec
        return original_transfer(*args, **kwargs)
    sys.nvlink_fabric.transfer = traced_transfer
    
    comm = Comm(rank=0, world_size=4, system=sys)
    comm._recorder = rec
    send = np.full(64, 1.0, dtype=np.float32)
    recv = np.zeros(16, dtype=np.float32)
    comm.reduce_scatter(send, recv, op="sum")
    
    # N-1 = 3 transfers for N=4
    assert len(rec.nvlink_transfer_events) == 3


def test_send_one_transfer():
    """Single send produces single NVLink transfer."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    cfg = load_default()
    cfg.n_gpus = 2
    sys = MultiGpuSystem.from_config(cfg)
    rec = Recorder()
    comm = Comm(rank=0, world_size=2, system=sys)
    comm._recorder = rec
    buf = np.arange(64, dtype=np.float32)
    comm.send(buf, dst_rank=1)
    assert len(rec.nvlink_transfer_events) == 1
```

- [ ] **Step 2: test_phase12_runtime.py:**

```python
import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_reduce_scatter_fsdp_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "reduce_scatter_fsdp"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_pytorch_dist_simple_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "pytorch_dist_simple"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30
```

- [ ] **Step 3: Append 3 kernel names to gen_reference.py:**

```python
"reduce_scatter_fsdp",
"send_recv_pipeline_parallel",
"pytorch_dist_simple",
```

- [ ] **Step 4: Create 3 ref JSONs:**

```bash
for k in reduce_scatter_fsdp send_recv_pipeline_parallel pytorch_dist_simple; do
  cat > tests/reference/data/$k.ref.json <<JSON
{
  "kernel": "$k",
  "phase": 12,
  "metrics": {
    "reduce_scatter_step_count": null,
    "dist_api_call_breakdown": null
  },
  "tolerance": {
    "reduce_scatter_step_count_pct": 0,
    "dist_api_call_breakdown_pct": 0
  },
  "notes": "Generated by gen_reference.py on real Hopper. null = not yet measured."
}
JSON
done
```

- [ ] **Step 5: Run + commit:**

```
.venv/bin/pytest tests/microbench/test_phase12_facts.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add tests/microbench/test_phase12_facts.py tests/microbench/test_phase12_runtime.py \
        tests/reference/gen_reference.py tests/reference/data/reduce_scatter_fsdp.ref.json \
        tests/reference/data/send_recv_pipeline_parallel.ref.json \
        tests/reference/data/pytorch_dist_simple.ref.json
git commit -m "test(microbench+reference): Phase 12 facts + 3 ref stubs"
```

---

### Task 16: Phase 1-11 regression rename

```bash
git mv tests/parity/test_phase1_10_examples_unchanged.py tests/parity/test_phase1_11_examples_unchanged.py
```

Edit:
- Rename `PHASE_1_10_EXAMPLES` → `PHASE_1_11_EXAMPLES`
- Append 4 Phase 11 examples: `graph_explicit_build`, `graph_capture_from_stream`, `graph_replay_perf`, `graph_iterative_train_step`
- Update test function names from `phase_1_10_*` → `phase_1_11_*` if any

```bash
git add tests/parity/test_phase1_11_examples_unchanged.py
git commit -m "test(regression): rename phase1_10 → phase1_11 + 4 Phase 11 examples"
```

---

### Task 17: README v12 + final tag phase12-complete

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update to v12:**
- Phase status: 1-12 ✅
- Phase 12 features section:
  - Comm.reduce_scatter (ring) for FSDP
  - Comm.send + Comm.recv (blocking P2P)
  - gpusim.dist module: init_process_group, all_reduce, all_gather, reduce_scatter, broadcast, send, recv, barrier
  - 2 metrics (reduce_scatter_step_count, dist_api_call_breakdown)
  - Numpy-first; torch optional via lazy import
  - 3 examples + 3 tutorials chapters 48-50
  - Backward compatible: Phase 1-11 unchanged
- Examples list: add 3 (was 46, now 49)
- Tutorials list: add 48-50 (was 47, now 50)

- [ ] **Step 2: Run final suite + 3 examples:**

```
.venv/bin/pytest -q -m "not slow"
.venv/bin/python examples/reduce_scatter_fsdp/run.py
.venv/bin/python examples/send_recv_pipeline_parallel/run.py
.venv/bin/python examples/pytorch_dist_simple/run.py
```

- [ ] **Step 3: Commit + tag:**

```bash
git add README.md
git commit -m "docs(readme): v12 — Phase 12 capabilities (NCCL completion)"
git tag phase12-complete
```

---

### Task 18: Final sanity sweep + done

```
.venv/bin/pytest -q -m "not slow"
.venv/bin/pytest tests/parity/test_phase1_11_examples_unchanged.py -v
```

Phase 12 ships when all tasks complete + tags landed.

---

## End-of-plan checklist

- [ ] M1 (Comm.reduce_scatter + reduce_scatter_fsdp): T1-T3
- [ ] M2 (Comm.send/recv + send_recv_pipeline_parallel): T4-T6
- [ ] M3 (gpusim.dist + pytorch_dist_simple): T7-T11
- [ ] M4 (2 metrics): T12-T13
- [ ] M5 (Tutorials + microbench + regression + README): T14-T18
- [ ] All 5 milestone tags + phase12-complete
- [ ] Phase 1-11 regression unbroken
- [ ] 3 new examples + 3 tutorials shipped
- [ ] README v12 reflects Phase 12
