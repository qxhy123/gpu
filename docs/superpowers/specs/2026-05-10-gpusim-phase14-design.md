# gpusim Phase 14 — Persistent Kernels + Dynamic Parallelism

> **Status:** Brainstormed 2026-05-10. Implementation TBD.

## 1. Goals & Non-goals

### Goals
- Add **persistent kernels**: long-running kernels that pull work items from a `WorkQueue` until told to stop.
- Add **dynamic parallelism**: parent kernel triggers child kernel launches via host-side `gpusim.device_launch()` API.
- 4 examples + 4 tutorial chapters (54-57).
- 3 new metrics: `persistent_kernel_throughput`, `dynamic_parallelism_depth`, `dynamic_parallelism_fanout`.
- Reuse Phase 7-9 `KernelLaunch` event with new fields `parent_kernel_id` + `is_persistent`.
- 100% backward compatible: Phase 1-13 unchanged.

### Non-goals (deferred to Phase 15+)
- True device-side launch via PTX (`cudaLaunchKernel` from device); we model it as host-side callback.
- Persistent grid resizing (queue-driven `n_threads` change between iterations).
- CUDA Graphs containing persistent kernel nodes.
- Multi-GPU dynamic parallelism (parent + child on different GPUs).

---

## 2. Architecture

```
gpusim.WorkQueue (NEW gpusim/persistent/queue.py):
├── items: deque
├── stopped: bool
├── push(item)
├── pop() → item | None (None when empty)
├── stop()
└── is_empty() / is_stopped()

gpusim.PersistentKernel (NEW gpusim/persistent/kernel.py):
├── ptx_src, grid, block, params_template, work_queue
├── start(config, recorder=None) → list[Result] (one Result per work item)
└── (internally: loops pop()→launch()→pop()→... until queue stopped/empty)

gpusim.device_launch (NEW gpusim/persistent/dynamic.py):
├── device_launch(parent_kernel_id, ptx, grid, block, params, kernel_name)
│   → schedules child kernel (executes after parent retires); records
│     KernelLaunch with parent_kernel_id set
└── _pending_child_launches: list (module-level state, drained by Stream/run)

KernelLaunch event extension (gpusim/trace/events.py):
├── parent_kernel_id: int = -1   (NEW Phase 14: -1 = top-level)
└── is_persistent: bool = False  (NEW Phase 14: True if from PersistentKernel)
```

### Key invariants
- `WorkQueue.pop()` returns `None` once stopped AND empty.
- `PersistentKernel.start()` is blocking: returns after queue stopped + drained.
- `device_launch()` enqueues a child KernelLaunch in module-level state; the next `Stream.synchronize` or `Device.run` drain processes them.
- Persistent kernel records `KernelLaunch` with `is_persistent=True` per work item iteration.
- Dynamic parallelism records `parent_kernel_id` for traceability; depth = max chain (parent → child → grandchild ...).
- Phase 1-13 unchanged.

---

## 3. Data model

### 3.1 WorkQueue

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
    
    def is_empty(self) -> bool:
        return len(self.items) == 0
    
    def is_stopped(self) -> bool:
        return self.stopped
```

### 3.2 PersistentKernel

```python
@dataclass
class PersistentKernel:
    ptx_src: str
    grid: tuple
    block: tuple
    params_template: dict           # base params; per-item params override
    work_queue: WorkQueue
    kernel_name: str = "<persistent>"
    
    def start(self, config, recorder=None) -> list:
        """Block until queue is stopped + empty. Returns Result per work item."""
        from gpusim.api import Stream, synchronize
        results = []
        while True:
            item = self.work_queue.pop()
            if item is None:
                if self.work_queue.is_stopped():
                    break
                # Queue empty but not stopped — would normally wait; for sim, exit
                break
            # Item is a dict that overrides/extends params_template
            params = {**self.params_template, **item}
            s = Stream()
            s.launch(ptx_src=self.ptx_src, grid=self.grid, block=self.block,
                      params=params, kernel_name=self.kernel_name, config=config)
            multi_res = synchronize(streams=[s], config=config)
            if s.stream_id in multi_res.streams and multi_res.streams[s.stream_id]:
                res = multi_res.streams[s.stream_id][0]
                results.append(res)
                if recorder is not None:
                    cycles = res.metrics.get("cycles", 0)
                    recorder.kernel_launch(
                        stream_id=s.stream_id,
                        kernel_name=self.kernel_name,
                        grid=self.grid, block=self.block,
                        launch_cycle=0, complete_cycle=cycles,
                        n_ctas=self.grid[0]*self.grid[1]*self.grid[2],
                        # Phase 14 new fields:
                        parent_kernel_id=-1, is_persistent=True,
                    )
        return results
```

⚠ The plan's `recorder.kernel_launch` call assumes the new fields exist; T1 adds them to KernelLaunch.

### 3.3 device_launch (dynamic parallelism)

```python
"""Module-level state for pending child launches."""
_pending_child_launches: list = []


def device_launch(parent_kernel_id: int, ptx_src: str, grid: tuple, block: tuple,
                    params: dict, *, kernel_name: str = "<child>") -> None:
    """Schedule a child kernel launch from a parent. Phase 14.
    
    Drained by Stream.synchronize or explicit drain_pending_child_launches."""
    _pending_child_launches.append({
        "parent_kernel_id": parent_kernel_id,
        "ptx_src": ptx_src, "grid": grid, "block": block,
        "params": params, "kernel_name": kernel_name,
    })


def drain_pending_child_launches(config, recorder=None) -> list:
    """Process all pending child launches; return list of Results."""
    from gpusim.api import Stream, synchronize
    results = []
    while _pending_child_launches:
        item = _pending_child_launches.pop(0)
        s = Stream()
        s.launch(ptx_src=item["ptx_src"], grid=item["grid"], block=item["block"],
                  params=item["params"], kernel_name=item["kernel_name"],
                  config=config)
        multi_res = synchronize(streams=[s], config=config)
        if s.stream_id in multi_res.streams and multi_res.streams[s.stream_id]:
            res = multi_res.streams[s.stream_id][0]
            results.append(res)
            if recorder is not None:
                recorder.kernel_launch(
                    stream_id=s.stream_id,
                    kernel_name=item["kernel_name"],
                    grid=item["grid"], block=item["block"],
                    launch_cycle=0, complete_cycle=res.metrics.get("cycles", 0),
                    n_ctas=item["grid"][0]*item["grid"][1]*item["grid"][2],
                    parent_kernel_id=item["parent_kernel_id"], is_persistent=False,
                )
    return results


def reset_pending_child_launches() -> None:
    """Test helper."""
    _pending_child_launches.clear()
```

### 3.4 KernelLaunch trace extension (`gpusim/trace/events.py`)

Add 2 new defaulted fields:

```python
@dataclass(frozen=True)
class KernelLaunch:
    stream_id: int
    kernel_name: str
    grid: tuple
    block: tuple
    launch_cycle: int
    complete_cycle: int
    n_ctas: int
    parent_kernel_id: int = -1     # NEW Phase 14: -1 = top-level
    is_persistent: bool = False     # NEW Phase 14
```

Update `Recorder.kernel_launch` to accept new kwargs (default values).

---

## 4. Trace + Analysis

### 4.1 Reuse KernelLaunch with new fields (covered in §3.4)

### 4.2 3 new metrics

```python
def persistent_kernel_throughput(kernel_launch_df, total_cycles: int) -> float:
    """Persistent kernel iterations per 1000 cycles."""
    if kernel_launch_df is None or kernel_launch_df.empty or total_cycles <= 0:
        return 0.0
    persistent = kernel_launch_df[kernel_launch_df.get("is_persistent", False) == True]
    return float(len(persistent)) / max(total_cycles, 1) * 1000


def dynamic_parallelism_depth(kernel_launch_df) -> int:
    """Maximum parent→child chain depth in this trace."""
    if kernel_launch_df is None or kernel_launch_df.empty:
        return 0
    # Build parent→children map; recursively find longest chain
    children = {}
    for _, row in kernel_launch_df.iterrows():
        pid = int(row.get("parent_kernel_id", -1))
        if pid >= 0:
            children.setdefault(pid, []).append(row)
    # Use stream_id as proxy for kernel_id (each launch has unique stream)
    cache = {}
    def depth_from(node_id):
        if node_id in cache: return cache[node_id]
        if node_id not in children:
            cache[node_id] = 1
            return 1
        cache[node_id] = 1 + max(depth_from(int(c.get("stream_id", -1)))
                                    for c in children[node_id])
        return cache[node_id]
    if not children: return 0
    roots = [int(c.get("stream_id", -1)) for cs in children.values() for c in cs
              if int(c.get("parent_kernel_id", -1)) not in children]
    if not roots: return 0
    return max(depth_from(r) for r in roots)


def dynamic_parallelism_fanout(kernel_launch_df) -> dict:
    """Per-parent count of child launches: {parent_id: n_children}."""
    if kernel_launch_df is None or kernel_launch_df.empty:
        return {}
    out = {}
    for _, row in kernel_launch_df.iterrows():
        pid = int(row.get("parent_kernel_id", -1))
        if pid >= 0:
            out[pid] = out.get(pid, 0) + 1
    return out
```

### 4.3 Result API extension

```python
class MultiStreamResult:
    def persistent_kernel_throughput(self) -> float: ...
    def dynamic_parallelism_depth(self) -> int: ...
    def dynamic_parallelism_fanout(self) -> dict: ...
```

---

## 5. Viz

Reuse Phase 7+ HTML §27 (kernel launches) + Perfetto Stream-N swimlane. Phase 14 adds no new viz section; the new fields appear as additional columns in §27 table.

---

## 6. Examples (4)

### 6.1 `persistent_kernel_server/`
- Persistent kernel processes 5 work items from queue.
- **Verifies:** all 5 items processed; KernelLaunch events have `is_persistent=True`.

### 6.2 `persistent_work_queue/`
- Push items dynamically while kernel running (queue grows).
- **Verifies:** queue can grow before kernel processes; final item count correct.

### 6.3 `dynamic_parallelism_recursive/`
- Parent kernel triggers child via `device_launch()`; child triggers grandchild.
- **Verifies:** depth chain == 3; trace shows parent→child→grandchild relationships.

### 6.4 `persistent_pipeline/` ⭐ Capstone
- Combine: persistent producer kernel + persistent consumer kernel via WorkQueue.

---

## 7. Tutorials

`docs/tutorial/` chapters 54-57:
- **54-persistent-kernel-server.md** — example 1
- **55-persistent-work-queue.md** — example 2
- **56-dynamic-parallelism-recursive.md** — example 3
- **57-persistent-pipeline.md** — example 4 ⭐

---

## 8. Testing strategy

### Unit tests (~14 new)
- `tests/unit/persistent/test_work_queue.py` — push/pop/stop/empty
- `tests/unit/persistent/test_persistent_kernel.py` — start drains queue
- `tests/unit/persistent/test_dynamic_parallelism.py` — device_launch + drain
- `tests/unit/persistent/test_persistent_recorder.py` — KernelLaunch new fields recorded
- `tests/unit/trace/test_kernel_launch_phase14_fields.py` — backward compat
- `tests/unit/analysis/test_phase14_metrics.py` — 3 metrics

### Parity tests (~4 — one per example)

### Microbench
- `test_phase14_facts.py` (fast):
  - WorkQueue FIFO order
  - Persistent kernel processes N items in N iterations
  - Dynamic parallelism depth correct
- `test_phase14_runtime.py` (slow): 4 examples each under 60s

### Regression
- Rename `tests/parity/test_phase1_12_examples_unchanged.py` → `test_phase1_13_examples_unchanged.py`
- Add 3 Phase 13 examples to the regression list

### Test count target
684 (Phase 13 baseline) → ~710 (+26).

---

## 9. Milestones

| Milestone | Scope | Tag |
|---|---|---|
| **M1** WorkQueue + KernelLaunch new fields + recorder accepts | T1–T3 | `M1-phase14-complete` |
| **M2** PersistentKernel + persistent_kernel_server example | T4–T6 | `M2-phase14-complete` |
| **M3** device_launch + dynamic_parallelism_recursive example | T7–T10 | `M3-phase14-complete` |
| **M4** 3 metrics + 2 more examples (persistent_work_queue + persistent_pipeline) | T11–T16 | `M4-phase14-complete` |
| **M5** Tutorials + microbench + regression rename + README v14 + ship | T17–T22 | `phase14-complete` |

Estimated 22 tasks total.

---

## 10. File list

### New files
```
gpusim/persistent/__init__.py
gpusim/persistent/queue.py       # WorkQueue
gpusim/persistent/kernel.py      # PersistentKernel
gpusim/persistent/dynamic.py     # device_launch + drain_pending_child_launches
examples/persistent_kernel_server/    # 5 files (M2)
examples/dynamic_parallelism_recursive/    # 5 files (M3)
examples/persistent_work_queue/       # 5 files (M4)
examples/persistent_pipeline/         # 5 files (M4)
docs/tutorial/54-persistent-kernel-server.md
docs/tutorial/55-persistent-work-queue.md
docs/tutorial/56-dynamic-parallelism-recursive.md
docs/tutorial/57-persistent-pipeline.md
tests/unit/persistent/__init__.py
tests/unit/persistent/test_work_queue.py
tests/unit/persistent/test_persistent_kernel.py
tests/unit/persistent/test_dynamic_parallelism.py
tests/unit/persistent/test_persistent_recorder.py
tests/unit/trace/test_kernel_launch_phase14_fields.py
tests/unit/analysis/test_phase14_metrics.py
tests/parity/test_persistent_kernel_server.py
tests/parity/test_dynamic_parallelism_recursive.py
tests/parity/test_persistent_work_queue.py
tests/parity/test_persistent_pipeline.py
tests/microbench/test_phase14_facts.py
tests/microbench/test_phase14_runtime.py
tests/reference/data/{4 example names}.ref.json
```

### Modified files
```
gpusim/__init__.py               # + WorkQueue + PersistentKernel + device_launch exports
gpusim/trace/events.py           # +parent_kernel_id +is_persistent on KernelLaunch
gpusim/trace/recorder.py         # kernel_launch accepts new kwargs
gpusim/analysis/metrics.py       # +3 metrics
gpusim/api.py                    # MultiStreamResult.persistent_kernel_throughput etc
tests/parity/test_phase1_12_examples_unchanged.py → test_phase1_13_examples_unchanged.py
tests/reference/gen_reference.py # +4 kernel names
README.md                        # v14 — Phase 14 capabilities
```

---

## 11. Backward compatibility

- All Phase 1-13 examples + tests pass unchanged.
- KernelLaunch new fields default to `parent_kernel_id=-1` and `is_persistent=False` — existing code works.
- `gpusim.WorkQueue`, `gpusim.PersistentKernel`, `gpusim.device_launch` are new opt-in additions.

---

## 12. Acceptance criteria

Phase 14 ships when:
- [ ] All 5 milestone tags present (`M1-phase14-complete` ... `M4-phase14-complete`, `phase14-complete`)
- [ ] All 4 examples run cleanly
- [ ] All 4 parity tests pass
- [ ] Microbench: WorkQueue FIFO order verified; persistent kernel processes N items in N iterations
- [ ] Phase 1-13 regression test (renamed) passes
- [ ] Test count: 684 → ~710 (+26)
- [ ] README v14 documents Phase 14 capabilities
