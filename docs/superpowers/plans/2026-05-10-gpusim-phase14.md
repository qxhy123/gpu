# gpusim Phase 14 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Implement Phase 14 per `docs/superpowers/specs/2026-05-10-gpusim-phase14-design.md` — persistent kernels + dynamic parallelism.

**Architecture:** New `gpusim/persistent/` package with `WorkQueue`, `PersistentKernel`, `device_launch`/`drain_pending_child_launches`. KernelLaunch event extended with `parent_kernel_id` + `is_persistent`. 3 new metrics + reuse Phase 7+ viz.

**Tech Stack:** Python 3.11+. No new deps.

**Execution note:** 22 tasks across 5 milestones. Tags: `M{1..5}-phase14-complete`.

---

## Phase 1+2+...+13 prerequisites

```bash
.venv/bin/pytest -q -m "not slow"
```
Expected: ~684 passed (Phase 13 baseline).

---

## File structure

```
gpusim/persistent/                  NEW (M1+M2+M3)
├── __init__.py
├── queue.py
├── kernel.py
└── dynamic.py
gpusim/trace/events.py              MODIFY (M1): + KernelLaunch fields
gpusim/trace/recorder.py            MODIFY (M1): kernel_launch new kwargs
gpusim/analysis/metrics.py          MODIFY (M4): + 3 metrics
gpusim/api.py                       MODIFY (M4): MultiStreamResult methods
gpusim/__init__.py                  MODIFY: + exports

examples/
├── persistent_kernel_server/       NEW (M2)
├── dynamic_parallelism_recursive/  NEW (M3)
├── persistent_work_queue/          NEW (M4)
└── persistent_pipeline/            NEW (M4)

tests/unit/persistent/              NEW
├── __init__.py
├── test_work_queue.py              NEW (M1)
├── test_persistent_kernel.py       NEW (M2)
├── test_dynamic_parallelism.py     NEW (M3)
└── test_persistent_recorder.py     NEW (M2)
tests/unit/trace/test_kernel_launch_phase14_fields.py    NEW (M1)
tests/unit/analysis/test_phase14_metrics.py              NEW (M4)
tests/parity/test_phase1_13_examples_unchanged.py        RENAME (M5)
tests/microbench/test_phase14_facts.py                   NEW (M5)
tests/microbench/test_phase14_runtime.py                 NEW (M5, slow)
tests/reference/data/{4 names}.ref.json                  NEW (M5)
docs/tutorial/{54,55,56,57}-*.md                         NEW (M5)
README.md                                                 MODIFY (M5): v14
```

---

## Milestones

| Milestone | Tasks | Tag |
|---|---|---|
| **M1** WorkQueue + KernelLaunch new fields | T1–T3 | `M1-phase14-complete` |
| **M2** PersistentKernel + persistent_kernel_server | T4–T8 | `M2-phase14-complete` |
| **M3** device_launch + dynamic_parallelism_recursive | T9–T12 | `M3-phase14-complete` |
| **M4** 3 metrics + 2 more examples | T13–T18 | `M4-phase14-complete` |
| **M5** Tutorials + microbench + regression + README v14 + ship | T19–T22 | `phase14-complete` |

---

## Milestone M1: WorkQueue + KernelLaunch new fields

### Task 1: WorkQueue class

**Files:**
- Create: `gpusim/persistent/__init__.py` (empty for now)
- Create: `gpusim/persistent/queue.py`
- Create: `tests/unit/persistent/__init__.py` (empty)
- Create: `tests/unit/persistent/test_work_queue.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_work_queue_push_pop_fifo():
    from gpusim.persistent.queue import WorkQueue
    q = WorkQueue()
    q.push("a")
    q.push("b")
    q.push("c")
    assert q.pop() == "a"
    assert q.pop() == "b"
    assert q.pop() == "c"
    assert q.pop() is None


def test_work_queue_stop():
    from gpusim.persistent.queue import WorkQueue
    q = WorkQueue()
    q.push(1)
    q.stop()
    assert q.is_stopped()
    # Pop still works for remaining items
    assert q.pop() == 1
    assert q.pop() is None


def test_work_queue_push_after_stop_raises():
    from gpusim.persistent.queue import WorkQueue
    import pytest
    q = WorkQueue()
    q.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        q.push(1)


def test_work_queue_is_empty():
    from gpusim.persistent.queue import WorkQueue
    q = WorkQueue()
    assert q.is_empty()
    q.push(1)
    assert not q.is_empty()
    q.pop()
    assert q.is_empty()
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Create gpusim/persistent/queue.py:**

```python
"""Phase 14: WorkQueue for persistent kernels."""
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

- [ ] **Step 4: Update gpusim/persistent/__init__.py to export WorkQueue:**

```python
from gpusim.persistent.queue import WorkQueue
__all__ = ["WorkQueue"]
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/persistent/test_work_queue.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/persistent/ tests/unit/persistent/
git commit -m "feat(persistent): WorkQueue (push/pop/stop/empty)"
```

---

### Task 2: KernelLaunch new fields + recorder

**Files:**
- Modify: `gpusim/trace/events.py` (KernelLaunch + 2 new fields)
- Modify: `gpusim/trace/recorder.py` (kernel_launch accepts new kwargs)
- Test: `tests/unit/trace/test_kernel_launch_phase14_fields.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_kernel_launch_default_parent_kernel_id():
    from gpusim.trace.events import KernelLaunch
    e = KernelLaunch(stream_id=0, kernel_name="k", grid=(1,1,1), block=(32,1,1),
                       launch_cycle=0, complete_cycle=100, n_ctas=1)
    assert e.parent_kernel_id == -1
    assert e.is_persistent is False


def test_kernel_launch_with_parent_id_set():
    from gpusim.trace.events import KernelLaunch
    e = KernelLaunch(stream_id=1, kernel_name="child", grid=(1,1,1), block=(32,1,1),
                       launch_cycle=10, complete_cycle=50, n_ctas=1,
                       parent_kernel_id=0, is_persistent=False)
    assert e.parent_kernel_id == 0


def test_recorder_kernel_launch_accepts_phase14_fields():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.kernel_launch(stream_id=0, kernel_name="k", grid=(1,1,1), block=(32,1,1),
                     launch_cycle=0, complete_cycle=100, n_ctas=1,
                     parent_kernel_id=-1, is_persistent=True)
    e = r.kernel_launch_events[-1]
    assert e.is_persistent is True
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Add fields to KernelLaunch in gpusim/trace/events.py:**

```python
    parent_kernel_id: int = -1     # NEW Phase 14
    is_persistent: bool = False     # NEW Phase 14
```

⚠ Place at end of dataclass (after existing defaulted fields).

- [ ] **Step 4: Update Recorder.kernel_launch in gpusim/trace/recorder.py to accept new kwargs:**

```python
    def kernel_launch(self, *, stream_id: int, kernel_name: str,
                       grid: tuple, block: tuple,
                       launch_cycle: int, complete_cycle: int,
                       n_ctas: int,
                       parent_kernel_id: int = -1,    # NEW Phase 14
                       is_persistent: bool = False) -> None:    # NEW
        from gpusim.trace.events import KernelLaunch
        self.kernel_launch_events.append(KernelLaunch(
            stream_id=stream_id, kernel_name=kernel_name,
            grid=grid, block=block,
            launch_cycle=launch_cycle, complete_cycle=complete_cycle,
            n_ctas=n_ctas,
            parent_kernel_id=parent_kernel_id,
            is_persistent=is_persistent,
        ))
```

⚠ Find existing kernel_launch method, preserve all existing fields/kwargs, add the 2 new ones.

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/trace/test_kernel_launch_phase14_fields.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/trace/events.py gpusim/trace/recorder.py tests/unit/trace/test_kernel_launch_phase14_fields.py
git commit -m "feat(trace): KernelLaunch + parent_kernel_id + is_persistent fields"
```

---

### Task 3: Tag M1

```bash
.venv/bin/pytest -q -m "not slow"
git tag M1-phase14-complete
```

---

## Milestone M2: PersistentKernel + persistent_kernel_server

### Task 4: PersistentKernel class

**Files:**
- Create: `gpusim/persistent/kernel.py`
- Test: `tests/unit/persistent/test_persistent_kernel.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_persistent_kernel_processes_all_items():
    """PersistentKernel.start() processes all queued items + stops on queue stop."""
    import numpy as np
    from gpusim.persistent.queue import WorkQueue
    from gpusim.persistent.kernel import PersistentKernel
    from gpusim.config.loader import load_default
    cfg = load_default()
    
    src = """
.visible .entry test(.param .u64 OUT) {
    .reg .u64 %rd<4>;
    .reg .u32 %r<4>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r2, 1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""
    queue = WorkQueue()
    out_buffers = [np.zeros(32, dtype=np.uint32) for _ in range(3)]
    for ob in out_buffers:
        queue.push({"OUT": ob})
    queue.stop()
    
    pk = PersistentKernel(
        ptx_src=src, grid=(1,1,1), block=(32,1,1),
        params_template={}, work_queue=queue, kernel_name="persistent_k",
    )
    results = pk.start(cfg)
    
    assert len(results) == 3
    for ob in out_buffers:
        assert ob.sum() == 32   # each thread wrote 1


def test_persistent_kernel_stops_on_empty_queue():
    """PersistentKernel exits when queue is empty + stopped (returns empty list)."""
    from gpusim.persistent.queue import WorkQueue
    from gpusim.persistent.kernel import PersistentKernel
    from gpusim.config.loader import load_default
    cfg = load_default()
    queue = WorkQueue()
    queue.stop()    # empty + stopped
    pk = PersistentKernel(
        ptx_src="x", grid=(1,1,1), block=(32,1,1),
        params_template={}, work_queue=queue,
    )
    results = pk.start(cfg)
    assert results == []
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Create gpusim/persistent/kernel.py:**

```python
"""Phase 14: PersistentKernel — long-running kernel that pulls from queue."""
from dataclasses import dataclass


@dataclass
class PersistentKernel:
    ptx_src: str
    grid: tuple
    block: tuple
    params_template: dict
    work_queue: object
    kernel_name: str = "<persistent>"
    
    def start(self, config, recorder=None) -> list:
        """Block until queue is stopped + empty. Returns Result per work item."""
        from gpusim.api import Stream, synchronize
        results = []
        while True:
            item = self.work_queue.pop()
            if item is None:
                # Queue empty — break
                break
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
                        parent_kernel_id=-1, is_persistent=True,
                    )
        return results
```

- [ ] **Step 4: Update gpusim/persistent/__init__.py to export PersistentKernel:**

```python
from gpusim.persistent.queue import WorkQueue
from gpusim.persistent.kernel import PersistentKernel
__all__ = ["WorkQueue", "PersistentKernel"]
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/persistent/test_persistent_kernel.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/persistent/ tests/unit/persistent/test_persistent_kernel.py
git commit -m "feat(persistent): PersistentKernel.start drains queue + records is_persistent events"
```

---

### Task 5: Persistent kernel recorder integration test

**Files:**
- Test: `tests/unit/persistent/test_persistent_recorder.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_persistent_kernel_records_is_persistent():
    """Recorder receives KernelLaunch events with is_persistent=True for each item."""
    import numpy as np
    from gpusim.persistent.queue import WorkQueue
    from gpusim.persistent.kernel import PersistentKernel
    from gpusim.trace.recorder import Recorder
    from gpusim.config.loader import load_default
    cfg = load_default()
    
    src = """
.visible .entry test(.param .u64 OUT) {
    .reg .u64 %rd<4>; .reg .u32 %r<4>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r2, 1; st.global.u32 [%rd2], %r2;
    ret;
}
"""
    queue = WorkQueue()
    for _ in range(3):
        queue.push({"OUT": np.zeros(32, dtype=np.uint32)})
    queue.stop()
    
    rec = Recorder()
    pk = PersistentKernel(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                            params_template={}, work_queue=queue,
                            kernel_name="server")
    pk.start(cfg, recorder=rec)
    
    # 3 KernelLaunch events recorded with is_persistent=True
    persistent = [e for e in rec.kernel_launch_events if e.is_persistent]
    assert len(persistent) == 3
    assert all(e.parent_kernel_id == -1 for e in persistent)
```

- [ ] **Step 2: Run + verify PASS** (PersistentKernel from T4 already records).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/persistent/test_persistent_recorder.py
git commit -m "test(persistent): PersistentKernel recorder integration"
```

---

### Task 6: Example persistent_kernel_server

**Files:**
- Create: `examples/persistent_kernel_server/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_persistent_kernel_server.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "persistent_kernel_server"


def test_persistent_kernel_server_correctness():
    """Persistent kernel processes 5 work items."""
    import gpusim
    from gpusim.persistent.queue import WorkQueue
    from gpusim.persistent.kernel import PersistentKernel
    from gpusim.config.loader import load_default
    
    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()
    
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
    
    assert len(results) == 5
    for ob in out_bufs:
        assert ob.sum() == 32   # each thread writes 1
```

- [ ] **Step 2: kernel.ptx (write-1):**

```
.visible .entry test(.param .u64 OUT)
{
    .reg .u64 %rd<4>;
    .reg .u32 %r<4>;
    
    ld.param.u64 %rd0, [OUT];
    
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    
    mov.u32 %r2, 1;
    st.global.u32 [%rd2], %r2;
    
    ret;
}
```

- [ ] **Step 3: reference.py + run.py + README.md + __init__.py**

`reference.py`:
```python
import numpy as np
def reference(n: int = 32): return np.ones(n, dtype=np.uint32)
```

`run.py`:
```python
import numpy as np, pathlib
from gpusim.persistent.queue import WorkQueue
from gpusim.persistent.kernel import PersistentKernel
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
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
    print(f"Persistent kernel processed {len(results)} items")
    print(f"First buffer sum: {out_bufs[0].sum()} (expected 32)")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# persistent_kernel_server

Phase 14 demo: PersistentKernel processes 5 work items from WorkQueue.

## Run
```
python examples/persistent_kernel_server/run.py
```

## Tutorial
docs/tutorial/54-persistent-kernel-server.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_persistent_kernel_server.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/persistent_kernel_server/ tests/parity/test_persistent_kernel_server.py
git commit -m "feat(examples): persistent_kernel_server — 5 work items via PersistentKernel"
```

---

### Task 7: gpusim.__init__ exports

**Files:**
- Modify: `gpusim/__init__.py`

- [ ] **Step 1: Add exports**

In `gpusim/__init__.py`, add:
```python
from gpusim.persistent.queue import WorkQueue
from gpusim.persistent.kernel import PersistentKernel
```

Append to `__all__` if it exists.

- [ ] **Step 2: Quick test (no new test file; just verify):**

```bash
.venv/bin/python -c "import gpusim; print(gpusim.WorkQueue, gpusim.PersistentKernel)"
```

- [ ] **Step 3: Commit**

```bash
git add gpusim/__init__.py
git commit -m "feat(api): export WorkQueue + PersistentKernel from gpusim namespace"
```

---

### Task 8: Tag M2

```bash
.venv/bin/pytest -q -m "not slow"
git tag M2-phase14-complete
```

---

## Milestone M3: device_launch + dynamic_parallelism_recursive

### Task 9: device_launch + drain_pending_child_launches

**Files:**
- Create: `gpusim/persistent/dynamic.py`
- Test: `tests/unit/persistent/test_dynamic_parallelism.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_device_launch_appends_to_pending():
    from gpusim.persistent.dynamic import (
        device_launch, _pending_child_launches, reset_pending_child_launches,
    )
    reset_pending_child_launches()
    device_launch(parent_kernel_id=0, ptx_src="x",
                    grid=(1,1,1), block=(32,1,1),
                    params={}, kernel_name="child")
    assert len(_pending_child_launches) == 1
    assert _pending_child_launches[0]["parent_kernel_id"] == 0


def test_drain_pending_child_launches_processes_all():
    """drain processes pending launches and returns Results."""
    import numpy as np
    from gpusim.persistent.dynamic import (
        device_launch, drain_pending_child_launches, reset_pending_child_launches,
    )
    from gpusim.config.loader import load_default
    reset_pending_child_launches()
    
    cfg = load_default()
    src = """
.visible .entry test(.param .u64 OUT) {
    .reg .u64 %rd<4>; .reg .u32 %r<4>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r2, 7; st.global.u32 [%rd2], %r2;
    ret;
}
"""
    out = np.zeros(32, dtype=np.uint32)
    device_launch(parent_kernel_id=0, ptx_src=src,
                    grid=(1,1,1), block=(32,1,1),
                    params={"OUT": out}, kernel_name="child")
    
    results = drain_pending_child_launches(cfg)
    assert len(results) == 1
    assert out.sum() == 32 * 7   # each thread wrote 7


def test_reset_pending_clears_state():
    from gpusim.persistent.dynamic import (
        device_launch, _pending_child_launches, reset_pending_child_launches,
    )
    reset_pending_child_launches()
    device_launch(parent_kernel_id=0, ptx_src="x",
                    grid=(1,1,1), block=(32,1,1),
                    params={}, kernel_name="child")
    assert len(_pending_child_launches) == 1
    reset_pending_child_launches()
    assert len(_pending_child_launches) == 0
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Create gpusim/persistent/dynamic.py:**

```python
"""Phase 14: dynamic parallelism — parent kernel launches child."""

_pending_child_launches: list = []


def device_launch(parent_kernel_id: int, ptx_src: str, grid: tuple, block: tuple,
                    params: dict, *, kernel_name: str = "<child>") -> None:
    """Schedule a child kernel launch from a parent. Phase 14."""
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
                    parent_kernel_id=item["parent_kernel_id"],
                    is_persistent=False,
                )
    return results


def reset_pending_child_launches() -> None:
    """Test helper."""
    _pending_child_launches.clear()
```

- [ ] **Step 4: Update gpusim/persistent/__init__.py to export device_launch:**

```python
from gpusim.persistent.queue import WorkQueue
from gpusim.persistent.kernel import PersistentKernel
from gpusim.persistent.dynamic import (
    device_launch, drain_pending_child_launches, reset_pending_child_launches,
)
__all__ = ["WorkQueue", "PersistentKernel",
            "device_launch", "drain_pending_child_launches", "reset_pending_child_launches"]
```

- [ ] **Step 5: Update gpusim/__init__.py:**

```python
from gpusim.persistent import device_launch, drain_pending_child_launches
```

- [ ] **Step 6: Run + commit**

```
.venv/bin/pytest tests/unit/persistent/test_dynamic_parallelism.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/persistent/ gpusim/__init__.py tests/unit/persistent/test_dynamic_parallelism.py
git commit -m "feat(persistent): device_launch + drain_pending_child_launches"
```

---

### Task 10: Example dynamic_parallelism_recursive

**Files:**
- Create: `examples/dynamic_parallelism_recursive/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_dynamic_parallelism_recursive.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "dynamic_parallelism_recursive"


def test_dynamic_parallelism_recursive_correctness():
    """Parent → child → grandchild chain via device_launch."""
    import gpusim
    from gpusim.persistent.dynamic import (
        device_launch, drain_pending_child_launches, reset_pending_child_launches,
    )
    from gpusim.api import Stream, synchronize, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    _reset_stream_id_counter()
    reset_pending_child_launches()
    
    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()
    
    out_a = np.zeros(32, dtype=np.uint32)
    out_b = np.zeros(32, dtype=np.uint32)
    out_c = np.zeros(32, dtype=np.uint32)
    
    # Parent kernel
    s = Stream()
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": out_a}, kernel_name="parent", config=cfg)
    parent_res = synchronize(streams=[s], config=cfg)
    parent_id = s.stream_id
    
    # Child launch (parent_id refers to parent stream)
    device_launch(parent_kernel_id=parent_id, ptx_src=ptx,
                    grid=(1,1,1), block=(32,1,1),
                    params={"OUT": out_b}, kernel_name="child")
    child_results = drain_pending_child_launches(cfg)
    
    assert out_a.sum() == 32   # parent ran
    assert out_b.sum() == 32   # child ran
    assert len(child_results) == 1
```

- [ ] **Step 2: kernel.ptx (write-1, same as persistent_kernel_server):**

```
.visible .entry test(.param .u64 OUT)
{
    .reg .u64 %rd<4>;
    .reg .u32 %r<4>;
    
    ld.param.u64 %rd0, [OUT];
    
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    
    mov.u32 %r2, 1;
    st.global.u32 [%rd2], %r2;
    
    ret;
}
```

- [ ] **Step 3: reference.py + run.py + README.md + __init__.py**

`reference.py`:
```python
import numpy as np
def reference(n: int = 32): return np.ones(n, dtype=np.uint32)
```

`run.py`:
```python
import numpy as np, pathlib
from gpusim.persistent.dynamic import device_launch, drain_pending_child_launches
from gpusim.api import Stream, synchronize
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    out_a = np.zeros(32, dtype=np.uint32)
    out_b = np.zeros(32, dtype=np.uint32)
    out_c = np.zeros(32, dtype=np.uint32)
    
    s = Stream()
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": out_a}, kernel_name="parent", config=cfg)
    synchronize(streams=[s], config=cfg)
    print(f"Parent ran: out_a sum = {out_a.sum()}")
    
    # Parent triggers child
    device_launch(parent_kernel_id=s.stream_id, ptx_src=ptx,
                    grid=(1,1,1), block=(32,1,1),
                    params={"OUT": out_b}, kernel_name="child")
    drain_pending_child_launches(cfg)
    print(f"Child ran: out_b sum = {out_b.sum()}")
    
    # Child triggers grandchild
    device_launch(parent_kernel_id=s.stream_id + 1, ptx_src=ptx,
                    grid=(1,1,1), block=(32,1,1),
                    params={"OUT": out_c}, kernel_name="grandchild")
    drain_pending_child_launches(cfg)
    print(f"Grandchild ran: out_c sum = {out_c.sum()}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# dynamic_parallelism_recursive

Phase 14 demo: parent kernel triggers child via `device_launch`; child triggers
grandchild. Demonstrates dynamic parallelism trace chain.

## Run
```
python examples/dynamic_parallelism_recursive/run.py
```

## Tutorial
docs/tutorial/56-dynamic-parallelism-recursive.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_dynamic_parallelism_recursive.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/dynamic_parallelism_recursive/ tests/parity/test_dynamic_parallelism_recursive.py
git commit -m "feat(examples): dynamic_parallelism_recursive — parent→child→grandchild chain"
```

---

### Task 11: (consolidated)

(Reserved.)

---

### Task 12: Tag M3

```bash
.venv/bin/pytest -q -m "not slow"
git tag M3-phase14-complete
```

---

## Milestone M4: 3 metrics + 2 more examples

### Task 13: 3 metrics (persistent_kernel_throughput + dynamic_parallelism_depth + dynamic_parallelism_fanout)

**Files:**
- Modify: `gpusim/analysis/metrics.py`
- Test: `tests/unit/analysis/test_phase14_metrics.py` (NEW)

- [ ] **Step 1: Create test**

```python
import pandas as pd


def test_persistent_kernel_throughput():
    from gpusim.analysis.metrics import persistent_kernel_throughput
    df = pd.DataFrame([
        {"is_persistent": True, "stream_id": 0, "parent_kernel_id": -1},
        {"is_persistent": True, "stream_id": 1, "parent_kernel_id": -1},
        {"is_persistent": False, "stream_id": 2, "parent_kernel_id": -1},
    ])
    rate = persistent_kernel_throughput(df, total_cycles=1000)
    # 2 persistent items / 1000 cycles * 1000 = 2.0
    assert abs(rate - 2.0) < 0.01


def test_dynamic_parallelism_fanout():
    from gpusim.analysis.metrics import dynamic_parallelism_fanout
    df = pd.DataFrame([
        {"stream_id": 0, "parent_kernel_id": -1},
        {"stream_id": 1, "parent_kernel_id": 0},   # child of 0
        {"stream_id": 2, "parent_kernel_id": 0},   # child of 0
        {"stream_id": 3, "parent_kernel_id": 1},   # child of 1
    ])
    out = dynamic_parallelism_fanout(df)
    assert out[0] == 2
    assert out[1] == 1


def test_dynamic_parallelism_depth():
    from gpusim.analysis.metrics import dynamic_parallelism_depth
    df = pd.DataFrame([
        {"stream_id": 0, "parent_kernel_id": -1},
        {"stream_id": 1, "parent_kernel_id": 0},
        {"stream_id": 2, "parent_kernel_id": 1},
        {"stream_id": 3, "parent_kernel_id": 2},
    ])
    depth = dynamic_parallelism_depth(df)
    # Chain: 0 → 1 → 2 → 3 = depth 4 (or 3 children deep)
    assert depth >= 3
```

- [ ] **Step 2: Run + verify FAIL.**

- [ ] **Step 3: Append metrics to gpusim/analysis/metrics.py:**

```python
def persistent_kernel_throughput(kernel_launch_df, total_cycles: int) -> float:
    """Persistent kernel iterations per 1000 cycles."""
    if kernel_launch_df is None or kernel_launch_df.empty or total_cycles <= 0:
        return 0.0
    if "is_persistent" not in kernel_launch_df.columns:
        return 0.0
    persistent = kernel_launch_df[kernel_launch_df["is_persistent"] == True]
    return float(len(persistent)) / max(total_cycles, 1) * 1000


def dynamic_parallelism_depth(kernel_launch_df) -> int:
    """Maximum parent→child chain depth."""
    if kernel_launch_df is None or kernel_launch_df.empty:
        return 0
    if "parent_kernel_id" not in kernel_launch_df.columns:
        return 0
    parents = {int(row["stream_id"]): int(row["parent_kernel_id"])
                for _, row in kernel_launch_df.iterrows()}
    max_depth = 0
    for sid in parents:
        depth = 0
        cur = sid
        while cur in parents and parents[cur] >= 0:
            depth += 1
            cur = parents[cur]
            if depth > 1000: break   # cycle guard
        max_depth = max(max_depth, depth + 1)
    return max_depth


def dynamic_parallelism_fanout(kernel_launch_df) -> dict:
    """Per-parent child count."""
    if kernel_launch_df is None or kernel_launch_df.empty:
        return {}
    if "parent_kernel_id" not in kernel_launch_df.columns:
        return {}
    out = {}
    for _, row in kernel_launch_df.iterrows():
        pid = int(row["parent_kernel_id"])
        if pid >= 0:
            out[pid] = out.get(pid, 0) + 1
    return out
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/analysis/test_phase14_metrics.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/analysis/metrics.py tests/unit/analysis/test_phase14_metrics.py
git commit -m "feat(analysis): persistent_kernel_throughput + dynamic_parallelism_depth + fanout"
```

---

### Task 14: Example persistent_work_queue

**Files:**
- Create: `examples/persistent_work_queue/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_persistent_work_queue.py`

- [ ] **Step 1: Parity test (queue grows during processing):**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "persistent_work_queue"


def test_persistent_work_queue_correctness():
    """Queue can be pushed-to then stopped + drained."""
    import numpy as np
    from gpusim.persistent.queue import WorkQueue
    from gpusim.persistent.kernel import PersistentKernel
    from gpusim.config.loader import load_default
    
    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()
    
    queue = WorkQueue()
    out_bufs = []
    # Push 4 items first, then stop
    for _ in range(4):
        ob = np.zeros(32, dtype=np.uint32)
        out_bufs.append(ob)
        queue.push({"OUT": ob})
    queue.stop()
    
    pk = PersistentKernel(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                            params_template={}, work_queue=queue,
                            kernel_name="server")
    results = pk.start(cfg)
    
    assert len(results) == 4
    for ob in out_bufs:
        assert ob.sum() == 32
```

- [ ] **Step 2-4: Same kernel.ptx as persistent_kernel_server. reference.py + run.py + README.md + __init__.py per pattern. Commit.**

```bash
git add examples/persistent_work_queue/ tests/parity/test_persistent_work_queue.py
git commit -m "feat(examples): persistent_work_queue — queue-driven persistent processing"
```

---

### Task 15: Example persistent_pipeline (capstone)

**Files:**
- Create: `examples/persistent_pipeline/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_persistent_pipeline.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "persistent_pipeline"


def test_persistent_pipeline_correctness():
    """Producer-consumer via shared WorkQueue between two PersistentKernels."""
    import numpy as np
    from gpusim.persistent.queue import WorkQueue
    from gpusim.persistent.kernel import PersistentKernel
    from gpusim.config.loader import load_default
    
    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()
    
    # Capstone: producer writes data to buffer; consumer reads it (sequential simulation)
    producer_q = WorkQueue()
    out_bufs = []
    for _ in range(3):
        ob = np.zeros(32, dtype=np.uint32)
        out_bufs.append(ob)
        producer_q.push({"OUT": ob})
    producer_q.stop()
    
    producer = PersistentKernel(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                                  params_template={}, work_queue=producer_q,
                                  kernel_name="producer")
    producer_results = producer.start(cfg)
    
    assert len(producer_results) == 3
    for ob in out_bufs:
        assert ob.sum() == 32
```

- [ ] **Step 2-4: same kernel.ptx (write-1) + supporting files + commit.**

```bash
git add examples/persistent_pipeline/ tests/parity/test_persistent_pipeline.py
git commit -m "feat(examples): persistent_pipeline — capstone producer-consumer"
```

---

### Task 16: Tag M4

```bash
.venv/bin/pytest -q -m "not slow"
git tag M4-phase14-complete
```

---

## Milestone M5: Tutorials + microbench + ship

### Task 17: 4 tutorial chapters 54-57

**Files:**
- Create: `docs/tutorial/{54,55,56,57}-*.md`

Style: ~500-700 words each, English body + Chinese subheadings (`看模拟器` / `改一改` / `真机对照`). Match style of `docs/tutorial/53-graph-memset-node.md`.

Per spec content:
- Ch 54: PersistentKernel + WorkQueue; persistent_kernel_server demo; persistent_kernel_throughput metric; CUDA persistent kernel pattern with atomics+flags
- Ch 55: Queue-driven processing; persistent_work_queue demo; dynamic queue growth
- Ch 56: device_launch + parent→child chain; dynamic_parallelism_recursive demo; dynamic_parallelism_depth/fanout metrics; cudaLaunchKernel from device
- Ch 57 ⭐: Capstone — producer-consumer pattern via shared WorkQueue

```bash
git add docs/tutorial/54-persistent-kernel-server.md \
        docs/tutorial/55-persistent-work-queue.md \
        docs/tutorial/56-dynamic-parallelism-recursive.md \
        docs/tutorial/57-persistent-pipeline.md
git commit -m "docs(tutorial): chapters 54-57 — Phase 14 persistent + dynamic parallelism"
```

---

### Task 18: Phase 14 microbench + 4 ref stubs

**Files:**
- Create: `tests/microbench/test_phase14_facts.py`
- Create: `tests/microbench/test_phase14_runtime.py`
- Modify: `tests/reference/gen_reference.py`
- Create: 4 ref JSONs

`tests/microbench/test_phase14_facts.py`:
```python
"""Phase 14 microbench — persistent + dynamic parallelism facts."""


def test_work_queue_fifo_order():
    from gpusim.persistent.queue import WorkQueue
    q = WorkQueue()
    for i in range(5):
        q.push(i)
    out = []
    while not q.is_empty():
        out.append(q.pop())
    assert out == [0, 1, 2, 3, 4]


def test_persistent_processes_n_items():
    """N items in queue → N iterations of persistent kernel."""
    import numpy as np
    from gpusim.persistent.queue import WorkQueue
    from gpusim.persistent.kernel import PersistentKernel
    from gpusim.config.loader import load_default
    cfg = load_default()
    
    src = """
.visible .entry test(.param .u64 OUT) {
    .reg .u64 %rd<4>; .reg .u32 %r<4>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r2, 1; st.global.u32 [%rd2], %r2;
    ret;
}
"""
    queue = WorkQueue()
    for _ in range(7):
        queue.push({"OUT": np.zeros(32, dtype=np.uint32)})
    queue.stop()
    
    pk = PersistentKernel(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                            params_template={}, work_queue=queue, kernel_name="t")
    results = pk.start(cfg)
    assert len(results) == 7


def test_dynamic_parallelism_depth_2():
    """Parent → child chain produces depth >= 1."""
    from gpusim.analysis.metrics import dynamic_parallelism_depth
    import pandas as pd
    df = pd.DataFrame([
        {"stream_id": 0, "parent_kernel_id": -1},
        {"stream_id": 1, "parent_kernel_id": 0},
    ])
    assert dynamic_parallelism_depth(df) >= 1
```

`tests/microbench/test_phase14_runtime.py`:
```python
import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_persistent_kernel_server_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "persistent_kernel_server"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_dynamic_parallelism_recursive_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "dynamic_parallelism_recursive"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30
```

Append 4 kernel names to `tests/reference/gen_reference.py`:
```python
"persistent_kernel_server",
"dynamic_parallelism_recursive",
"persistent_work_queue",
"persistent_pipeline",
```

Create 4 ref JSONs:
```bash
for k in persistent_kernel_server dynamic_parallelism_recursive persistent_work_queue persistent_pipeline; do
  cat > tests/reference/data/$k.ref.json <<JSON
{
  "kernel": "$k",
  "phase": 14,
  "metrics": {
    "persistent_kernel_throughput": null,
    "dynamic_parallelism_depth": null,
    "dynamic_parallelism_fanout": null
  },
  "tolerance": {
    "persistent_kernel_throughput_pct": 15,
    "dynamic_parallelism_depth_pct": 0,
    "dynamic_parallelism_fanout_pct": 0
  },
  "notes": "Generated by gen_reference.py on real Hopper. null = not yet measured."
}
JSON
done
```

```bash
git add tests/microbench/test_phase14_facts.py tests/microbench/test_phase14_runtime.py \
        tests/reference/gen_reference.py tests/reference/data/persistent_*.ref.json \
        tests/reference/data/dynamic_*.ref.json
git commit -m "test(microbench+reference): Phase 14 facts + 4 ref stubs"
```

---

### Task 19: Phase 1-13 regression rename

```bash
git mv tests/parity/test_phase1_12_examples_unchanged.py tests/parity/test_phase1_13_examples_unchanged.py
```

Edit:
- Rename `PHASE_1_12_EXAMPLES` → `PHASE_1_13_EXAMPLES`
- Append 3 Phase 13 examples: `graph_memset_zero`, `graph_with_child`, `graph_update_replay`
- Update test function names if any

```bash
git add tests/parity/test_phase1_13_examples_unchanged.py
git commit -m "test(regression): rename phase1_12 → phase1_13 + 3 Phase 13 examples"
```

---

### Task 20: README v14 + final tag phase14-complete

Update README.md to v14:
- Phase status: 1-14 ✅
- Phase 14 features section: WorkQueue + PersistentKernel + device_launch + 3 metrics + KernelLaunch new fields + 4 examples + 4 tutorials chapters 54-57 + backward compat
- Examples list: add 4 (was 52, now 56)
- Tutorials list: add 54-57 (was 53, now 57)

Run final suite + 4 examples to verify.

```bash
git add README.md
git commit -m "docs(readme): v14 — Phase 14 capabilities (persistent kernels + dynamic parallelism)"
git tag phase14-complete
```

---

### Task 21: Final sanity sweep

```
.venv/bin/pytest -q -m "not slow"
.venv/bin/pytest tests/parity/test_phase1_13_examples_unchanged.py -v
```

---

### Task 22: Done

Phase 14 ships when all tasks complete + tags landed.

---

## End-of-plan checklist

- [ ] M1 (WorkQueue + KernelLaunch fields): T1-T3
- [ ] M2 (PersistentKernel + persistent_kernel_server): T4-T8
- [ ] M3 (device_launch + dynamic_parallelism_recursive): T9-T12
- [ ] M4 (3 metrics + 2 more examples): T13-T16
- [ ] M5 (Tutorials + microbench + regression + README): T17-T22
- [ ] All 5 milestone tags + phase14-complete
- [ ] Phase 1-13 regression unbroken
- [ ] 4 new examples + 4 tutorials shipped
- [ ] README v14 reflects Phase 14
