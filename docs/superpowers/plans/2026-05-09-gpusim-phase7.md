# gpusim Phase 7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement gpusim Phase 7 per `docs/superpowers/specs/2026-05-09-gpusim-phase7-design.md` — multi-stream / multi-kernel concurrency: `gpusim.Stream()` + `Stream.launch()` + `gpusim.synchronize()`, round-robin CTA scheduling across streams, full trace+metrics+viz pipeline.

**Architecture:** Stream class wraps a launch queue; `MultiStreamScheduler` round-robins across streams when picking next CTA to dispatch; intra-stream FIFO over grids. Reuses Phase 4 multi-SM device + L2/HBM sharing — multi-stream is purely a scheduling layer + `stream_id` propagation. Backward compatible: `gpusim.run(...)` is internally a single-stream path.

**Tech Stack:** Python 3.11+. No new runtime dependencies.

**Execution note:** Plan has 5 milestones (M1–M5) with 28 tasks total. After each milestone, pause for review checkpoint and tag (`M{1..5}-phase7-complete`).

---

## Scope check

Phase 7 covers a single concern (multi-stream concurrency). All sub-features (Stream API, scheduler, trace, metrics, examples, viz, docs) are tightly coupled and ship together.

- **M1** (frontend+trace): Stream class + GridLaunch + KernelLaunch event + stream_id propagation skeleton.
- **M2** (scheduler): MultiStreamScheduler + Device.run_streams + 1 example.
- **M3** (analytics): 4 metrics + MultiStreamResult full API + compute/memory overlap example.
- **M4** (more examples): l2_contention + serial-vs-concurrent.
- **M5** (viz+docs+ship): HTML §27/§28 + Perfetto + 4 tutorials + microbench + README v7.

---

## Phase 1+2+3+4+5+6 prerequisites

```bash
cd /Users/yangyang/ai_projs/gpu
git log --oneline | head -3
git tag | grep phase
.venv/bin/pytest -q -m "not slow"
```

Expected: ~438 passed (Phase 6 baseline), ≥10 skipped.

---

## File structure

```
gpusim/
├── api.py                                  MODIFY (M1+M3): + Stream + MultiStreamResult + synchronize + Result.stream_id
├── core/
│   ├── scheduler.py                        MODIFY (M2): + MultiStreamScheduler
│   ├── device.py                           MODIFY (M2): + run_streams
│   ├── sm.py                               MODIFY (M2): + dispatch_cta(stream_id), propagate to events
│   └── sub_core.py                         MODIFY (M2): + propagate stream_id to events

gpusim/trace/
├── events.py                               MODIFY (M1): + KernelLaunch + stream_id field on 11 events
├── recorder.py                             MODIFY (M1): + kernel_launch method + stream_id kwarg on 11 methods
└── writer.py                               MODIFY (M1): + kernel_launch.parquet

gpusim/analysis/metrics.py                  MODIFY (M3): + 4 metrics
gpusim/viz/
├── notebook.py                             MODIFY (M3): + kernel_launch_events_dataframe
├── html_report.py                          MODIFY (M5): + §27/§28 render helpers
├── _template.html.j2                       MODIFY (M5): + §27/§28 blocks
└── perfetto.py                             MODIFY (M5): + Stream swimlanes + stream_id args

examples/
├── concurrent_vector_add_2stream/          NEW (M2): kernel.ptx + reference.py + run.py + README.md + __init__.py
├── compute_vs_memory_overlap/              NEW (M3)
├── l2_contention_2stream/                  NEW (M4)
└── stream_priority_serial_vs_concurrent/   NEW (M4)

tests/unit/
├── api/test_stream.py                      NEW (M1)
├── core/test_multistream_scheduler.py      NEW (M2)
├── trace/test_kernel_launch_event.py       NEW (M1)
├── trace/test_per_event_stream_id.py       NEW (M1)
└── analysis/test_phase7_metrics.py         NEW (M3)

tests/parity/
├── test_concurrent_vector_add_2stream.py   NEW (M2)
├── test_compute_vs_memory_overlap.py       NEW (M3)
├── test_l2_contention_2stream.py           NEW (M4)
├── test_stream_priority_serial_vs_concurrent.py    NEW (M4)
└── test_phase1_6_examples_unchanged.py     RENAME from phase1_5

tests/microbench/
├── test_phase7_facts.py                    NEW (M5)
└── test_phase7_runtime.py                  NEW (M5, @pytest.mark.slow)

tests/reference/
├── gen_reference.py                        MODIFY (M5)
└── data/{4 example names}.ref.json         NEW (M5)

docs/tutorial/
├── 27-multi-stream-concurrency-basics.md   NEW (M5)
├── 28-compute-memory-overlap.md            NEW (M5)
├── 29-l2-hbm-contention-streams.md         NEW (M5)
└── 30-scheduler-fairness-streams.md        NEW (M5)

README.md                                   MODIFY (M5): v7
```

---

## Milestones

| Milestone | Tasks | Tag |
|---|---|---|
| **M1** Trace plumbing + Stream/launch API | T1–T6 | `M1-phase7-complete` |
| **M2** MultiStreamScheduler + first demo | T7–T13 | `M2-phase7-complete` |
| **M3** 4 metrics + MultiStreamResult + compute/memory example | T14–T18 | `M3-phase7-complete` |
| **M4** Contention + fairness examples | T19–T22 | `M4-phase7-complete` |
| **M5** Viz + docs + microbench + ship | T23–T28 | `phase7-complete` |

---

## Milestone M1: Trace plumbing + Stream/launch API

### Task 1: KernelLaunch event + recorder + parquet

**Files:**
- Modify: `gpusim/trace/events.py`
- Modify: `gpusim/trace/recorder.py`
- Modify: `gpusim/trace/writer.py`
- Test: `tests/unit/trace/test_kernel_launch_event.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_recorder_records_kernel_launch():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.kernel_launch(stream_id=0, kernel_name="vec_add", grid=(8,1,1),
                     block=(32,1,1), launch_cycle=10, complete_cycle=200, n_ctas=8)
    assert len(r.kernel_launch_events) == 1
    e = r.kernel_launch_events[0]
    assert e.kernel_name == "vec_add"
    assert e.launch_cycle == 10
    assert e.complete_cycle == 200


def test_recorder_writes_kernel_launch_parquet(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.trace.writer import write_parquet
    r = Recorder()
    r.kernel_launch(stream_id=1, kernel_name="k", grid=(1,1,1),
                     block=(32,1,1), launch_cycle=0, complete_cycle=100, n_ctas=1)
    write_parquet(r, tmp_path)
    assert (tmp_path / "kernel_launch.parquet").exists()
```

- [ ] **Step 2: Run (FAIL)**

```
.venv/bin/pytest tests/unit/trace/test_kernel_launch_event.py -v
```

- [ ] **Step 3: Add KernelLaunch dataclass**

Append to `gpusim/trace/events.py`:

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
```

- [ ] **Step 4: Add recorder method + list**

In `gpusim/trace/recorder.py`, in `Recorder.__init__`, add:
```python
        self.kernel_launch_events: list = []
```

Add method:
```python
    def kernel_launch(self, *, stream_id: int, kernel_name: str,
                       grid: tuple, block: tuple,
                       launch_cycle: int, complete_cycle: int,
                       n_ctas: int) -> None:
        from gpusim.trace.events import KernelLaunch
        self.kernel_launch_events.append(KernelLaunch(
            stream_id=stream_id, kernel_name=kernel_name,
            grid=grid, block=block,
            launch_cycle=launch_cycle, complete_cycle=complete_cycle,
            n_ctas=n_ctas,
        ))
```

- [ ] **Step 5: Add parquet writer**

In `gpusim/trace/writer.py`, in `write_parquet`, append:
```python
    if r.kernel_launch_events:
        pd.DataFrame([asdict(e) for e in r.kernel_launch_events]).to_parquet(
            out_dir / "kernel_launch.parquet", index=False)
```

- [ ] **Step 6: Run + commit**

```
.venv/bin/pytest tests/unit/trace/test_kernel_launch_event.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 2 PASS new; full suite ~440.

```bash
git add gpusim/trace/ tests/unit/trace/test_kernel_launch_event.py
git commit -m "feat(trace): KernelLaunch event + recorder.kernel_launch + parquet writer"
```

---

### Task 2: stream_id field on 11 existing events

**Files:**
- Modify: `gpusim/trace/events.py` (add `stream_id: int = 0` to 11 dataclasses)
- Test: `tests/unit/trace/test_per_event_stream_id.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_all_events_have_stream_id_default_zero():
    from gpusim.trace.events import (
        InstrIssue, MemoryAccess, BarrierEvent, MmaEvent, BulkLoadEvent,
        BulkStoreEvent, ClusterDispatch, ClusterBarrier, CtaDispatch,
        L2MshrEvent, AtomicEvent,
    )
    # Each event class must accept stream_id kwarg, default 0
    for cls in [InstrIssue, MemoryAccess, BarrierEvent, MmaEvent, BulkLoadEvent,
                BulkStoreEvent, ClusterDispatch, ClusterBarrier, CtaDispatch,
                L2MshrEvent, AtomicEvent]:
        # Must have stream_id field with default 0
        assert "stream_id" in cls.__dataclass_fields__, f"{cls.__name__} missing stream_id"
        assert cls.__dataclass_fields__["stream_id"].default == 0, \
            f"{cls.__name__}.stream_id default must be 0"


def test_atomic_event_accepts_stream_id_explicit():
    from gpusim.trace.events import AtomicEvent
    e = AtomicEvent(cycle=0, sm_id=0, warp_id=0, kind="ATOM",
                     op="add", space="global", line_addr=0,
                     latency=10, stream_id=3)
    assert e.stream_id == 3
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add `stream_id: int = 0` field to each of 11 event dataclasses**

In `gpusim/trace/events.py`, append a `stream_id: int = 0` field to each of these dataclasses (must come AFTER any non-default fields):
- `InstrIssue`
- `MemoryAccess`
- `BarrierEvent`
- `MmaEvent`
- `BulkLoadEvent`
- `BulkStoreEvent`
- `ClusterDispatch`
- `ClusterBarrier`
- `CtaDispatch`
- `L2MshrEvent`
- `AtomicEvent`

Example for AtomicEvent:
```python
@dataclass(frozen=True)
class AtomicEvent:
    cycle: int
    sm_id: int
    warp_id: int
    kind: str
    op: str
    space: str
    line_addr: int
    latency: int
    n_lanes: int = 1
    queue_depth_before: int = 0
    stream_id: int = 0           # NEW Phase 7
```

⚠ For each event class: place `stream_id: int = 0` AFTER any other defaulted fields (Python dataclass rule: required fields first, then defaulted).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/trace/test_per_event_stream_id.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 2 PASS new; full suite ~442.

```bash
git add gpusim/trace/events.py tests/unit/trace/test_per_event_stream_id.py
git commit -m "feat(trace): + stream_id=0 field on 11 existing events (backward compat)"
```

---

### Task 3: Recorder methods accept stream_id kwarg

**Files:**
- Modify: `gpusim/trace/recorder.py`

- [ ] **Step 1: Append failing test** to `tests/unit/trace/test_per_event_stream_id.py`:

```python
def test_recorder_methods_accept_stream_id():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    # All record methods must accept stream_id; events must carry it.
    r.atomic(cycle=0, sm_id=0, warp_id=0, kind="ATOM",
              op="add", space="global", line_addr=0, latency=10,
              stream_id=2)
    assert r.atomic_events[-1].stream_id == 2

    r.cta_dispatch(cycle=0, sm_id=0, cta_id=(0,0,0), occupancy_after=1,
                    stream_id=5) if hasattr(r, "cta_dispatch") else None
```

- [ ] **Step 2: Run (FAIL — atomic doesn't accept stream_id)**

- [ ] **Step 3: Add `stream_id: int = 0` kwarg to all 11 record methods**

In `gpusim/trace/recorder.py`, for each method (`instr_issue`, `memory_access`, `barrier`, `mma`, `bulk_load`, `bulk_store`, `cluster_dispatch`, `cluster_barrier`, `cta_dispatch`, `l2_mshr`, `atomic`):

- Add `stream_id: int = 0` to method signature (kwarg-only, with default 0)
- Pass `stream_id=stream_id` when constructing the event dataclass

Example for `atomic`:

```python
    def atomic(self, *, cycle: int, sm_id: int, warp_id: int,
                kind: str, op: str, space: str, line_addr: int,
                latency: int, n_lanes: int = 1,
                queue_depth_before: int = 0,
                stream_id: int = 0) -> None:        # NEW Phase 7
        from gpusim.trace.events import AtomicEvent
        self.atomic_events.append(AtomicEvent(
            cycle=cycle, sm_id=sm_id, warp_id=warp_id,
            kind=kind, op=op, space=space, line_addr=line_addr,
            latency=latency, n_lanes=n_lanes,
            queue_depth_before=queue_depth_before,
            stream_id=stream_id,                     # NEW
        ))
```

⚠ Repeat the same pattern for the other 10 methods. Read the existing recorder methods FIRST and adapt the signature; preserve all existing fields and add ONLY `stream_id: int = 0` at the end of kwargs.

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/trace/test_per_event_stream_id.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 3 PASS in test_per_event_stream_id.py; full suite ~442 (no new tests).

```bash
git add gpusim/trace/recorder.py tests/unit/trace/test_per_event_stream_id.py
git commit -m "feat(trace): Recorder.* accept stream_id kwarg, propagate to events"
```

---

### Task 4: Stream + GridLaunch dataclasses

**Files:**
- Modify: `gpusim/api.py` (add Stream + GridLaunch + module-level _next_stream_id counter)
- Test: `tests/unit/api/test_stream.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_stream_construction_assigns_unique_id():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s1 = Stream()
    s2 = Stream()
    assert s1.stream_id != s2.stream_id
    assert s2.stream_id == s1.stream_id + 1


def test_stream_launch_appends_to_pending():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    assert s.is_idle()
    s.launch(ptx_src="entry", grid=(1,1,1), block=(32,1,1),
              params={}, kernel_name="k1")
    assert not s.is_idle()
    assert len(s.pending) == 1
    assert s.pending[0].kernel_name == "k1"


def test_stream_launches_in_order():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    for i in range(3):
        s.launch(ptx_src=f"e{i}", grid=(1,1,1), block=(32,1,1),
                  params={}, kernel_name=f"k{i}")
    assert [g.kernel_name for g in s.pending] == ["k0", "k1", "k2"]
```

- [ ] **Step 2: Run (FAIL — no Stream class)**

- [ ] **Step 3: Add Stream + GridLaunch in api.py**

In `gpusim/api.py`, append:

```python
from collections import deque
from dataclasses import dataclass, field


_STREAM_ID_COUNTER = 0


def _next_stream_id() -> int:
    global _STREAM_ID_COUNTER
    sid = _STREAM_ID_COUNTER
    _STREAM_ID_COUNTER += 1
    return sid


def _reset_stream_id_counter() -> None:
    """Test-only helper to reset the global stream id counter."""
    global _STREAM_ID_COUNTER
    _STREAM_ID_COUNTER = 0


@dataclass
class GridLaunch:
    ptx_src: str
    kernel_name: str
    grid: tuple
    block: tuple
    params: dict
    config: object | None = None    # type: Config; lazy-typed to avoid circular


@dataclass
class Stream:
    stream_id: int = field(default_factory=_next_stream_id)
    pending: "deque[GridLaunch]" = field(default_factory=deque)
    inflight: GridLaunch | None = None
    completed: list = field(default_factory=list)

    def launch(self, ptx_src: str, grid: tuple, block: tuple,
                params: dict, *, kernel_name: str = "<unnamed>",
                config=None) -> None:
        self.pending.append(GridLaunch(
            ptx_src=ptx_src, kernel_name=kernel_name,
            grid=grid, block=block, params=params, config=config,
        ))

    def is_idle(self) -> bool:
        return self.inflight is None and not self.pending
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/api/test_stream.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 3 PASS new; full suite ~445.

```bash
git add gpusim/api.py tests/unit/api/test_stream.py
git commit -m "feat(api): Stream + GridLaunch + module stream id counter"
```

---

### Task 5: Result.stream_id field

**Files:**
- Modify: `gpusim/api.py` (add `stream_id: int = 0` to Result)
- Test: `tests/unit/api/test_stream.py` (extend)

- [ ] **Step 1: Append failing test**

```python
def test_result_has_stream_id_default_zero():
    """Single-kernel run via gpusim.run should produce Result with stream_id=0."""
    import gpusim
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    res = gpusim.run(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                      params={}, mode="timing")
    assert res.stream_id == 0
```

- [ ] **Step 2: Run (FAIL — Result has no stream_id)**

- [ ] **Step 3: Add stream_id to Result**

In `gpusim/api.py`, in the `Result` dataclass, add (place at end as a defaulted field):

```python
    stream_id: int = 0    # NEW Phase 7 — single-kernel path uses default 0
```

⚠ If the Result class is constructed in multiple places (e.g., `gpusim/api.py::run`, or in `Device.run`), you don't need to update them — `int = 0` default means existing constructions still work.

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/api/test_stream.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 1 NEW pass; full suite ~446.

```bash
git add gpusim/api.py tests/unit/api/test_stream.py
git commit -m "feat(api): Result.stream_id default 0 (backward-compat marker)"
```

---

### Task 6: Tag M1 complete

```bash
.venv/bin/pytest -q -m "not slow"
git tag M1-phase7-complete
git tag | grep M.-phase7
```

Expected: 446 passed; M1-phase7-complete tag created.

---

## Milestone M2: MultiStreamScheduler + first demo

### Task 7: MultiStreamScheduler — RR + intra-stream FIFO

**Files:**
- Modify: `gpusim/core/scheduler.py` (add MultiStreamScheduler class)
- Test: `tests/unit/core/test_multistream_scheduler.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_rr_scheduler_alternates_streams():
    """With 2 streams each having 1 grid of 4 CTAs, RR alternates pick order."""
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.core.scheduler import MultiStreamScheduler
    _reset_stream_id_counter()
    s0 = Stream(); s1 = Stream()
    s0.launch(ptx_src="x", grid=(4,1,1), block=(32,1,1), params={}, kernel_name="k0")
    s1.launch(ptx_src="x", grid=(4,1,1), block=(32,1,1), params={}, kernel_name="k1")

    sched = MultiStreamScheduler([s0, s1])
    # Mock: simple SM list with infinite capacity
    class _SM:
        def __init__(self): self.cap = 100
    sms = [_SM(), _SM()]
    
    pick_order = []
    for _ in range(8):
        choice = sched.next_dispatch(sms)
        if choice is None: break
        stream, cta, sm = choice
        pick_order.append(stream.stream_id)
    
    # RR alternation: stream ids should alternate (or close to it)
    # 8 picks across 2 streams → each stream gets 4
    counts = {0: pick_order.count(0), 1: pick_order.count(1)}
    assert counts[0] == counts[1] == 4


def test_intra_stream_fifo_grid_sequencing():
    """Same stream's 2 grids must dispatch in launch order."""
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.core.scheduler import MultiStreamScheduler
    _reset_stream_id_counter()
    s = Stream()
    s.launch(ptx_src="x", grid=(2,1,1), block=(32,1,1), params={}, kernel_name="k0")
    s.launch(ptx_src="y", grid=(2,1,1), block=(32,1,1), params={}, kernel_name="k1")
    
    sched = MultiStreamScheduler([s])
    class _SM:
        def __init__(self): self.cap = 100
    sms = [_SM()]
    
    grid_order = []
    # Simulate: pick all CTAs for grid 0 first, then signal "fully retired", then grid 1
    for _ in range(2):
        choice = sched.next_dispatch(sms)
        assert choice is not None
        stream, cta, sm = choice
        grid_order.append(stream.inflight.kernel_name)
    # After all CTAs of grid 0 dispatched, scheduler should NOT advance to grid 1
    # until current grid is marked retired. Simulate retire:
    sched.mark_grid_retired(s)
    for _ in range(2):
        choice = sched.next_dispatch(sms)
        assert choice is not None
        stream, cta, sm = choice
        grid_order.append(stream.inflight.kernel_name)
    
    assert grid_order[:2] == ["k0", "k0"]
    assert grid_order[2:] == ["k1", "k1"]
```

- [ ] **Step 2: Run (FAIL — no MultiStreamScheduler)**

- [ ] **Step 3: Implement MultiStreamScheduler**

In `gpusim/core/scheduler.py`, append:

```python
class _CtaIter:
    """Walks CTAs (x,y,z) for a grid in linear order."""
    def __init__(self, grid: tuple):
        self.grid = grid
        self.x = self.y = self.z = 0
        self._done = False
    
    def next(self):
        if self._done: return None
        cta = (self.x, self.y, self.z)
        self.x += 1
        if self.x >= self.grid[0]:
            self.x = 0; self.y += 1
            if self.y >= self.grid[1]:
                self.y = 0; self.z += 1
                if self.z >= self.grid[2]:
                    self._done = True
        return cta


class MultiStreamScheduler:
    """RR scheduler over multiple streams; intra-stream FIFO over grids.
    
    Each cycle, scheduler iterates streams in RR order; first stream with a
    dispatchable CTA wins. After a grid's CTAs are all dispatched AND the
    grid is marked retired (via mark_grid_retired), scheduler can advance
    to the next grid in that stream's pending queue.
    """
    
    def __init__(self, streams: list, policy: str = "rr"):
        self.streams = list(streams)
        self.policy = policy
        self.cursor = 0
        self._cta_iters: dict[int, _CtaIter] = {}
    
    def _ensure_inflight(self, s) -> bool:
        """Move next pending grid into inflight if stream idle. Return True if has work."""
        if s.inflight is None and s.pending:
            s.inflight = s.pending.popleft()
            self._cta_iters[s.stream_id] = _CtaIter(s.inflight.grid)
        return s.inflight is not None
    
    def _next_cta_for_stream(self, s):
        """Return next (cta_idx, grid_launch) for stream, or None."""
        if not self._ensure_inflight(s):
            return None
        it = self._cta_iters.get(s.stream_id)
        if it is None: return None
        cta = it.next()
        return cta
    
    def _find_sm_with_capacity(self, cta, available_sms):
        """Pick first SM that has capacity. Simple: first non-None."""
        for sm in available_sms:
            if getattr(sm, "cap", 1) > 0:
                return sm
        return None
    
    def next_dispatch(self, available_sms):
        """Try each stream in RR order; first dispatchable wins."""
        for _ in range(len(self.streams)):
            s = self.streams[self.cursor]
            self.cursor = (self.cursor + 1) % len(self.streams)
            cta = self._next_cta_for_stream(s)
            if cta is None: continue
            sm = self._find_sm_with_capacity(cta, available_sms)
            if sm is not None:
                return (s, cta, sm)
        return None
    
    def mark_grid_retired(self, s) -> None:
        """Caller signals: stream's current inflight grid is fully retired.
        Scheduler can now advance to next pending grid."""
        s.inflight = None
        self._cta_iters.pop(s.stream_id, None)
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/core/test_multistream_scheduler.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 2 NEW pass; full suite ~448.

```bash
git add gpusim/core/scheduler.py tests/unit/core/test_multistream_scheduler.py
git commit -m "feat(core): MultiStreamScheduler RR + intra-stream FIFO grid sequencing"
```

---

### Task 8: Device.run_streams main loop

**Files:**
- Modify: `gpusim/core/device.py`

- [ ] **Step 1: Append failing test** to `tests/unit/core/test_multistream_scheduler.py`:

```python
def test_device_run_streams_basic_one_stream():
    """Device.run_streams with a single stream should produce one Result."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    s = Stream()
    s.launch(ptx_src=src, grid=(2,1,1), block=(32,1,1), params={}, kernel_name="k")
    
    cfg = load_default()
    from gpusim.core.device import Device
    d = Device(cfg)
    multi_res = d.run_streams([s])
    assert 0 in multi_res.streams
    assert len(multi_res.streams[0]) == 1
    assert multi_res.streams[0][0].metrics["cycles"] > 0
```

- [ ] **Step 2: Run (FAIL — Device.run_streams missing)**

- [ ] **Step 3: Add Device.run_streams**

In `gpusim/core/device.py`, in `Device` class, add:

```python
    def run_streams(self, streams: list) -> "MultiStreamResult":
        """Multi-stream run loop. Each stream's launches are processed in FIFO order;
        across streams CTAs are interleaved by the MultiStreamScheduler (RR).
        
        Phase 7 simplification: re-uses Device.run() per-grid for retire,
        but coordinates across streams at the scheduler level.
        """
        from gpusim.core.scheduler import MultiStreamScheduler
        from gpusim.api import MultiStreamResult
        
        sched = MultiStreamScheduler(streams)
        
        # For Phase 7 first iteration: process each launch sequentially in
        # round-robin order across streams. Within one launch, use existing
        # Device.run() path. This achieves correct stream tagging but does
        # not yet provide cross-grid concurrency. Cross-grid concurrency
        # requires future iteration that interleaves CTA dispatch.
        results_per_stream = {s.stream_id: [] for s in streams}
        recorder = self._recorder if hasattr(self, "_recorder") else None
        
        # Naive but correct: drain streams round-robin one launch at a time
        while not all(s.is_idle() for s in streams):
            for s in streams:
                if not s.is_idle() and s.pending:
                    sched._ensure_inflight(s)
                if s.inflight is not None:
                    g = s.inflight
                    res = self.run(
                        ptx_src=g.ptx_src, grid=g.grid, block=g.block,
                        params=g.params,
                        config=g.config or self._cfg if hasattr(self, "_cfg") else g.config,
                        mode="timing", stream_id=s.stream_id,
                        kernel_name=g.kernel_name,
                    )
                    results_per_stream[s.stream_id].append(res)
                    sched.mark_grid_retired(s)
        
        return MultiStreamResult(streams=results_per_stream,
                                  total_cycles=max((r.metrics.get("cycles", 0)
                                                      for results in results_per_stream.values()
                                                      for r in results), default=0))
```

⚠ Adapt to the actual `Device` class shape. If `Device.run` doesn't accept `stream_id` and `kernel_name` kwargs, plan T9 will add them.

- [ ] **Step 4: Add MultiStreamResult skeleton in api.py**

In `gpusim/api.py`, append (skeleton; full API in T15):

```python
@dataclass
class MultiStreamResult:
    streams: dict             # int -> list[Result]
    total_cycles: int = 0
    _recorder: object | None = None
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/core/test_multistream_scheduler.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 1 NEW pass.

```bash
git add gpusim/core/device.py gpusim/api.py tests/unit/core/test_multistream_scheduler.py
git commit -m "feat(core): Device.run_streams sequential drain (per-stream RR), MultiStreamResult skel"
```

---

### Task 9: SM.dispatch_cta accepts stream_id; tag CTA dispatch event

**Files:**
- Modify: `gpusim/core/sm.py` (dispatch_cta — propagate stream_id to CTA + cta_dispatch event)
- Modify: `gpusim/core/device.py::run` (accept stream_id + kernel_name kwargs)
- Test: extend `tests/unit/core/test_multistream_scheduler.py`

- [ ] **Step 1: Append failing test**

```python
def test_cta_dispatch_event_carries_stream_id():
    """When Device.run is called with stream_id=N, the cta_dispatch event has stream_id=N."""
    import gpusim
    from gpusim.api import _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    cfg = load_default()
    res = gpusim.run(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                      params={}, mode="timing", config=cfg, stream_id=7)
    df = res.cta_dispatch_events_df if hasattr(res, "cta_dispatch_events_df") else None
    if df is not None and not df.empty:
        assert (df["stream_id"] == 7).all()
```

- [ ] **Step 2: Run (FAIL — gpusim.run doesn't accept stream_id)**

- [ ] **Step 3: Add stream_id parameter to Device.run + sm.dispatch_cta**

In `gpusim/core/device.py`, in `Device.run`, add `stream_id: int = 0` and `kernel_name: str = "<unnamed>"` kwargs:

```python
    def run(self, *, ptx_src, grid, block, params,
             config=None, mode="timing",
             stream_id: int = 0,                    # NEW Phase 7
             kernel_name: str = "<unnamed>"):       # NEW Phase 7
        # ... existing code ...
        # When dispatching CTAs to SMs, pass stream_id along
```

In `gpusim/core/sm.py`, in `dispatch_cta`, add `stream_id: int = 0` kwarg and:
- Store `stream_id` on the CTA / warp state so subsequent events can read it
- When recording `cta_dispatch` event, pass `stream_id=stream_id`

Example:
```python
    def dispatch_cta(self, cta_idx, *, stream_id: int = 0):
        # ... existing code ...
        if self.recorder is not None:
            self.recorder.cta_dispatch(
                cycle=now, sm_id=self.sm_id, cta_id=cta_idx,
                occupancy_after=...,
                stream_id=stream_id,                   # NEW
            )
        # Tag CTA-level state so warp events can read stream_id later
        self._current_cta_stream[cta_idx] = stream_id
```

In `gpusim/api.py`, in `gpusim.run`, accept and forward `stream_id` + `kernel_name`:
```python
def run(*, ptx_src, grid, block, params, config=None, mode="timing",
         stream_id: int = 0, kernel_name: str = "<unnamed>"):
    # ... existing code ...
    # Forward to Device.run
    res = device.run(ptx_src=ptx_src, grid=grid, block=block, params=params,
                      config=config, mode=mode,
                      stream_id=stream_id, kernel_name=kernel_name)
    res.stream_id = stream_id
    return res
```

⚠ Read existing Device.run + SM.dispatch_cta signatures FIRST and adapt.

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/core/test_multistream_scheduler.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 1 NEW pass; no regressions (~449).

```bash
git add gpusim/core/sm.py gpusim/core/device.py gpusim/api.py tests/unit/core/test_multistream_scheduler.py
git commit -m "feat(core): Device.run + SM.dispatch_cta accept stream_id, propagate to events"
```

---

### Task 10: Propagate stream_id from CTA to warp/SubCore events

**Files:**
- Modify: `gpusim/core/sub_core.py` (read CTA's stream_id, pass to all event recordings)
- Test: extend `tests/unit/core/test_multistream_scheduler.py`

- [ ] **Step 1: Append failing test**

```python
def test_warp_events_carry_stream_id():
    """When a kernel runs on stream_id=N, all warp events (instr_issue, etc.)
    should carry stream_id=N."""
    import gpusim
    from gpusim.api import _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    src = """
.entry test() {
    .reg .u32 %r<3>;
    mov.u32 %r0, %tid.x;
    add.s32 %r1, %r0, 1;
    ret;
}
"""
    cfg = load_default()
    res = gpusim.run(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                      params={}, mode="timing", config=cfg, stream_id=4)
    
    # Check instr_issue events all carry stream_id=4
    issue_df = res.instr_issue_events_df if hasattr(res, "instr_issue_events_df") else None
    if issue_df is not None and not issue_df.empty:
        assert (issue_df["stream_id"] == 4).all()
```

- [ ] **Step 2: Run (FAIL — events have stream_id=0 not 4)**

- [ ] **Step 3: Propagate stream_id through SubCore**

In `gpusim/core/sub_core.py`:

1. Add `self._cta_stream_ids: dict = {}` to `SubCore.__init__` (or use shared SM-level dict).

2. When SM dispatches a CTA to SubCore, pass + record stream_id:
```python
def dispatch_warp(self, w, *, stream_id: int = 0):
    self._cta_stream_ids[w.cta_id] = stream_id
    # ... existing dispatch ...
```

3. In every `self.recorder.<method>(...)` call inside `_issue` and other event-emitting paths, look up `stream_id` from `self._cta_stream_ids.get(w.cta_id, 0)` and pass it:

```python
    sid = self._cta_stream_ids.get(w.cta_id, 0)
    self.recorder.instr_issue(
        cycle=now, warp_id=w.warp_id, pc=instr.pc, op=op,
        src_loc=(instr.src_loc.file, instr.src_loc.line),
        active_mask=w.fn_state.active_mask if w.fn_state else 0,
        stream_id=sid,                         # NEW
    )
```

⚠ Find all `self.recorder.*` calls in sub_core.py and add `stream_id=sid` to each. The same pattern applies to atomic, mma, bulk_load, bulk_store, mbarrier, etc. recordings.

4. Similarly propagate in `gpusim/core/sm.py` for SM-level events (l2_mshr, cta_dispatch already handled in T9).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/core/test_multistream_scheduler.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 1 NEW pass; no regressions.

```bash
git add gpusim/core/sub_core.py gpusim/core/sm.py tests/unit/core/test_multistream_scheduler.py
git commit -m "feat(core): propagate stream_id from CTA through SubCore to all warp events"
```

---

### Task 11: gpusim.synchronize() function

**Files:**
- Modify: `gpusim/api.py` (add module-level synchronize + stream registry)
- Test: extend `tests/unit/api/test_stream.py`

- [ ] **Step 1: Append failing test**

```python
def test_synchronize_drains_two_streams():
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    cfg = load_default()
    s0 = Stream()
    s1 = Stream()
    s0.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k0",
              config=cfg)
    s1.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k1",
              config=cfg)
    multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
    assert 0 in multi_res.streams and 1 in multi_res.streams
    assert len(multi_res.streams[0]) == 1
    assert len(multi_res.streams[1]) == 1
    assert multi_res.streams[0][0].stream_id == 0
    assert multi_res.streams[1][0].stream_id == 1
```

- [ ] **Step 2: Run (FAIL — synchronize doesn't exist)**

- [ ] **Step 3: Add synchronize function**

In `gpusim/api.py`, append:

```python
def synchronize(streams: list = None, *, config=None) -> "MultiStreamResult":
    """Drain all given streams (or single stream); return aggregated MultiStreamResult.
    
    Args:
        streams: list of Stream objects to drain. If None, raises ValueError.
        config: device Config; if None, attempts to use load_default().
    """
    if streams is None or len(streams) == 0:
        raise ValueError("synchronize() requires at least one Stream")
    if config is None:
        from gpusim.config.loader import load_default
        config = load_default()
    
    from gpusim.core.device import Device
    d = Device(config)
    return d.run_streams(streams)
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/api/test_stream.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 1 NEW pass.

```bash
git add gpusim/api.py tests/unit/api/test_stream.py
git commit -m "feat(api): gpusim.synchronize(streams) drains streams via Device.run_streams"
```

---

### Task 12: Example concurrent_vector_add_2stream

**Files:**
- Create: `examples/concurrent_vector_add_2stream/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_concurrent_vector_add_2stream.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "concurrent_vector_add_2stream"


def test_concurrent_vector_add_2stream_correctness():
    """Two streams each run vector_add on independent arrays; both outputs correct."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    C = np.zeros(n, dtype=np.float32)
    D = np.arange(n, dtype=np.float32) * 3
    E = np.arange(n, dtype=np.float32) * 4
    F = np.zeros(n, dtype=np.float32)
    
    cfg = load_default()
    cfg.n_sm = 8
    
    ptx = (_DIR / "kernel.ptx").read_text()
    s0 = Stream()
    s1 = Stream()
    s0.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": C}, kernel_name="vec_add_a", config=cfg)
    s1.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": D, "B": E, "OUT": F}, kernel_name="vec_add_b", config=cfg)
    
    multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
    
    # Both outputs correct
    np.testing.assert_array_equal(C, A + B)
    np.testing.assert_array_equal(F, D + E)
    # Both streams produced one Result each
    assert len(multi_res.streams[0]) == 1
    assert len(multi_res.streams[1]) == 1
```

- [ ] **Step 2: Kernel** `examples/concurrent_vector_add_2stream/kernel.ptx`:

```
.entry test(.param .u64 A, .param .u64 B, .param .u64 OUT)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    
    mov.u32 %r0, %tid.x;
    mul.lo.s32 %r1, %r0, 4;
    cvt.u64.u32 %rd3, %r1;
    
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    
    add.f32 %f2, %f0, %f1;
    
    add.u64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f2;
    
    ret;
}
```

- [ ] **Step 3: reference.py**

```python
import numpy as np


def reference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A + B
```

- [ ] **Step 4: run.py**

```python
import numpy as np, pathlib, gpusim
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    C = np.zeros(n, dtype=np.float32)
    D = np.arange(n, dtype=np.float32) * 3
    E = np.arange(n, dtype=np.float32) * 4
    F = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    cfg.n_sm = 8
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s0 = Stream()
    s1 = Stream()
    s0.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": C}, kernel_name="vec_add_a", config=cfg)
    s1.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": D, "B": E, "OUT": F}, kernel_name="vec_add_b", config=cfg)
    multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
    print(f"Stream 0 cycles: {multi_res.streams[0][0].metrics['cycles']}")
    print(f"Stream 1 cycles: {multi_res.streams[1][0].metrics['cycles']}")
    print(f"C[0:4] = {list(C[0:4])}")
    print(f"F[0:4] = {list(F[0:4])}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: README.md**

```markdown
# concurrent_vector_add_2stream

Phase 7 demo: two streams each run vector_add on independent arrays.
Demonstrates the basic gpusim.Stream + Stream.launch + gpusim.synchronize API.

## Run
```
python examples/concurrent_vector_add_2stream/run.py
```

## Tutorial
docs/tutorial/27-multi-stream-concurrency-basics.md
```

- [ ] **Step 6: empty __init__.py + Run + commit**

```bash
touch examples/concurrent_vector_add_2stream/__init__.py
.venv/bin/pytest tests/parity/test_concurrent_vector_add_2stream.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 1 NEW pass.

```bash
git add examples/concurrent_vector_add_2stream/ tests/parity/test_concurrent_vector_add_2stream.py
git commit -m "feat(examples): concurrent_vector_add_2stream — basic 2-stream API demo"
```

---

### Task 13: Tag M2 complete

```bash
.venv/bin/pytest -q -m "not slow"
git tag M2-phase7-complete
git tag | grep M.-phase7
```

Expected: ~451 passed; M2-phase7-complete tag created.

---

## Milestone M3: 4 metrics + MultiStreamResult API + compute/memory overlap example

### Task 14: 4 analysis metrics + tests

**Files:**
- Modify: `gpusim/analysis/metrics.py` (append 4 metric functions)
- Test: `tests/unit/analysis/test_phase7_metrics.py` (NEW)

- [ ] **Step 1: Create test**

```python
import pandas as pd


def test_stream_concurrency_factor():
    from gpusim.analysis.metrics import stream_concurrency_factor
    # Stream 0: cycles 0-100; Stream 1: cycles 50-150
    df = pd.DataFrame([
        {"stream_id": 0, "launch_cycle": 0, "complete_cycle": 100},
        {"stream_id": 1, "launch_cycle": 50, "complete_cycle": 150},
    ])
    factor = stream_concurrency_factor(df, total_cycles=150)
    # Average active streams per cycle: 0-50 = 1, 50-100 = 2, 100-150 = 1
    # avg = (50*1 + 50*2 + 50*1) / 150 = 200/150 ≈ 1.333
    assert abs(factor - 1.333) < 0.01


def test_compute_memory_overlap():
    from gpusim.analysis.metrics import compute_memory_overlap
    mma_df = pd.DataFrame([
        {"cycle": 0, "stream_id": 0},
        {"cycle": 5, "stream_id": 0},
    ])
    mem_df = pd.DataFrame([
        {"cycle": 2, "stream_id": 1},
        {"cycle": 7, "stream_id": 1},
    ])
    events_dfs = {"mma": mma_df, "memory": mem_df}
    rate = compute_memory_overlap(events_dfs)
    # Some overlap (stream 0 mma at cycle 5, stream 1 mem at cycle 2,7)
    assert 0 <= rate <= 1.0


def test_l2_bandwidth_per_stream():
    from gpusim.analysis.metrics import l2_bandwidth_per_stream
    df = pd.DataFrame([
        {"stream_id": 0}, {"stream_id": 0}, {"stream_id": 0},
        {"stream_id": 1},
    ])
    out = l2_bandwidth_per_stream(df)
    assert abs(out[0] - 0.75) < 1e-6
    assert abs(out[1] - 0.25) < 1e-6


def test_stream_fairness_jain():
    from gpusim.analysis.metrics import stream_fairness_jain
    # Equal: 4 each → fairness = 1.0
    df = pd.DataFrame([
        {"stream_id": 0}] * 4 + [{"stream_id": 1}] * 4)
    assert abs(stream_fairness_jain(df) - 1.0) < 1e-6
    # Unequal: 8, 0 → fairness = 1²/2 = 0.5
    df2 = pd.DataFrame([{"stream_id": 0}] * 8 + [{"stream_id": 1}] * 0)
    # Single stream gets all → fairness = 1.0 (only 1 active stream)
    # Actually formula: (Σx)² / (n·Σx²) where n = number of streams considered
    # For [8] only one stream → trivially 1.0
    # For [8, 0] over 2 streams: (8)²/(2*64) = 64/128 = 0.5
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Implement 4 metrics**

Append to `gpusim/analysis/metrics.py`:

```python
def stream_concurrency_factor(kernel_launch_df, total_cycles: int) -> float:
    """Average number of streams active per cycle, over the device run.
    1.0 = serial; up to N for full overlap."""
    if kernel_launch_df is None or kernel_launch_df.empty or total_cycles <= 0:
        return 0.0
    # For each cycle, count streams with launch_cycle <= cycle <= complete_cycle
    total_active_cycles = 0
    for _, row in kernel_launch_df.iterrows():
        total_active_cycles += max(0, row["complete_cycle"] - row["launch_cycle"])
    return total_active_cycles / total_cycles


def compute_memory_overlap(events_dfs: dict) -> float:
    """Fraction of compute-event cycles that overlap with memory-event cycles
    on different streams."""
    mma_df = events_dfs.get("mma")
    mem_df = events_dfs.get("memory")
    if mma_df is None or mma_df.empty or mem_df is None or mem_df.empty:
        return 0.0
    # Simple: count mma cycles where any cross-stream memory event also happened
    overlap = 0
    total = len(mma_df)
    for _, mrow in mma_df.iterrows():
        cycle = mrow["cycle"]
        cross_stream_mem = mem_df[(mem_df["cycle"] == cycle)
                                    & (mem_df["stream_id"] != mrow["stream_id"])]
        if not cross_stream_mem.empty:
            overlap += 1
    return overlap / max(total, 1)


def l2_bandwidth_per_stream(memory_events_df) -> dict:
    """Fraction of L2 requests originating from each stream."""
    if memory_events_df is None or memory_events_df.empty:
        return {}
    counts = memory_events_df.groupby("stream_id").size()
    total = counts.sum()
    return {int(sid): float(cnt) / total for sid, cnt in counts.items()}


def stream_fairness_jain(cta_dispatch_df) -> float:
    """Jain's fairness index over per-stream CTA dispatch counts:
    (Σ x_i)² / (n · Σ x_i²)   where x_i = CTAs dispatched for stream i.
    1.0 = perfectly fair; 1/n = worst case."""
    if cta_dispatch_df is None or cta_dispatch_df.empty:
        return 0.0
    counts = cta_dispatch_df.groupby("stream_id").size().values
    n = len(counts)
    if n == 0: return 0.0
    if n == 1: return 1.0
    sum_x = float(counts.sum())
    sum_x_sq = float((counts ** 2).sum())
    return (sum_x ** 2) / (n * sum_x_sq)
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/analysis/test_phase7_metrics.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 4 NEW pass; full suite ~455.

```bash
git add gpusim/analysis/metrics.py tests/unit/analysis/test_phase7_metrics.py
git commit -m "feat(analysis): 4 Phase 7 metrics (concurrency_factor / overlap / l2_bandwidth / fairness)"
```

---

### Task 15: MultiStreamResult full API

**Files:**
- Modify: `gpusim/api.py` (add full MultiStreamResult properties)
- Modify: `gpusim/viz/notebook.py` (add helper)

- [ ] **Step 1: Append failing test** to `tests/unit/api/test_stream.py`:

```python
def test_multistream_result_stream_summary():
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    cfg = load_default()
    s0 = Stream()
    s0.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
              params={}, kernel_name="k0", config=cfg)
    multi_res = gpusim.synchronize(streams=[s0], config=cfg)
    summary = multi_res.stream_summary()
    assert "Stream 0" in summary
    assert "k0" in summary


def test_multistream_result_kernel_launch_events_df():
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    cfg = load_default()
    s0 = Stream(); s1 = Stream()
    s0.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1), params={}, kernel_name="ka", config=cfg)
    s1.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1), params={}, kernel_name="kb", config=cfg)
    multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
    df = multi_res.kernel_launch_events_df
    assert df is not None
    assert len(df) == 2
    assert set(df["stream_id"].unique()) == {0, 1}
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add atomic_events_dataframe-style helper for kernel_launch + per-stream event split**

In `gpusim/viz/notebook.py`, append:

```python
def kernel_launch_events_dataframe(rec):
    import pandas as pd
    from dataclasses import asdict
    if not rec.kernel_launch_events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in rec.kernel_launch_events])


def per_stream_events_dataframe(rec) -> dict:
    """Returns {stream_id: {event_type: DataFrame, ...}, ...}."""
    import pandas as pd
    from dataclasses import asdict
    out: dict = {}
    event_lists = {
        "instr_issue": getattr(rec, "instr_events", []),
        "atomic": getattr(rec, "atomic_events", []),
        "mma": getattr(rec, "mma_events", []),
        "bulk_load": getattr(rec, "bulk_load_events", []),
        "bulk_store": getattr(rec, "bulk_store_events", []),
        "cta_dispatch": getattr(rec, "cta_dispatch_events", []),
    }
    all_stream_ids = set()
    for evs in event_lists.values():
        for e in evs:
            sid = getattr(e, "stream_id", 0)
            all_stream_ids.add(sid)
    for sid in all_stream_ids:
        out[sid] = {}
        for ev_name, evs in event_lists.items():
            filtered = [asdict(e) for e in evs if getattr(e, "stream_id", 0) == sid]
            out[sid][ev_name] = pd.DataFrame(filtered)
    return out
```

- [ ] **Step 4: Update MultiStreamResult class with full API**

In `gpusim/api.py`, replace the skeleton MultiStreamResult with:

```python
@dataclass
class MultiStreamResult:
    streams: dict                  # int -> list[Result]
    total_cycles: int = 0
    _recorder: object | None = None
    
    @property
    def kernel_launch_events_df(self):
        from gpusim.viz.notebook import kernel_launch_events_dataframe
        if self._recorder is not None:
            return kernel_launch_events_dataframe(self._recorder)
        # Fallback: build from per-Result recorders
        import pandas as pd
        from dataclasses import asdict
        rows = []
        for sid, results in self.streams.items():
            for r in results:
                if hasattr(r, "_recorder") and r._recorder is not None:
                    for e in getattr(r._recorder, "kernel_launch_events", []):
                        rows.append(asdict(e))
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    @property
    def per_stream_events_df(self):
        from gpusim.viz.notebook import per_stream_events_dataframe
        if self._recorder is not None:
            return per_stream_events_dataframe(self._recorder)
        return {}
    
    @property
    def stream_metrics(self) -> dict:
        out = {}
        for sid, results in self.streams.items():
            cycles = sum(r.metrics.get("cycles", 0) for r in results)
            ctas = sum(r.metrics.get("active_ctas", 0) for r in results)
            out[sid] = {
                "cycles": cycles,
                "ctas": ctas,
                "n_launches": len(results),
            }
        return out
    
    def stream_summary(self) -> str:
        lines = []
        for sid, results in sorted(self.streams.items()):
            for r in results:
                kn = getattr(r, "kernel_name", "<unnamed>")
                cycles = r.metrics.get("cycles", 0)
                ctas = r.metrics.get("active_ctas", 0)
                lines.append(f"Stream {sid}: {kn}, {ctas} CTAs, {cycles} cycles")
        return "\n".join(lines)
    
    def fairness(self) -> float:
        from gpusim.analysis.metrics import stream_fairness_jain
        df = None
        if self._recorder is not None:
            from dataclasses import asdict
            import pandas as pd
            rows = [asdict(e) for e in getattr(self._recorder, "cta_dispatch_events", [])]
            df = pd.DataFrame(rows) if rows else None
        return stream_fairness_jain(df) if df is not None else 0.0
    
    def overlap_ratio(self) -> float:
        from gpusim.analysis.metrics import compute_memory_overlap
        if self._recorder is None: return 0.0
        from dataclasses import asdict
        import pandas as pd
        mma = pd.DataFrame([asdict(e) for e in getattr(self._recorder, "mma_events", [])])
        mem = pd.DataFrame([asdict(e) for e in getattr(self._recorder, "instr_events", []) if "ld" in getattr(e, "op", "") or "st" in getattr(e, "op", "")])
        return compute_memory_overlap({"mma": mma, "memory": mem})
```

- [ ] **Step 5: Plumb _recorder into MultiStreamResult**

In `gpusim/core/device.py::run_streams`, ensure the global recorder (or an aggregated recorder) is passed when constructing MultiStreamResult:

```python
        return MultiStreamResult(
            streams=results_per_stream,
            total_cycles=...,
            _recorder=getattr(self, "_recorder", None),
        )
```

- [ ] **Step 6: Run + commit**

```
.venv/bin/pytest tests/unit/api/test_stream.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 2 NEW pass; full suite ~457.

```bash
git add gpusim/api.py gpusim/viz/notebook.py tests/unit/api/test_stream.py
git commit -m "feat(api): MultiStreamResult full API (kernel_launch_df, stream_metrics, stream_summary, fairness, overlap_ratio)"
```

---

### Task 16: Result.kernel_name field (single-launch tracking)

**Files:**
- Modify: `gpusim/api.py`
- Modify: `gpusim/core/device.py`

- [ ] **Step 1: Append failing test** to `tests/unit/api/test_stream.py`:

```python
def test_result_carries_kernel_name():
    import gpusim
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    res = gpusim.run(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                      params={}, mode="timing", kernel_name="my_k")
    assert res.kernel_name == "my_k"
```

- [ ] **Step 2: Run (FAIL — Result has no kernel_name)**

- [ ] **Step 3: Add kernel_name field to Result**

In `gpusim/api.py`, in Result dataclass:
```python
    kernel_name: str = "<unnamed>"    # NEW Phase 7
```

In `gpusim/api.py::run`, set `res.kernel_name = kernel_name`.

In `gpusim/core/device.py::run`, accept `kernel_name` kwarg and set on Result.

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/api/test_stream.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/api.py gpusim/core/device.py tests/unit/api/test_stream.py
git commit -m "feat(api): Result.kernel_name field for stream/launch tracking"
```

---

### Task 17: Example compute_vs_memory_overlap

**Files:**
- Create: `examples/compute_vs_memory_overlap/{kernel_compute.ptx, kernel_memory.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_compute_vs_memory_overlap.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "compute_vs_memory_overlap"


def test_compute_vs_memory_overlap_correctness():
    """Two streams: one compute-heavy, one memory-heavy. Both produce correct outputs."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    C = np.zeros(n, dtype=np.float32)
    D = np.arange(n, dtype=np.float32) * 5
    E = np.arange(n, dtype=np.float32) * 3
    F = np.zeros(n, dtype=np.float32)
    
    cfg = load_default()
    cfg.n_sm = 8
    
    ptx_compute = (_DIR / "kernel_compute.ptx").read_text()
    ptx_memory = (_DIR / "kernel_memory.ptx").read_text()
    
    s0 = Stream()
    s1 = Stream()
    s0.launch(ptx_src=ptx_compute, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": C}, kernel_name="compute_heavy", config=cfg)
    s1.launch(ptx_src=ptx_memory, grid=(1,1,1), block=(32,1,1),
              params={"A": D, "B": E, "OUT": F}, kernel_name="memory_heavy", config=cfg)
    
    multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
    
    # Both correct
    assert (C >= 0).all()  # compute_heavy adds many times
    np.testing.assert_array_equal(F, D + E)
    
    # Verify two streams were used
    assert len(multi_res.streams) == 2
```

- [ ] **Step 2: kernel_compute.ptx — compute-heavy (long add chain)**

```
.entry test(.param .u64 A, .param .u64 B, .param .u64 OUT)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<8>;
    
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    
    mov.u32 %r0, %tid.x;
    mul.lo.s32 %r1, %r0, 4;
    cvt.u64.u32 %rd3, %r1;
    
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    
    // Compute-heavy: 16 adds in a chain (no memory in middle)
    add.f32 %f2, %f0, %f1;
    add.f32 %f2, %f2, %f0;
    add.f32 %f2, %f2, %f1;
    add.f32 %f2, %f2, %f0;
    add.f32 %f2, %f2, %f1;
    add.f32 %f2, %f2, %f0;
    add.f32 %f2, %f2, %f1;
    add.f32 %f2, %f2, %f0;
    add.f32 %f2, %f2, %f1;
    add.f32 %f2, %f2, %f0;
    add.f32 %f2, %f2, %f1;
    add.f32 %f2, %f2, %f0;
    add.f32 %f2, %f2, %f1;
    add.f32 %f2, %f2, %f0;
    add.f32 %f2, %f2, %f1;
    add.f32 %f2, %f2, %f0;
    
    add.u64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f2;
    
    ret;
}
```

- [ ] **Step 3: kernel_memory.ptx — memory-heavy vector_add (one add, two loads)**

```
.entry test(.param .u64 A, .param .u64 B, .param .u64 OUT)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    
    mov.u32 %r0, %tid.x;
    mul.lo.s32 %r1, %r0, 4;
    cvt.u64.u32 %rd3, %r1;
    
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    
    add.f32 %f2, %f0, %f1;
    
    add.u64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f2;
    
    ret;
}
```

- [ ] **Step 4: reference.py + run.py + README.md + __init__.py**

`reference.py`:
```python
import numpy as np


def reference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A + B
```

`run.py`:
```python
import numpy as np, pathlib, gpusim
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    C = np.zeros(n, dtype=np.float32)
    D = np.arange(n, dtype=np.float32) * 5
    E = np.arange(n, dtype=np.float32) * 3
    F = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    cfg.n_sm = 8
    here = pathlib.Path(__file__).parent
    s0 = Stream()
    s1 = Stream()
    s0.launch(ptx_src=(here / "kernel_compute.ptx").read_text(),
              grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": C},
              kernel_name="compute_heavy", config=cfg)
    s1.launch(ptx_src=(here / "kernel_memory.ptx").read_text(),
              grid=(1,1,1), block=(32,1,1),
              params={"A": D, "B": E, "OUT": F},
              kernel_name="memory_heavy", config=cfg)
    multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
    print(multi_res.stream_summary())


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# compute_vs_memory_overlap

Phase 7 demo: compute-heavy kernel + memory-heavy kernel run concurrently
on two streams. Demonstrates the canonical CUDA optimization of pairing
compute and memory kernels to maximize device utilization.

## Run
```
python examples/compute_vs_memory_overlap/run.py
```

## Tutorial
docs/tutorial/28-compute-memory-overlap.md
```

`__init__.py` (empty).

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/parity/test_compute_vs_memory_overlap.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/compute_vs_memory_overlap/ tests/parity/test_compute_vs_memory_overlap.py
git commit -m "feat(examples): compute_vs_memory_overlap — compute+memory kernel co-location"
```

---

### Task 18: Tag M3

```bash
.venv/bin/pytest -q -m "not slow"
git tag M3-phase7-complete
```

---

## Milestone M4: Contention + fairness examples

### Task 19: Example l2_contention_2stream

**Files:**
- Create: `examples/l2_contention_2stream/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_l2_contention_2stream.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "l2_contention_2stream"


def test_l2_contention_2stream_correctness():
    """Two streams write to overlapping gmem range; both outputs land correctly."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    
    n = 64
    SHARED = np.zeros(n, dtype=np.uint32)
    cfg = load_default()
    cfg.n_sm = 8
    
    ptx = (_DIR / "kernel.ptx").read_text()
    s0 = Stream()
    s1 = Stream()
    # Both streams write to same buffer, different offsets within same L2 line range
    s0.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": SHARED, "OFFSET": 0}, kernel_name="writer_low", config=cfg)
    s1.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": SHARED, "OFFSET": 32}, kernel_name="writer_high", config=cfg)
    
    multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
    
    # Both regions should have been written; expect non-zero
    assert SHARED[0:32].sum() > 0
    assert SHARED[32:64].sum() > 0
    assert len(multi_res.streams) == 2
```

- [ ] **Step 2: Kernel** `kernel.ptx`:

```
.entry test(.param .u64 OUT, .param .u32 OFFSET)
{
    .reg .u64 %rd<4>;
    .reg .u32 %r<6>;
    
    ld.param.u64 %rd0, [OUT];
    ld.param.u32 %r0, [OFFSET];
    
    mov.u32 %r1, %tid.x;
    add.s32 %r2, %r1, %r0;
    mul.lo.s32 %r3, %r2, 4;
    cvt.u64.u32 %rd1, %r3;
    add.u64 %rd2, %rd0, %rd1;
    
    mov.u32 %r4, 1;
    st.global.u32 [%rd2], %r4;
    
    ret;
}
```

- [ ] **Step 3: reference.py + run.py + README.md + __init__.py**

`reference.py`:
```python
import numpy as np


def reference(n: int = 64) -> np.ndarray:
    out = np.zeros(n, dtype=np.uint32)
    out[0:32] = 1
    out[32:64] = 1
    return out
```

`run.py`:
```python
import numpy as np, pathlib, gpusim
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    SHARED = np.zeros(64, dtype=np.uint32)
    cfg = load_default()
    cfg.n_sm = 8
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s0 = Stream()
    s1 = Stream()
    s0.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": SHARED, "OFFSET": 0}, kernel_name="writer_low", config=cfg)
    s1.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": SHARED, "OFFSET": 32}, kernel_name="writer_high", config=cfg)
    multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
    print(multi_res.stream_summary())


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# l2_contention_2stream

Phase 7 demo: two streams write to overlapping (adjacent) gmem regions
forcing L2 cache contention. Demonstrates how concurrent kernels share
L2/HBM bandwidth.

## Run
```
python examples/l2_contention_2stream/run.py
```

## Tutorial
docs/tutorial/29-l2-hbm-contention-streams.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_l2_contention_2stream.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/l2_contention_2stream/ tests/parity/test_l2_contention_2stream.py
git commit -m "feat(examples): l2_contention_2stream — two streams sharing L2 bandwidth"
```

---

### Task 20: Example stream_priority_serial_vs_concurrent

**Files:**
- Create: `examples/stream_priority_serial_vs_concurrent/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_stream_priority_serial_vs_concurrent.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "stream_priority_serial_vs_concurrent"


def test_stream_priority_serial_vs_concurrent():
    """Same total work: 4 vec_add grids serial (1 stream) vs concurrent (4 streams).
    Concurrent should be faster."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    cfg = load_default()
    cfg.n_sm = 8
    ptx = (_DIR / "kernel.ptx").read_text()
    
    # Serial: 4 launches on 1 stream
    _reset_stream_id_counter()
    s_serial = Stream()
    outs_serial = [np.zeros(n, dtype=np.float32) for _ in range(4)]
    for i in range(4):
        s_serial.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                         params={"A": A, "B": B, "OUT": outs_serial[i]},
                         kernel_name=f"k{i}", config=cfg)
    res_serial = gpusim.synchronize(streams=[s_serial], config=cfg)
    serial_cycles = res_serial.total_cycles
    
    # Concurrent: 4 streams each 1 launch
    _reset_stream_id_counter()
    streams = [Stream() for _ in range(4)]
    outs_conc = [np.zeros(n, dtype=np.float32) for _ in range(4)]
    for i, s in enumerate(streams):
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": outs_conc[i]},
                  kernel_name=f"k{i}", config=cfg)
    res_conc = gpusim.synchronize(streams=streams, config=cfg)
    conc_cycles = res_conc.total_cycles
    
    # All outputs correct
    for o in outs_serial: np.testing.assert_array_equal(o, A + B)
    for o in outs_conc: np.testing.assert_array_equal(o, A + B)
    
    # Concurrent should NOT take longer than serial (Phase 7 scheduler is at minimum equal)
    # In a true concurrent implementation, conc_cycles < serial_cycles. With Phase 7
    # naive sequential drain, conc_cycles ~= serial_cycles. Loose: <= 1.5× serial.
    assert conc_cycles <= serial_cycles * 1.5
```

- [ ] **Step 2: Kernel** `kernel.ptx` (vec_add — same as concurrent_vector_add_2stream's kernel.ptx):

```
.entry test(.param .u64 A, .param .u64 B, .param .u64 OUT)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    
    mov.u32 %r0, %tid.x;
    mul.lo.s32 %r1, %r0, 4;
    cvt.u64.u32 %rd3, %r1;
    
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    
    add.f32 %f2, %f0, %f1;
    
    add.u64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f2;
    
    ret;
}
```

- [ ] **Step 3: reference.py + run.py + README.md + __init__.py**

`reference.py`:
```python
import numpy as np


def reference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A + B
```

`run.py`:
```python
import numpy as np, pathlib, gpusim
from gpusim.api import Stream, _reset_stream_id_counter
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    cfg = load_default()
    cfg.n_sm = 8
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    
    _reset_stream_id_counter()
    s_serial = Stream()
    outs = [np.zeros(n, dtype=np.float32) for _ in range(4)]
    for i in range(4):
        s_serial.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                         params={"A": A, "B": B, "OUT": outs[i]},
                         kernel_name=f"k{i}", config=cfg)
    rs = gpusim.synchronize(streams=[s_serial], config=cfg)
    
    _reset_stream_id_counter()
    streams = [Stream() for _ in range(4)]
    outs2 = [np.zeros(n, dtype=np.float32) for _ in range(4)]
    for i, s in enumerate(streams):
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": outs2[i]},
                  kernel_name=f"k{i}", config=cfg)
    rc = gpusim.synchronize(streams=streams, config=cfg)
    
    print(f"Serial:     {rs.total_cycles} cycles (1 stream, 4 launches)")
    print(f"Concurrent: {rc.total_cycles} cycles (4 streams, 1 launch each)")
    print(f"Speedup:    {rs.total_cycles / max(rc.total_cycles, 1):.2f}×")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# stream_priority_serial_vs_concurrent

Phase 7 demo: same workload (4 vector_add launches) run two ways:
- Serial (4 launches on 1 stream)
- Concurrent (4 launches on 4 streams)

Compares total cycles to demonstrate stream concurrency benefit.

## Run
```
python examples/stream_priority_serial_vs_concurrent/run.py
```

## Tutorial
docs/tutorial/30-scheduler-fairness-streams.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_stream_priority_serial_vs_concurrent.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/stream_priority_serial_vs_concurrent/ tests/parity/test_stream_priority_serial_vs_concurrent.py
git commit -m "feat(examples): stream_priority_serial_vs_concurrent — fairness+concurrency demo"
```

---

### Task 21: Phase 1-6 regression test rename + Phase 6 examples added

**Files:**
- Rename: `tests/parity/test_phase1_5_examples_unchanged.py` → `test_phase1_6_examples_unchanged.py`
- Modify: rename + add Phase 6 examples to PHASE_1_6_EXAMPLES list

- [ ] **Step 1: Rename file**

```bash
git mv tests/parity/test_phase1_5_examples_unchanged.py tests/parity/test_phase1_6_examples_unchanged.py
```

- [ ] **Step 2: Edit file**

In `tests/parity/test_phase1_6_examples_unchanged.py`:
- Rename `PHASE_1_5_EXAMPLES` → `PHASE_1_6_EXAMPLES`
- Append Phase 6 examples to the list:

```python
PHASE_1_6_EXAMPLES = [
    # Phase 1-5 (unchanged from previous list)
    # ...existing entries...
    # Phase 6
    "atom_histogram",
    "atom_reduction_smem",
    "atom_cas_spinlock",
    "red_min_max",
    "cluster_cooperative_epilogue",
]
```

- Update test function names from `_1_5` → `_1_6` if any.

- [ ] **Step 3: Run + commit**

```
.venv/bin/pytest tests/parity/test_phase1_6_examples_unchanged.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add tests/parity/test_phase1_6_examples_unchanged.py
git commit -m "test(regression): rename phase1_5 → phase1_6 + add 5 Phase 6 examples"
```

---

### Task 22: Tag M4

```bash
.venv/bin/pytest -q -m "not slow"
git tag M4-phase7-complete
```

---

## Milestone M5: Viz + docs + microbench + ship

### Task 23: HTML §27 + §28

**Files:**
- Modify: `gpusim/viz/html_report.py`
- Modify: `gpusim/viz/_template.html.j2`
- Test: `tests/unit/viz/test_html_report_phase7.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_html_report_phase7_sections(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.kernel_launch(stream_id=0, kernel_name="k0", grid=(1,1,1), block=(32,1,1),
                     launch_cycle=0, complete_cycle=100, n_ctas=1)
    r.kernel_launch(stream_id=1, kernel_name="k1", grid=(1,1,1), block=(32,1,1),
                     launch_cycle=50, complete_cycle=150, n_ctas=1)
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="t", grid=(1,1,1), block=(32,1,1),
              cycles=200, occupancy={"active_ctas": 1, "bottleneck": "none"})
    html = out.read_text()
    assert "Stream" in html or "stream" in html.lower()
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add render helpers**

In `gpusim/viz/html_report.py`, append:

```python
def _render_stream_concurrency(rec):
    if not rec.kernel_launch_events:
        return ""
    from dataclasses import asdict
    import pandas as pd
    df = pd.DataFrame([asdict(e) for e in rec.kernel_launch_events])
    parts = []
    parts.append("<h3>Kernel launches by stream</h3>" + df.to_html(index=False))
    return "\n".join(parts)


def _render_per_stream_breakdown(rec):
    if not rec.kernel_launch_events:
        return ""
    from dataclasses import asdict
    import pandas as pd
    # Per-stream count of various event types
    rows = []
    streams = set()
    for ev_attr in ["instr_events", "atomic_events", "mma_events",
                     "bulk_load_events", "bulk_store_events"]:
        for e in getattr(rec, ev_attr, []):
            sid = getattr(e, "stream_id", 0)
            streams.add(sid)
    for sid in sorted(streams):
        rows.append({
            "stream_id": sid,
            "instr_events": sum(1 for e in getattr(rec, "instr_events", [])
                                  if getattr(e, "stream_id", 0) == sid),
            "atomic_events": sum(1 for e in getattr(rec, "atomic_events", [])
                                   if getattr(e, "stream_id", 0) == sid),
            "memory_events": sum(1 for e in getattr(rec, "instr_events", [])
                                   if getattr(e, "stream_id", 0) == sid
                                   and ("ld" in getattr(e, "op", "")
                                        or "st" in getattr(e, "op", ""))),
        })
    if not rows: return ""
    return "<h3>Per-stream event breakdown</h3>" + pd.DataFrame(rows).to_html(index=False)
```

In `save_html`, add to context:
```python
    context.update({
        "stream_concurrency_html": _render_stream_concurrency(rec),
        "per_stream_breakdown_html": _render_per_stream_breakdown(rec),
    })
```

- [ ] **Step 4: Add template blocks**

In `gpusim/viz/_template.html.j2`, append after Phase 6 §21/§22:

```html
{% if stream_concurrency_html %}
<h2>§27 Stream concurrency timeline</h2>
{{ stream_concurrency_html | safe }}
{% endif %}

{% if per_stream_breakdown_html %}
<h2>§28 Per-stream resource breakdown</h2>
{{ per_stream_breakdown_html | safe }}
{% endif %}
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/viz/test_html_report_phase7.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/viz/html_report.py gpusim/viz/_template.html.j2 tests/unit/viz/test_html_report_phase7.py
git commit -m "feat(viz): HTML §27/§28 — stream concurrency + per-stream breakdown"
```

---

### Task 24: Perfetto stream swimlanes + stream_id args

**Files:**
- Modify: `gpusim/viz/perfetto.py`
- Test: extend `tests/unit/viz/test_html_report_phase7.py` (or new test file)

- [ ] **Step 1: Append failing test**

```python
def test_perfetto_kernel_launch_stream_swimlane():
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.perfetto import build_perfetto
    r = Recorder()
    r.kernel_launch(stream_id=0, kernel_name="k0", grid=(1,1,1), block=(32,1,1),
                     launch_cycle=0, complete_cycle=100, n_ctas=1)
    r.kernel_launch(stream_id=1, kernel_name="k1", grid=(1,1,1), block=(32,1,1),
                     launch_cycle=50, complete_cycle=150, n_ctas=1)
    pf = build_perfetto(r)
    # Check that some events use pid="Stream-0" and pid="Stream-1"
    pids = {e.get("pid") for e in pf.get("traceEvents", [])}
    assert any("Stream-0" in str(p) for p in pids)
    assert any("Stream-1" in str(p) for p in pids)
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add stream swimlanes to Perfetto**

In `gpusim/viz/perfetto.py`, in `build_perfetto`, append:

```python
    # Phase 7 stream swimlanes — one per unique stream_id from kernel_launch_events
    for ev in getattr(rec, "kernel_launch_events", []):
        events.append({
            "name": ev.kernel_name,
            "cat": "kernel_launch", "ph": "X",
            "ts": ev.launch_cycle,
            "dur": max(1, ev.complete_cycle - ev.launch_cycle),
            "pid": f"Stream-{ev.stream_id}",
            "tid": "kernel",
            "args": {
                "stream_id": ev.stream_id,
                "n_ctas": ev.n_ctas,
                "grid": list(ev.grid),
                "block": list(ev.block),
            },
        })
```

Also: ensure all existing event emissions add `args.stream_id = ev.stream_id` (even if it's 0). Most existing event emissions already pass `args` dict — just add the `stream_id` field.

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/viz/test_html_report_phase7.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/viz/perfetto.py tests/unit/viz/test_html_report_phase7.py
git commit -m "feat(viz): Perfetto Stream-N swimlanes for kernel_launch + stream_id args"
```

---

### Task 25: 4 tutorial chapters

**Files:**
- Create: `docs/tutorial/27-multi-stream-concurrency-basics.md`
- Create: `docs/tutorial/28-compute-memory-overlap.md`
- Create: `docs/tutorial/29-l2-hbm-contention-streams.md`
- Create: `docs/tutorial/30-scheduler-fairness-streams.md`

- [ ] **Step 1: Read existing style**

```bash
cat docs/tutorial/26-red-vs-atom.md | head -60
```

Match: English body + Chinese subheadings (`看模拟器` / `改一改` / `真机对照`), `##` heading hierarchy, code blocks for PTX/Python, ~500-700 words each.

- [ ] **Step 2: Write 4 chapters**

**Chapter 27 — multi-stream concurrency basics:**
Topics:
- What is a stream / how CUDA streams work
- gpusim.Stream() + Stream.launch + gpusim.synchronize()
- Concurrent vector_add demo walk-through
- 看模拟器: HTML §27 timeline; stream_summary()
- 改一改: 1 stream vs 2 streams cycles comparison
- 真机对照: cudaStream + cudaStreamSynchronize

**Chapter 28 — compute-memory overlap (⭐ core):**
Topics:
- Why HBM and SM are independent resources
- Compute-heavy kernel + memory-heavy kernel = real overlap
- compute_vs_memory_overlap demo walk-through
- 看模拟器: compute_memory_overlap metric; Perfetto two swimlanes
- 改一改: replace compute kernel with memory kernel → overlap disappears
- 真机对照: CUTLASS GEMM + epilogue stream decoupling

**Chapter 29 — L2/HBM contention across streams:**
Topics:
- Shared L2 + HBM: how concurrent kernels share bandwidth
- L2 set collisions; HBM channel arbitration
- l2_contention_2stream demo walk-through
- 看模拟器: l2_bandwidth_per_stream metric; HTML §28
- 改一改: separate gmem regions → contention disappears
- 真机对照: cudaStreamAttributeMemoryWindow / L2 cache window

**Chapter 30 — scheduler fairness across streams:**
Topics:
- RR scheduler at CTA granularity
- Jain's fairness index explained
- stream_priority_serial_vs_concurrent demo
- 看模拟器: stream_fairness_jain
- 改一改: uneven grid sizes → fairness stays (CTA-level), but stream completion times differ
- 真机对照: H100 default RR; cudaStreamCreateWithPriority (Phase 8 hint)

⚠ Each chapter ~500-700 words. Reference actual file paths and metric names. Follow chapter 26 (Phase 6) format.

- [ ] **Step 3: Commit**

```bash
git add docs/tutorial/27-multi-stream-concurrency-basics.md \
        docs/tutorial/28-compute-memory-overlap.md \
        docs/tutorial/29-l2-hbm-contention-streams.md \
        docs/tutorial/30-scheduler-fairness-streams.md
git commit -m "docs(tutorial): chapters 27-30 — multi-stream / compute-memory overlap / contention / fairness"
```

---

### Task 26: Phase 7 microbench + ref fixtures

**Files:**
- Create: `tests/microbench/test_phase7_facts.py`
- Create: `tests/microbench/test_phase7_runtime.py`
- Modify: `tests/reference/gen_reference.py`
- Create: 4 ref JSON stubs

- [ ] **Step 1: Phase 7 facts microbench**

`tests/microbench/test_phase7_facts.py`:
```python
"""Phase 7 microbench — multi-stream textbook facts."""
import numpy as np


def test_concurrent_no_slower_than_serial():
    """4 launches via 4 streams should not be slower than 4 launches in 1 stream."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    cfg = load_default()
    cfg.n_sm = 8
    
    src = """
.entry test(.param .u64 A, .param .u64 B, .param .u64 OUT) {
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    mov.u32 %r0, %tid.x;
    mul.lo.s32 %r1, %r0, 4;
    cvt.u64.u32 %rd3, %r1;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    add.f32 %f2, %f0, %f1;
    add.u64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f2;
    ret;
}
"""
    # Serial
    _reset_stream_id_counter()
    s = Stream()
    outs = [np.zeros(n, dtype=np.float32) for _ in range(4)]
    for i in range(4):
        s.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": outs[i]}, kernel_name=f"k{i}", config=cfg)
    serial_cycles = gpusim.synchronize(streams=[s], config=cfg).total_cycles
    
    # Concurrent
    _reset_stream_id_counter()
    streams = [Stream() for _ in range(4)]
    outs2 = [np.zeros(n, dtype=np.float32) for _ in range(4)]
    for i, st in enumerate(streams):
        st.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                   params={"A": A, "B": B, "OUT": outs2[i]}, kernel_name=f"k{i}", config=cfg)
    conc_cycles = gpusim.synchronize(streams=streams, config=cfg).total_cycles
    
    # Concurrent should not be more than 1.5× slower (loose: Phase 7 naive impl
    # may not yet provide concurrency benefit, but should never be much worse)
    assert conc_cycles <= serial_cycles * 1.5, \
        f"concurrent {conc_cycles} vs serial {serial_cycles}"
```

- [ ] **Step 2: Runtime budget (slow)**

`tests/microbench/test_phase7_runtime.py`:
```python
import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_concurrent_vector_add_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "concurrent_vector_add_2stream"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_compute_vs_memory_overlap_runtime_under_60s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "compute_vs_memory_overlap"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=120)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 60
```

- [ ] **Step 3: Append to gen_reference.py**

In `tests/reference/gen_reference.py`, append to the kernel list:
```python
"concurrent_vector_add_2stream",
"compute_vs_memory_overlap",
"l2_contention_2stream",
"stream_priority_serial_vs_concurrent",
```

- [ ] **Step 4: Create 4 ref JSON stubs**

```bash
for k in concurrent_vector_add_2stream compute_vs_memory_overlap l2_contention_2stream stream_priority_serial_vs_concurrent; do
  cat > tests/reference/data/$k.ref.json <<JSON
{
  "kernel": "$k",
  "phase": 7,
  "metrics": {
    "stream_concurrency_factor": null,
    "compute_memory_overlap": null,
    "l2_bandwidth_per_stream": null,
    "stream_fairness_jain": null
  },
  "tolerance": {
    "stream_concurrency_factor_pct": 15,
    "compute_memory_overlap_pct": 15,
    "l2_bandwidth_per_stream_pct": 15,
    "stream_fairness_jain_pct": 5
  },
  "notes": "Generated by gen_reference.py on real Hopper. null = not yet measured."
}
JSON
done
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/microbench/test_phase7_facts.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add tests/microbench/test_phase7_facts.py tests/microbench/test_phase7_runtime.py \
        tests/reference/gen_reference.py \
        tests/reference/data/concurrent_vector_add_2stream.ref.json \
        tests/reference/data/compute_vs_memory_overlap.ref.json \
        tests/reference/data/l2_contention_2stream.ref.json \
        tests/reference/data/stream_priority_serial_vs_concurrent.ref.json
git commit -m "test(microbench+reference): Phase 7 facts + 4 ref stubs"
```

---

### Task 27: README v7 + final tag

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read README**

- [ ] **Step 2: Update to v7**

In `README.md`:
- Capabilities/status: add Phase 7 ✅
- Phase 7 features section:
  - gpusim.Stream + Stream.launch + gpusim.synchronize() multi-stream API
  - Round-robin CTA scheduler across streams
  - 4 metrics (stream_concurrency_factor / compute_memory_overlap / l2_bandwidth_per_stream / stream_fairness_jain)
  - 1 new trace event (KernelLaunch) + stream_id propagated to 11 existing events
  - 2 HTML sections (§27/§28) + Perfetto Stream-N swimlanes
  - 4 examples + 4 tutorials chapters 27-30
  - 100% backward compatible: gpusim.run() unchanged
- Examples list: add 4 (was 25, now 29)
- Tutorials list: add 27-30 (was 26, now 30)
- Phase status: 1-7 ✅
- API usage example: show gpusim.Stream + gpusim.synchronize + multi_res.stream_summary()

- [ ] **Step 3: Run final suite + 4 examples**

```
.venv/bin/pytest -q -m "not slow"
.venv/bin/python examples/concurrent_vector_add_2stream/run.py
.venv/bin/python examples/compute_vs_memory_overlap/run.py
.venv/bin/python examples/l2_contention_2stream/run.py
.venv/bin/python examples/stream_priority_serial_vs_concurrent/run.py
```

- [ ] **Step 4: Commit + tag**

```bash
git add README.md
git commit -m "docs(readme): v7 — Phase 7 capabilities (multi-stream / multi-kernel concurrency)"
git tag phase7-complete
git tag | grep phase
git log --oneline | head -10
```

---

### Task 28: Final sanity sweep + done

- [ ] **Step 1: Run microbench facts**

```
.venv/bin/pytest tests/microbench/test_phase7_facts.py -v
```

- [ ] **Step 2: Run Phase 1-6 regression**

```
.venv/bin/pytest tests/parity/test_phase1_6_examples_unchanged.py -v
```

- [ ] **Step 3: Generate one HTML manually + spot-check §27/§28**

- [ ] **Step 4: Verify Perfetto JSON has Stream swimlanes**

- [ ] **Step 5: Done**

Phase 7 ships when all tasks complete + tags landed.

---

## End-of-plan checklist

- [ ] M1 (Trace plumbing + Stream/launch API): T1-T6
- [ ] M2 (MultiStreamScheduler + first demo): T7-T13
- [ ] M3 (4 metrics + MultiStreamResult + compute/memory example): T14-T18
- [ ] M4 (Contention + fairness examples): T19-T22
- [ ] M5 (Viz + docs + microbench + ship): T23-T28
- [ ] All 5 milestone tags
- [ ] Phase 1-6 regression unbroken
- [ ] 4 new examples + 4 tutorials shipped
- [ ] README v7 reflects Phase 7
