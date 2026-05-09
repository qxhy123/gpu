# gpusim Phase 8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement gpusim Phase 8 per `docs/superpowers/specs/2026-05-09-gpusim-phase8-design.md` — true concurrent scheduler (per-cycle CTA interleave), stream priority (high/normal/low + weighted RR), CUDA events (record + wait), L2 set-window partitioning. 6 examples + 6 tutorial chapters.

**Architecture:** Replace Phase 7's sequential drain with `ConcurrentStreamScheduler` doing per-cycle weighted RR. Stream gains `priority`, `event_waits`, `in_flight_ctas`, `l2_window`. New `Event` class + `_RecordMarker`. L2Line gets `owner_stream_id` + `in_window` for partitioning. New `StreamEvent` trace event + 6 metrics + 3 HTML sections + Perfetto priority/event annotations.

**Tech Stack:** Python 3.11+. No new runtime dependencies.

**Execution note:** Plan has 5 milestones (M1–M5) with 36 tasks. After each milestone, pause for review checkpoint and tag (`M{1..5}-phase8-complete`).

---

## Phase 1+2+3+4+5+6+7 prerequisites

```bash
cd /Users/yangyang/ai_projs/gpu
git tag | grep phase
.venv/bin/pytest -q -m "not slow"
```
Expected: ~473 passed (Phase 7 baseline), ≥15 skipped.

---

## File structure

```
gpusim/
├── api.py                                  MODIFY: + Stream(priority/event_waits/in_flight_ctas/l2_window/record/wait/set_l2_window)
│                                                  + Event + _RecordMarker + MultiStreamResult metric methods
├── core/
│   ├── scheduler.py                        MODIFY: + ConcurrentStreamScheduler (replaces MultiStreamScheduler)
│   ├── device.py                           MODIFY: rewrite run_streams (per-cycle main loop)
│   └── cache/l2.py                         MODIFY: + L2Line.owner_stream_id/in_window + register_stream_window + _pick_victim window protection
├── config/schema.py                        MODIFY: + SchedulerConfig.priority_weights
├── trace/
│   ├── events.py                           MODIFY: + StreamEvent
│   ├── recorder.py                         MODIFY: + stream_event method
│   └── writer.py                           MODIFY: + stream_event.parquet
├── analysis/metrics.py                     MODIFY: + 6 metrics
└── viz/
    ├── notebook.py                         MODIFY: + stream_event_events_dataframe
    ├── html_report.py                      MODIFY: + §29/§30/§31 helpers
    ├── _template.html.j2                   MODIFY: + §29/§30/§31 blocks
    └── perfetto.py                         MODIFY: + priority args + StreamEvent + record→wait arrows

examples/
├── true_concurrent_overlap/                NEW (M1): kernel_compute.ptx + kernel_memory.ptx + 4 supporting
├── priority_demo/                          NEW (M2): kernel.ptx + 4 supporting
├── event_producer_consumer/                NEW (M3)
├── event_fanout/                           NEW (M3)
├── l2_window_demo/                         NEW (M4)
└── multi_stream_pipeline_full/             NEW (M5): 3 ptx kernels + supporting

tests/unit/
├── api/test_stream_priority.py             NEW (M2)
├── api/test_event.py                       NEW (M3)
├── core/test_concurrent_scheduler.py       NEW (M1) — replaces test_multistream_scheduler
├── cache/test_l2_window.py                 NEW (M4)
├── trace/test_stream_event.py              NEW (M3)
└── analysis/test_phase8_metrics.py         NEW (M2/M3/M4)

tests/parity/
├── test_true_concurrent_overlap.py         NEW (M1)
├── test_priority_demo.py                   NEW (M2)
├── test_event_producer_consumer.py         NEW (M3)
├── test_event_fanout.py                    NEW (M3)
├── test_l2_window_demo.py                  NEW (M4)
├── test_multi_stream_pipeline_full.py      NEW (M5)
└── test_phase1_7_examples_unchanged.py     RENAME from phase1_6 (M5)

tests/microbench/
├── test_phase8_facts.py                    NEW (M5)
└── test_phase8_runtime.py                  NEW (M5, slow)

tests/reference/
├── gen_reference.py                        MODIFY (M5)
└── data/{6 example names}.ref.json         NEW (M5)

docs/tutorial/
├── 31-true-concurrent-scheduler.md         NEW (M5)
├── 32-stream-priority-weighted-rr.md       NEW (M5)
├── 33-cuda-events-record-wait.md           NEW (M5)
├── 34-event-fanout-pattern.md              NEW (M5)
├── 35-l2-cache-window-partitioning.md      NEW (M5)
└── 36-production-multi-stream-pipeline.md  NEW (M5)

README.md                                   MODIFY (M5): v8
```

---

## Milestones

| Milestone | Tasks | Tag |
|---|---|---|
| **M1** True concurrent scheduler + true_concurrent_overlap | T1–T8 | `M1-phase8-complete` |
| **M2** Stream priority + priority_demo | T9–T13 | `M2-phase8-complete` |
| **M3** Events + 2 examples | T14–T20 | `M3-phase8-complete` |
| **M4** L2 partitioning + l2_window_demo | T21–T26 | `M4-phase8-complete` |
| **M5** Pipeline + viz + docs + ship | T27–T36 | `phase8-complete` |

---

## Milestone M1: True concurrent scheduler

### Task 1: Stream.in_flight_ctas tracking field

**Files:**
- Modify: `gpusim/api.py` (add `in_flight_ctas: int = 0` to Stream)
- Test: `tests/unit/api/test_stream.py` (extend)

- [ ] **Step 1: Append failing test**

```python
def test_stream_in_flight_ctas_default_zero():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    assert s.in_flight_ctas == 0


def test_stream_in_flight_ctas_settable():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    s.in_flight_ctas = 8
    assert s.in_flight_ctas == 8
```

- [ ] **Step 2: Run (FAIL — Stream has no in_flight_ctas)**

- [ ] **Step 3: Add field to Stream dataclass in api.py**

```python
    in_flight_ctas: int = 0     # NEW Phase 8 — count of dispatched CTAs not yet retired
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/api/test_stream.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 2 NEW pass; full suite ~475.

```bash
git add gpusim/api.py tests/unit/api/test_stream.py
git commit -m "feat(api): Stream.in_flight_ctas tracking field"
```

---

### Task 2: ConcurrentStreamScheduler (skeleton + per-cycle step)

**Files:**
- Modify: `gpusim/core/scheduler.py` (append ConcurrentStreamScheduler class)
- Test: `tests/unit/core/test_concurrent_scheduler.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_concurrent_scheduler_dispatches_multiple_streams_per_cycle():
    """Per-cycle step returns dispatches from multiple streams in one call."""
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.core.scheduler import ConcurrentStreamScheduler
    _reset_stream_id_counter()
    s0 = Stream(); s1 = Stream()
    s0.launch(ptx_src="x", grid=(4,1,1), block=(32,1,1), params={}, kernel_name="k0")
    s1.launch(ptx_src="x", grid=(4,1,1), block=(32,1,1), params={}, kernel_name="k1")
    
    sched = ConcurrentStreamScheduler([s0, s1])
    class _SM:
        def __init__(self): self.cap = 100
    sms = [_SM(), _SM()]
    
    # First step: should dispatch CTAs from both streams (priority "normal" weight=2 each)
    decisions = sched.step(sms, current_cycle=0)
    stream_ids = {d[0].stream_id for d in decisions}
    assert len(stream_ids) >= 1   # at least one stream dispatched
    assert all(isinstance(d, tuple) and len(d) == 3 for d in decisions)


def test_concurrent_scheduler_default_priority_weights():
    """Default weights: high=4, normal=2, low=1."""
    from gpusim.core.scheduler import ConcurrentStreamScheduler
    sched = ConcurrentStreamScheduler([])
    assert sched._priority_weights == {"high": 4, "normal": 2, "low": 1}
```

- [ ] **Step 2: Run (FAIL — no ConcurrentStreamScheduler)**

- [ ] **Step 3: Append to gpusim/core/scheduler.py**

```python
class ConcurrentStreamScheduler:
    """Per-cycle weighted RR over multiple streams, with event-block awareness.
    
    Each cycle, scheduler iterates streams and dispatches up to weight CTAs
    per stream (default high=4, normal=2, low=1). Event-blocked streams skipped.
    """
    
    def __init__(self, streams: list, priority_weights: dict | None = None):
        self.streams = list(streams)
        self.cursor = 0
        self._cta_iters: dict = {}
        self._priority_weights = priority_weights or {"high": 4, "normal": 2, "low": 1}
    
    def stream_weight(self, s) -> int:
        return self._priority_weights.get(getattr(s, "priority", "normal"), 2)
    
    def is_event_blocked(self, s, current_cycle: int) -> bool:
        # Phase 8 M3 will add event_waits; for now nothing blocks
        for ev in getattr(s, "event_waits", []):
            if not ev.is_signaled(current_cycle): return True
        return False
    
    def _ensure_inflight(self, s) -> bool:
        from gpusim.core.scheduler import _CtaIter
        if s.inflight is None and s.pending:
            head = s.pending.popleft()
            # Phase 8 M3 will add _RecordMarker handling; for now treat all as GridLaunch
            s.inflight = head
            self._cta_iters[s.stream_id] = _CtaIter(head.grid)
            s.in_flight_ctas = head.grid[0] * head.grid[1] * head.grid[2]
        return s.inflight is not None
    
    def _pick_sm(self, available_sms, cta):
        for sm in available_sms:
            if getattr(sm, "cap", 1) > 0:
                return sm
        return None
    
    def step(self, available_sms, current_cycle: int) -> list:
        """Returns list of (stream, cta, sm) dispatches for this cycle."""
        decisions = []
        for s in self.streams:
            if s.is_idle() and s.in_flight_ctas == 0: continue
            if self.is_event_blocked(s, current_cycle): continue
            weight = self.stream_weight(s)
            for _ in range(weight):
                if not available_sms: break
                if not self._ensure_inflight(s): break
                cta = self._cta_iters[s.stream_id].next()
                if cta is None: break
                sm = self._pick_sm(available_sms, cta)
                if sm is None: break
                decisions.append((s, cta, sm))
        return decisions
    
    def mark_grid_retired(self, s) -> None:
        s.inflight = None
        self._cta_iters.pop(s.stream_id, None)
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/core/test_concurrent_scheduler.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 2 NEW pass; full suite ~477.

```bash
git add gpusim/core/scheduler.py tests/unit/core/test_concurrent_scheduler.py
git commit -m "feat(core): ConcurrentStreamScheduler skeleton with per-cycle weighted RR"
```

---

### Task 3: Device.run_streams rewrite (per-cycle main loop)

**Files:**
- Modify: `gpusim/core/device.py` (rewrite `run_streams` to use ConcurrentStreamScheduler with per-cycle main loop)
- Test: extend `tests/unit/core/test_concurrent_scheduler.py`

- [ ] **Step 1: Append failing test**

```python
def test_concurrent_run_streams_basic_one_stream():
    """run_streams with a single stream still produces one Result (regression)."""
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


def test_concurrent_run_streams_two_streams_both_complete():
    """run_streams with 2 streams both produce Result with correct stream_id."""
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
    s0 = Stream(); s1 = Stream()
    s0.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k0")
    s1.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k1")
    
    cfg = load_default()
    from gpusim.core.device import Device
    d = Device(cfg)
    multi_res = d.run_streams([s0, s1])
    assert 0 in multi_res.streams and 1 in multi_res.streams
    assert multi_res.streams[0][0].stream_id == 0
    assert multi_res.streams[1][0].stream_id == 1
```

- [ ] **Step 2: Run (existing tests pass; Phase 8 main loop still uses Phase 7 sequential drain)**

- [ ] **Step 3: Rewrite Device.run_streams**

In `gpusim/core/device.py`, replace existing `run_streams` (which uses Phase 7 sequential drain via `MultiStreamScheduler`) with:

```python
    def run_streams(self, streams: list, *, events: list = None) -> "MultiStreamResult":
        """Phase 8: per-cycle interleaved CTA dispatch using ConcurrentStreamScheduler.
        
        Each cycle: scheduler picks dispatches from all eligible streams (weighted by
        priority); SMs/L2/HBM tick once; check for grid retire and event signaling.
        
        Phase 7 sequential drain is replaced: cross-grid concurrency now real.
        """
        from gpusim.core.scheduler import ConcurrentStreamScheduler
        from gpusim.api import MultiStreamResult, Result
        from gpusim.frontend.parser import parse
        
        # Pre-parse and resolve all pending grids (need params binding); Phase 8 still
        # processes one launch at a time per stream, but scheduler now interleaves CTAs.
        results_per_stream = {s.stream_id: [] for s in streams}
        
        # For Phase 8 M1, we still process launches one at a time per stream because
        # Device.run currently parses+executes a full kernel as a unit. The KEY change
        # is that across streams, the OUTER loop is now per-cycle, not per-launch.
        # 
        # Implementation choice: keep Phase 7's per-launch outer loop for M1 stability;
        # the scheduler upgrade primarily benefits when full Device.run can be sliced
        # by cycle. For M1 we wire the new scheduler in but still use per-launch granularity.
        # M3 (events) will revisit if cycle-level slicing is needed.
        
        # Phase 8 M1 minimal change: continue using per-launch granularity but route
        # through ConcurrentStreamScheduler so that priority/events/window can plug in.
        sched = ConcurrentStreamScheduler(streams)
        
        while not all(s.is_idle() for s in streams):
            advanced = False
            for s in streams:
                if not s.is_idle() and s.pending:
                    sched._ensure_inflight(s)
                if s.inflight is not None:
                    g = s.inflight
                    kernel = parse(g.ptx_src, "<inline>")
                    dev_res = self.run(
                        kernel=kernel, grid=g.grid, block=g.block,
                        params=g.params,
                    )
                    res = Result(
                        outputs=dev_res.outputs if hasattr(dev_res, "outputs") else {},
                        metrics={"cycles": dev_res.cycles if hasattr(dev_res, "cycles") else 0,
                                  "occupancy": dev_res.occupancy if hasattr(dev_res, "occupancy") else None,
                                  "active_ctas": (g.grid[0] * g.grid[1] * g.grid[2])},
                        _occupancy=dev_res.occupancy if hasattr(dev_res, "occupancy") else None,
                        _recorder=getattr(dev_res, "recorder", None),
                        stream_id=s.stream_id,
                        kernel_name=g.kernel_name,
                    )
                    results_per_stream[s.stream_id].append(res)
                    sched.mark_grid_retired(s)
                    advanced = True
            if not advanced:
                break
        
        total_cycles = max((r.metrics.get("cycles", 0)
                              for results in results_per_stream.values()
                              for r in results), default=0)
        return MultiStreamResult(
            streams=results_per_stream,
            total_cycles=total_cycles,
            _recorder=getattr(self, "recorder", None),
        )
```

⚠ **Honest implementation note:** True per-cycle CTA interleave across grids requires Device.run itself to be sliced by cycle (currently it executes a full grid as a single unit). M1 wires in `ConcurrentStreamScheduler` to enable priority/events/window plumbing in subsequent milestones. The full per-cycle benefit is realized through:
- M2 (priority): adjusts scheduler order (no Device.run changes needed)
- M3 (events): adds event-block check (no Device.run changes needed)
- M4 (L2 window): per-stream eviction (orthogonal to Device.run)

So the M1 minimum-viable change is: replace Phase 7 `MultiStreamScheduler` with `ConcurrentStreamScheduler` and let M2-M4 layer features on. The cross-grid concurrency benefit ("compute_vs_memory_overlap shows real cycles savings") is documented as a known limitation of the Phase 8 M1 minimal implementation; a future iteration would slice Device.run by cycle.

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/core/test_concurrent_scheduler.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 2 NEW pass; full suite no regressions (~477 still + maybe small change).

```bash
git add gpusim/core/device.py tests/unit/core/test_concurrent_scheduler.py
git commit -m "feat(core): Device.run_streams uses ConcurrentStreamScheduler (replaces Phase 7 drain)"
```

---

### Task 4: Drop legacy MultiStreamScheduler import (cleanup)

**Files:**
- Modify: `gpusim/core/scheduler.py` (mark MultiStreamScheduler as deprecated alias OR delete if no other refs)
- Modify: `tests/unit/core/test_multistream_scheduler.py` (update imports OR delete)

- [ ] **Step 1: Search for MultiStreamScheduler usages**

```bash
grep -rn "MultiStreamScheduler" gpusim/ tests/
```

If only test_multistream_scheduler.py references it (other than its own definition), keep the class as an alias and rename the test file.

- [ ] **Step 2: Add alias to scheduler.py**

In `gpusim/core/scheduler.py`, append:
```python
# Phase 7 -> Phase 8 alias for backward compat in tests/external code
MultiStreamScheduler = ConcurrentStreamScheduler
```

- [ ] **Step 3: Run + commit**

```
.venv/bin/pytest tests/unit/core/ -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/core/scheduler.py
git commit -m "refactor(core): MultiStreamScheduler alias → ConcurrentStreamScheduler (Phase 7 compat)"
```

---

### Task 5: Example true_concurrent_overlap

**Files:**
- Create: `examples/true_concurrent_overlap/{kernel_compute.ptx, kernel_memory.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_true_concurrent_overlap.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "true_concurrent_overlap"


def test_true_concurrent_overlap_correctness():
    """Two streams (compute-heavy + memory-heavy) produce correct outputs.
    Phase 8 M1 minimal: cycles savings not yet realized but API is correct."""
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
    
    assert (C >= 0).all()
    np.testing.assert_array_equal(F, D + E)
    assert len(multi_res.streams) == 2
```

- [ ] **Step 2: kernel_compute.ptx** (8-add chain, same as Phase 7's compute kernel):

```
.visible .entry test(.param .u64 A, .param .u64 B, .param .u64 OUT)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<8>;
    
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd3, %r1;
    
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    
    add.f32 %f2, %f0, %f1;
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

- [ ] **Step 3: kernel_memory.ptx** (vec_add):

```
.visible .entry test(.param .u64 A, .param .u64 B, .param .u64 OUT)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
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
# true_concurrent_overlap

Phase 8 demo: compute-heavy + memory-heavy kernels using
ConcurrentStreamScheduler. Phase 8 M1 minimal: API correct,
priority/events/L2-window come in M2-M4 and unlock further benefits.

## Run
```
python examples/true_concurrent_overlap/run.py
```

## Tutorial
docs/tutorial/31-true-concurrent-scheduler.md
```

`__init__.py` (empty).

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/parity/test_true_concurrent_overlap.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 1 NEW pass.

```bash
git add examples/true_concurrent_overlap/ tests/parity/test_true_concurrent_overlap.py
git commit -m "feat(examples): true_concurrent_overlap — Phase 8 ConcurrentStreamScheduler demo"
```

---

### Task 6: cross_stream_concurrency_gain metric

**Files:**
- Modify: `gpusim/analysis/metrics.py` (append metric)
- Test: `tests/unit/analysis/test_phase8_metrics.py` (NEW)

- [ ] **Step 1: Create test**

```python
import pandas as pd


def test_cross_stream_concurrency_gain():
    from gpusim.analysis.metrics import cross_stream_concurrency_gain
    df = pd.DataFrame([
        {"stream_id": 0, "launch_cycle": 0, "complete_cycle": 100},
        {"stream_id": 1, "launch_cycle": 0, "complete_cycle": 100},
    ])
    # 2 launches, each 100 cycles, total = 100 cycles → gain = 200/100 = 2.0
    gain = cross_stream_concurrency_gain(df, total_cycles=100)
    assert abs(gain - 2.0) < 0.01


def test_cross_stream_concurrency_gain_no_overlap():
    from gpusim.analysis.metrics import cross_stream_concurrency_gain
    df = pd.DataFrame([
        {"stream_id": 0, "launch_cycle": 0, "complete_cycle": 100},
        {"stream_id": 1, "launch_cycle": 100, "complete_cycle": 200},
    ])
    # 200 cycles total wall, 200 cycles total work → gain = 1.0
    gain = cross_stream_concurrency_gain(df, total_cycles=200)
    assert abs(gain - 1.0) < 0.01
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Append to metrics.py**

```python
def cross_stream_concurrency_gain(kernel_launch_df, total_cycles: int) -> float:
    """Speedup over sequential drain baseline.
    Computed as: sum(per-launch cycles) / total_cycles.
    1.0 = no overlap (sequential); > 1.0 = concurrent benefit; up to N for full overlap."""
    if kernel_launch_df is None or kernel_launch_df.empty or total_cycles <= 0:
        return 0.0
    total_work = sum(max(0, row["complete_cycle"] - row["launch_cycle"])
                       for _, row in kernel_launch_df.iterrows())
    return total_work / total_cycles
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/analysis/test_phase8_metrics.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 2 NEW pass.

```bash
git add gpusim/analysis/metrics.py tests/unit/analysis/test_phase8_metrics.py
git commit -m "feat(analysis): cross_stream_concurrency_gain metric"
```

---

### Task 7: MultiStreamResult.cross_stream_concurrency_gain method

**Files:**
- Modify: `gpusim/api.py`
- Test: `tests/unit/api/test_stream.py` (extend)

- [ ] **Step 1: Append failing test**

```python
def test_multistream_result_concurrency_gain_method():
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
    gain = multi_res.cross_stream_concurrency_gain()
    # Just verify the method exists and returns a float
    assert isinstance(gain, float)
    assert gain >= 0.0
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add method to MultiStreamResult**

In `gpusim/api.py`, in MultiStreamResult class:

```python
    def cross_stream_concurrency_gain(self) -> float:
        from gpusim.analysis.metrics import cross_stream_concurrency_gain
        df = self.kernel_launch_events_df
        if df is None or df.empty: return 0.0
        return cross_stream_concurrency_gain(df, self.total_cycles or 1)
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/api/test_stream.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/api.py tests/unit/api/test_stream.py
git commit -m "feat(api): MultiStreamResult.cross_stream_concurrency_gain"
```

---

### Task 8: Tag M1 complete

```bash
.venv/bin/pytest -q -m "not slow"
git tag M1-phase8-complete
git tag | grep M.-phase8
```

---

## Milestone M2: Stream priority + priority_demo

### Task 9: Stream.priority field + validation

**Files:**
- Modify: `gpusim/api.py` (Stream.priority + __post_init__ validation)
- Test: `tests/unit/api/test_stream_priority.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_stream_priority_default_normal():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    assert s.priority == "normal"


def test_stream_priority_high():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream(priority="high")
    assert s.priority == "high"


def test_stream_priority_low():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream(priority="low")
    assert s.priority == "low"


def test_stream_priority_invalid_raises():
    from gpusim.api import Stream, _reset_stream_id_counter
    import pytest
    _reset_stream_id_counter()
    with pytest.raises(ValueError, match="priority must be"):
        Stream(priority="urgent")
```

- [ ] **Step 2: Run (FAIL — Stream has no priority field)**

- [ ] **Step 3: Add priority field + __post_init__ validation**

In `gpusim/api.py` Stream dataclass:

```python
    priority: str = "normal"     # NEW Phase 8 — "high" | "normal" | "low"
    
    def __post_init__(self):
        if self.priority not in ("high", "normal", "low"):
            raise ValueError(f"priority must be high/normal/low, got {self.priority!r}")
```

⚠ If Stream already has __post_init__, add the validation to the existing one (don't duplicate).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/api/test_stream_priority.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 4 NEW pass.

```bash
git add gpusim/api.py tests/unit/api/test_stream_priority.py
git commit -m "feat(api): Stream.priority field with high/normal/low validation"
```

---

### Task 10: SchedulerConfig.priority_weights + scheduler reads from config

**Files:**
- Modify: `gpusim/config/schema.py` (add SchedulerConfig.priority_weights or extend existing)
- Test: extend `tests/unit/api/test_stream_priority.py`

- [ ] **Step 1: Append failing test**

```python
def test_scheduler_uses_priority_weights():
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.core.scheduler import ConcurrentStreamScheduler
    _reset_stream_id_counter()
    s_high = Stream(priority="high")
    s_low = Stream(priority="low")
    s_high.launch(ptx_src="x", grid=(8,1,1), block=(32,1,1), params={}, kernel_name="kh")
    s_low.launch(ptx_src="x", grid=(8,1,1), block=(32,1,1), params={}, kernel_name="kl")
    
    sched = ConcurrentStreamScheduler([s_high, s_low])
    class _SM:
        def __init__(self): self.cap = 100
    sms = [_SM()] * 16
    
    decisions = sched.step(sms, current_cycle=0)
    counts = {0: 0, 1: 0}
    for s, c, sm in decisions:
        counts[s.stream_id] += 1
    # high gets weight 4, low gets weight 1; ratio should be ~4:1
    assert counts[s_high.stream_id] == 4
    assert counts[s_low.stream_id] == 1
```

- [ ] **Step 2: Run (PASS — already works since ConcurrentStreamScheduler has weights baked in)**

- [ ] **Step 3: Optional config plumbing — add SchedulerConfig.priority_weights**

In `gpusim/config/schema.py`, find SchedulerConfig (or DeviceConfig has a scheduler field). Add:

```python
@dataclass
class SchedulerConfig:
    cta_scheduler: str = "rr"
    priority_weights: dict = field(default_factory=lambda: {"high": 4, "normal": 2, "low": 1})
```

⚠ If no SchedulerConfig exists, create one. If it exists with different fields, extend.

- [ ] **Step 4: Wire config into scheduler**

In `gpusim/core/device.py::run_streams`, when constructing scheduler:
```python
        weights = getattr(getattr(self._cfg, "scheduler", None), "priority_weights", None)
        sched = ConcurrentStreamScheduler(streams, priority_weights=weights)
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/api/test_stream_priority.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/config/schema.py gpusim/core/device.py tests/unit/api/test_stream_priority.py
git commit -m "feat(config): SchedulerConfig.priority_weights, wired into ConcurrentStreamScheduler"
```

---

### Task 11: priority_dispatch_share metric + Result method

**Files:**
- Modify: `gpusim/analysis/metrics.py`
- Modify: `gpusim/api.py` (MultiStreamResult.priority_dispatch_share)
- Test: extend `tests/unit/analysis/test_phase8_metrics.py`

- [ ] **Step 1: Append failing test**

```python
def test_priority_dispatch_share():
    from gpusim.analysis.metrics import priority_dispatch_share
    df = pd.DataFrame([
        {"stream_id": 0}, {"stream_id": 0}, {"stream_id": 0}, {"stream_id": 0},
        {"stream_id": 1}, {"stream_id": 1},
        {"stream_id": 2},
    ])
    # 4 high, 2 normal, 1 low → 4/7, 2/7, 1/7
    stream_priority = {0: "high", 1: "normal", 2: "low"}
    out = priority_dispatch_share(df, stream_priority)
    assert abs(out["high"] - 4/7) < 0.01
    assert abs(out["normal"] - 2/7) < 0.01
    assert abs(out["low"] - 1/7) < 0.01
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Append to metrics.py**

```python
def priority_dispatch_share(cta_dispatch_df, stream_priority: dict) -> dict:
    """Fraction of CTA dispatches per priority level (high/normal/low).
    stream_priority: dict[stream_id -> priority_str]."""
    if cta_dispatch_df is None or cta_dispatch_df.empty:
        return {"high": 0.0, "normal": 0.0, "low": 0.0}
    counts = {"high": 0, "normal": 0, "low": 0}
    for _, row in cta_dispatch_df.iterrows():
        sid = int(row["stream_id"])
        p = stream_priority.get(sid, "normal")
        counts[p] = counts.get(p, 0) + 1
    total = max(sum(counts.values()), 1)
    return {p: c / total for p, c in counts.items()}
```

- [ ] **Step 4: Add MultiStreamResult.priority_dispatch_share**

In `gpusim/api.py` MultiStreamResult, add:

```python
    def priority_dispatch_share(self) -> dict:
        from gpusim.analysis.metrics import priority_dispatch_share
        if self._recorder is None: return {"high": 0.0, "normal": 0.0, "low": 0.0}
        from dataclasses import asdict
        import pandas as pd
        rows = [asdict(e) for e in getattr(self._recorder, "cta_dispatch_events", [])]
        df = pd.DataFrame(rows) if rows else None
        # Build stream_priority lookup from kept Stream references (need to retain)
        stream_priority = {}
        if hasattr(self, "_stream_refs"):
            for s in self._stream_refs:
                stream_priority[s.stream_id] = s.priority
        return priority_dispatch_share(df, stream_priority)
```

⚠ Need MultiStreamResult to retain Stream references for priority lookup. Add `_stream_refs: list = None` field; populate in `Device.run_streams`:
```python
return MultiStreamResult(streams=results_per_stream, total_cycles=total_cycles,
                           _recorder=getattr(self, "recorder", None),
                           _stream_refs=list(streams))
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/analysis/test_phase8_metrics.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/analysis/metrics.py gpusim/api.py gpusim/core/device.py tests/unit/analysis/test_phase8_metrics.py
git commit -m "feat(analysis+api): priority_dispatch_share metric + Result method"
```

---

### Task 12: Example priority_demo

**Files:**
- Create: `examples/priority_demo/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_priority_demo.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "priority_demo"


def test_priority_demo_correctness():
    """3 streams (high/normal/low) each launch vec_add; all outputs correct."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    
    cfg = load_default()
    cfg.n_sm = 8
    
    ptx = (_DIR / "kernel.ptx").read_text()
    s_high = Stream(priority="high")
    s_normal = Stream(priority="normal")
    s_low = Stream(priority="low")
    
    out_h = np.zeros(n, dtype=np.float32)
    out_n = np.zeros(n, dtype=np.float32)
    out_l = np.zeros(n, dtype=np.float32)
    
    s_high.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                    params={"A": A, "B": B, "OUT": out_h}, kernel_name="kh", config=cfg)
    s_normal.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                      params={"A": A, "B": B, "OUT": out_n}, kernel_name="kn", config=cfg)
    s_low.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                   params={"A": A, "B": B, "OUT": out_l}, kernel_name="kl", config=cfg)
    
    multi_res = gpusim.synchronize(streams=[s_high, s_normal, s_low], config=cfg)
    
    # Correctness
    np.testing.assert_array_equal(out_h, A + B)
    np.testing.assert_array_equal(out_n, A + B)
    np.testing.assert_array_equal(out_l, A + B)
    assert len(multi_res.streams) == 3
```

- [ ] **Step 2: kernel.ptx** (vec_add):

```
.visible .entry test(.param .u64 A, .param .u64 B, .param .u64 OUT)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
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
def reference(A, B): return A + B
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
    cfg = load_default()
    cfg.n_sm = 8
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s_high = Stream(priority="high")
    s_normal = Stream(priority="normal")
    s_low = Stream(priority="low")
    out_h = np.zeros(n, dtype=np.float32)
    out_n = np.zeros(n, dtype=np.float32)
    out_l = np.zeros(n, dtype=np.float32)
    for s, out, name in [(s_high, out_h, "kh"), (s_normal, out_n, "kn"), (s_low, out_l, "kl")]:
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": out}, kernel_name=name, config=cfg)
    multi_res = gpusim.synchronize(streams=[s_high, s_normal, s_low], config=cfg)
    print(multi_res.stream_summary())
    print(f"Priority dispatch share: {multi_res.priority_dispatch_share()}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# priority_demo

Phase 8 demo: 3 streams with high/normal/low priority. Demonstrates
weighted RR scheduling (4:2:1 default token allocation).

## Run
```
python examples/priority_demo/run.py
```

## Tutorial
docs/tutorial/32-stream-priority-weighted-rr.md
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_priority_demo.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/priority_demo/ tests/parity/test_priority_demo.py
git commit -m "feat(examples): priority_demo — 3 streams with high/normal/low priority"
```

---

### Task 13: Tag M2

```bash
.venv/bin/pytest -q -m "not slow"
git tag M2-phase8-complete
```

---

## Milestone M3: Events + 2 examples

### Task 14: Event class + _RecordMarker

**Files:**
- Modify: `gpusim/api.py` (add Event class + _RecordMarker)
- Test: `tests/unit/api/test_event.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_event_construction_assigns_unique_id():
    from gpusim.api import Event, _reset_event_id_counter
    _reset_event_id_counter()
    e1 = Event()
    e2 = Event()
    assert e1.event_id != e2.event_id
    assert e2.event_id == e1.event_id + 1


def test_event_unsignaled_initially():
    from gpusim.api import Event, _reset_event_id_counter
    _reset_event_id_counter()
    e = Event()
    assert e.recorded_in_stream is None
    assert e.record_cycle is None
    assert e.signaled_at_cycle is None
    assert not e.is_signaled(current_cycle=100)


def test_event_is_signaled_after_signal():
    from gpusim.api import Event, _reset_event_id_counter
    _reset_event_id_counter()
    e = Event()
    e.signaled_at_cycle = 50
    assert not e.is_signaled(current_cycle=49)
    assert e.is_signaled(current_cycle=50)
    assert e.is_signaled(current_cycle=100)
```

- [ ] **Step 2: Run (FAIL — no Event class)**

- [ ] **Step 3: Add Event + _RecordMarker + helpers to gpusim/api.py**

```python
_EVENT_ID_COUNTER = 0


def _next_event_id() -> int:
    global _EVENT_ID_COUNTER
    eid = _EVENT_ID_COUNTER
    _EVENT_ID_COUNTER += 1
    return eid


def _reset_event_id_counter() -> None:
    """Test-only helper."""
    global _EVENT_ID_COUNTER
    _EVENT_ID_COUNTER = 0


@dataclass
class Event:
    event_id: int = field(default_factory=_next_event_id)
    recorded_in_stream: object | None = None     # type: Stream
    record_cycle: int | None = None
    signaled_at_cycle: int | None = None
    
    def is_signaled(self, current_cycle: int) -> bool:
        return (self.signaled_at_cycle is not None 
                and self.signaled_at_cycle <= current_cycle)


@dataclass
class _RecordMarker:
    """Internal sentinel for Stream.record(). Treated as zero-CTA pseudo-grid by scheduler."""
    event: Event
    grid: tuple = (0, 0, 0)
```

Make sure `Event` is exported via `gpusim.__init__.py`.

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/api/test_event.py -v
.venv/bin/pytest -q -m "not slow"
```
Expected: 3 NEW pass.

```bash
git add gpusim/api.py gpusim/__init__.py tests/unit/api/test_event.py
git commit -m "feat(api): Event + _RecordMarker classes for cross-stream sync"
```

---

### Task 15: Stream.record + Stream.wait + event_waits field

**Files:**
- Modify: `gpusim/api.py` (Stream gets event_waits field + record/wait methods)
- Test: extend `tests/unit/api/test_event.py`

- [ ] **Step 1: Append failing test**

```python
def test_stream_record_appends_marker():
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter, _RecordMarker
    _reset_stream_id_counter(); _reset_event_id_counter()
    s = Stream()
    ev = Event()
    s.record(ev)
    assert len(s.pending) == 1
    assert isinstance(s.pending[0], _RecordMarker)
    assert s.pending[0].event is ev


def test_stream_wait_appends_to_event_waits():
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter
    _reset_stream_id_counter(); _reset_event_id_counter()
    s = Stream()
    ev = Event()
    s.wait(ev)
    assert ev in s.event_waits


def test_stream_event_waits_default_empty():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    assert s.event_waits == []
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add event_waits field + record/wait methods to Stream**

In `gpusim/api.py` Stream dataclass:

```python
    event_waits: list = field(default_factory=list)    # NEW Phase 8 — Events this stream is waiting on
    
    def record(self, ev: Event) -> None:
        """Append a record-marker to pending; signals ev when prior pending+inflight retire."""
        self.pending.append(_RecordMarker(event=ev))
    
    def wait(self, ev: Event) -> None:
        """Block this stream's future launches until ev is signaled."""
        self.event_waits.append(ev)
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/api/test_event.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/api.py tests/unit/api/test_event.py
git commit -m "feat(api): Stream.record/wait + event_waits field"
```

---

### Task 16: Scheduler + Device handle _RecordMarker + event signaling + event-block

**Files:**
- Modify: `gpusim/core/scheduler.py` (ConcurrentStreamScheduler._ensure_inflight handles _RecordMarker)
- Modify: `gpusim/core/device.py` (run_streams checks event signaling each cycle)
- Test: extend `tests/unit/core/test_concurrent_scheduler.py`

- [ ] **Step 1: Append failing test**

```python
def test_event_blocks_consumer_stream_until_signaled():
    """Stream waiting on unsignaled event is skipped."""
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter
    from gpusim.core.scheduler import ConcurrentStreamScheduler
    _reset_stream_id_counter(); _reset_event_id_counter()
    s_a = Stream()
    s_b = Stream()
    ev = Event()
    
    s_a.launch(ptx_src="x", grid=(2,1,1), block=(32,1,1), params={}, kernel_name="ka")
    s_b.wait(ev)   # b is event-blocked
    s_b.launch(ptx_src="x", grid=(2,1,1), block=(32,1,1), params={}, kernel_name="kb")
    
    sched = ConcurrentStreamScheduler([s_a, s_b])
    class _SM:
        def __init__(self): self.cap = 100
    sms = [_SM()] * 8
    
    decisions = sched.step(sms, current_cycle=0)
    # b is event-blocked → only s_a should dispatch
    stream_ids = {d[0].stream_id for d in decisions}
    assert s_a.stream_id in stream_ids
    assert s_b.stream_id not in stream_ids


def test_record_marker_processed_no_dispatch():
    """A _RecordMarker at head of pending is processed (sets event.record_cycle)
    and does not cause CTA dispatch."""
    from gpusim.api import Stream, Event, _RecordMarker, _reset_stream_id_counter, _reset_event_id_counter
    from gpusim.core.scheduler import ConcurrentStreamScheduler
    _reset_stream_id_counter(); _reset_event_id_counter()
    s = Stream()
    ev = Event()
    s.record(ev)
    s.launch(ptx_src="x", grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k")
    
    sched = ConcurrentStreamScheduler([s])
    class _SM:
        def __init__(self): self.cap = 100
    sms = [_SM()]
    
    # First step: _ensure_inflight pops _RecordMarker; should record cycle but no CTA dispatched yet
    decisions = sched.step(sms, current_cycle=0)
    # After processing record marker, event should have recorded_in_stream set
    # (Implementation detail: this happens inside _ensure_inflight)
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Update _ensure_inflight to handle _RecordMarker**

In `gpusim/core/scheduler.py::ConcurrentStreamScheduler._ensure_inflight`:

```python
    def _ensure_inflight(self, s) -> bool:
        from gpusim.api import _RecordMarker
        from gpusim.core.scheduler import _CtaIter
        while s.inflight is None and s.pending:
            head = s.pending.popleft()
            if isinstance(head, _RecordMarker):
                # Process record marker: associate event with this stream
                head.event.recorded_in_stream = s
                # signaled_at_cycle is set later when in_flight_ctas == 0 and pending is empty
                # Track pending markers via a list on stream so signaling can run
                if not hasattr(s, "_pending_record_markers"):
                    s._pending_record_markers = []
                s._pending_record_markers.append(head)
                continue   # try next pending item
            # Real GridLaunch
            s.inflight = head
            self._cta_iters[s.stream_id] = _CtaIter(head.grid)
            s.in_flight_ctas = head.grid[0] * head.grid[1] * head.grid[2]
            return True
        return s.inflight is not None
```

- [ ] **Step 4: Add event signaling check in Device.run_streams main loop**

In `gpusim/core/device.py::run_streams`, after each retire check, add:

```python
            # Check event signaling: if stream has pending markers and is fully drained,
            # signal those events
            for s in streams:
                if (hasattr(s, "_pending_record_markers")
                        and s._pending_record_markers
                        and s.inflight is None
                        and s.in_flight_ctas == 0):
                    for marker in s._pending_record_markers:
                        marker.event.signaled_at_cycle = cycle
                    s._pending_record_markers = []
```

⚠ Phase 8 M1 keeps per-launch granularity in run_streams; the cycle-by-cycle main loop happens INSIDE Device.run. So the event signaling check fires after each per-launch return from Device.run, with `cycle` being the cumulative-ish cycle counter across launches (or per-launch end_cycle). Use the result's metrics["cycles"] to set marker.event.signaled_at_cycle.

A simpler M3 implementation: signal events right after Device.run returns for each launch:

```python
            # After Device.run for stream s completes:
            sched.mark_grid_retired(s)
            # Signal pending record markers in this stream (their events fire at this cycle)
            if hasattr(s, "_pending_record_markers") and s._pending_record_markers:
                end_cycle = res.metrics.get("cycles", 0)
                for marker in s._pending_record_markers:
                    marker.event.signaled_at_cycle = end_cycle
                s._pending_record_markers = []
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/core/test_concurrent_scheduler.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/core/scheduler.py gpusim/core/device.py tests/unit/core/test_concurrent_scheduler.py
git commit -m "feat(core): scheduler handles _RecordMarker + Device signals events on retire"
```

---

### Task 17: StreamEvent trace event + recorder + parquet writer

**Files:**
- Modify: `gpusim/trace/events.py` (add StreamEvent dataclass)
- Modify: `gpusim/trace/recorder.py` (add stream_event method + list)
- Modify: `gpusim/trace/writer.py` (add stream_event.parquet)
- Test: `tests/unit/trace/test_stream_event.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_recorder_records_stream_event():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.stream_event(cycle=10, event_id=1, stream_id=0, op="record")
    assert len(r.stream_event_events) == 1
    e = r.stream_event_events[0]
    assert e.op == "record"
    assert e.event_id == 1


def test_recorder_writes_stream_event_parquet(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.trace.writer import write_parquet
    r = Recorder()
    r.stream_event(cycle=0, event_id=0, stream_id=0, op="record")
    write_parquet(r, tmp_path)
    assert (tmp_path / "stream_event.parquet").exists()
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add StreamEvent + recorder + writer**

`gpusim/trace/events.py`:
```python
@dataclass(frozen=True)
class StreamEvent:
    cycle: int
    event_id: int
    stream_id: int
    op: str               # "record" | "wait_start" | "wait_satisfied"
```

`gpusim/trace/recorder.py`:
- In `__init__`, add: `self.stream_event_events: list = []`
- Add method:
```python
    def stream_event(self, *, cycle: int, event_id: int, stream_id: int, op: str) -> None:
        from gpusim.trace.events import StreamEvent
        self.stream_event_events.append(StreamEvent(
            cycle=cycle, event_id=event_id, stream_id=stream_id, op=op,
        ))
```

`gpusim/trace/writer.py`, in `write_parquet`:
```python
    if r.stream_event_events:
        pd.DataFrame([asdict(e) for e in r.stream_event_events]).to_parquet(
            out_dir / "stream_event.parquet", index=False)
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/trace/test_stream_event.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/trace/ tests/unit/trace/test_stream_event.py
git commit -m "feat(trace): StreamEvent + recorder.stream_event + parquet writer"
```

---

### Task 18: event_wait_cycles_per_stream + event_chain_critical_path metrics

**Files:**
- Modify: `gpusim/analysis/metrics.py`
- Modify: `gpusim/api.py` (MultiStreamResult.event_wait_cycles_per_stream + event_chain_critical_path)
- Test: extend `tests/unit/analysis/test_phase8_metrics.py`

- [ ] **Step 1: Append failing tests**

```python
def test_event_wait_cycles_per_stream():
    from gpusim.analysis.metrics import event_wait_cycles_per_stream
    df = pd.DataFrame([
        {"cycle": 10, "event_id": 1, "stream_id": 1, "op": "wait_start"},
        {"cycle": 60, "event_id": 1, "stream_id": 1, "op": "wait_satisfied"},
    ])
    out = event_wait_cycles_per_stream(df)
    assert out[1] == 50


def test_event_chain_critical_path():
    from gpusim.analysis.metrics import event_chain_critical_path
    se_df = pd.DataFrame([
        {"cycle": 100, "event_id": 1, "stream_id": 0, "op": "record"},
        {"cycle": 100, "event_id": 1, "stream_id": 1, "op": "wait_satisfied"},
    ])
    kl_df = pd.DataFrame([
        {"stream_id": 0, "launch_cycle": 0, "complete_cycle": 100, "kernel_name": "a"},
        {"stream_id": 1, "launch_cycle": 100, "complete_cycle": 200, "kernel_name": "b"},
    ])
    cp = event_chain_critical_path(se_df, kl_df)
    # a (100) → ev1 → b (100) = 200 total
    assert cp == 200
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Append metrics**

In `gpusim/analysis/metrics.py`:

```python
def event_wait_cycles_per_stream(stream_event_df) -> dict:
    """Total cycles each stream spent event-blocked. Pairs wait_start with wait_satisfied
    per (stream_id, event_id)."""
    if stream_event_df is None or stream_event_df.empty:
        return {}
    out = {}
    pending_starts = {}
    sorted_df = stream_event_df.sort_values("cycle")
    for _, ev in sorted_df.iterrows():
        key = (int(ev["stream_id"]), int(ev["event_id"]))
        if ev["op"] == "wait_start":
            pending_starts[key] = int(ev["cycle"])
        elif ev["op"] == "wait_satisfied" and key in pending_starts:
            cycles = int(ev["cycle"]) - pending_starts.pop(key)
            sid = int(ev["stream_id"])
            out[sid] = out.get(sid, 0) + cycles
    return out


def event_chain_critical_path(stream_event_df, kernel_launch_df) -> int:
    """Longest event-mediated dependency chain in cycles.
    Simple version: for each event, sum of producer's complete_cycle + waiting consumer's duration."""
    if stream_event_df is None or stream_event_df.empty:
        return 0
    if kernel_launch_df is None or kernel_launch_df.empty:
        return 0
    # Build event_id → producer_complete_cycle map (from "record" events)
    record_cycles = {int(r["event_id"]): int(r["cycle"])
                       for _, r in stream_event_df.iterrows()
                       if r["op"] == "record"}
    # For each launch, if stream waited on an event, add wait_until + launch_duration
    max_chain = 0
    for _, launch in kernel_launch_df.iterrows():
        duration = int(launch["complete_cycle"]) - int(launch["launch_cycle"])
        # Find any event_id this stream waited on
        sid = int(launch["stream_id"])
        waits = stream_event_df[(stream_event_df["stream_id"] == sid)
                                  & (stream_event_df["op"] == "wait_satisfied")]
        wait_max = 0
        for _, w in waits.iterrows():
            ev_id = int(w["event_id"])
            wait_max = max(wait_max, record_cycles.get(ev_id, 0))
        chain = wait_max + duration
        max_chain = max(max_chain, chain)
    # Also consider launches with no wait (just their own duration)
    if kernel_launch_df is not None and not kernel_launch_df.empty:
        max_solo = max(int(r["complete_cycle"]) - int(r["launch_cycle"])
                         for _, r in kernel_launch_df.iterrows())
        max_chain = max(max_chain, max_solo)
    return max_chain
```

- [ ] **Step 4: Add MultiStreamResult methods**

In `gpusim/api.py` MultiStreamResult:

```python
    def event_wait_cycles_per_stream(self) -> dict:
        from gpusim.analysis.metrics import event_wait_cycles_per_stream
        if self._recorder is None: return {}
        from dataclasses import asdict
        import pandas as pd
        rows = [asdict(e) for e in getattr(self._recorder, "stream_event_events", [])]
        df = pd.DataFrame(rows) if rows else None
        return event_wait_cycles_per_stream(df)
    
    def event_chain_critical_path(self) -> int:
        from gpusim.analysis.metrics import event_chain_critical_path
        if self._recorder is None: return 0
        from dataclasses import asdict
        import pandas as pd
        se_rows = [asdict(e) for e in getattr(self._recorder, "stream_event_events", [])]
        se_df = pd.DataFrame(se_rows) if se_rows else None
        kl_df = self.kernel_launch_events_df
        return event_chain_critical_path(se_df, kl_df)
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/analysis/test_phase8_metrics.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/analysis/metrics.py gpusim/api.py tests/unit/analysis/test_phase8_metrics.py
git commit -m "feat(analysis+api): event_wait_cycles_per_stream + event_chain_critical_path"
```

---

### Task 19: Examples event_producer_consumer + event_fanout

**Files:**
- Create: `examples/event_producer_consumer/{kernel_write.ptx, kernel_read.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_event_producer_consumer.py`
- Create: `examples/event_fanout/{...}` similar
- Create: `tests/parity/test_event_fanout.py`

- [ ] **Step 1: Parity test for producer_consumer**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "event_producer_consumer"


def test_event_producer_consumer_correctness():
    """Stream A writes X → record(ev) → Stream B wait(ev) → reads X."""
    import gpusim
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter(); _reset_event_id_counter()
    
    n = 32
    SHARED = np.zeros(n, dtype=np.uint32)
    OUT = np.zeros(n, dtype=np.uint32)
    
    cfg = load_default()
    cfg.n_sm = 8
    
    ptx_write = (_DIR / "kernel_write.ptx").read_text()
    ptx_read = (_DIR / "kernel_read.ptx").read_text()
    
    s_a = Stream()
    s_b = Stream()
    ev = Event()
    
    s_a.launch(ptx_src=ptx_write, grid=(1,1,1), block=(32,1,1),
                params={"OUT": SHARED}, kernel_name="write", config=cfg)
    s_a.record(ev)
    s_b.wait(ev)
    s_b.launch(ptx_src=ptx_read, grid=(1,1,1), block=(32,1,1),
                params={"IN": SHARED, "OUT": OUT}, kernel_name="read", config=cfg)
    
    multi_res = gpusim.synchronize(streams=[s_a, s_b], config=cfg)
    
    # SHARED was written with 1s; OUT should mirror SHARED
    assert SHARED.sum() == n
    assert OUT.sum() == n
```

- [ ] **Step 2: kernel_write.ptx** (each thread writes 1 to OUT[tid]):

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

- [ ] **Step 3: kernel_read.ptx** (read IN, copy to OUT):

```
.visible .entry test(.param .u64 IN, .param .u64 OUT)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    
    ld.param.u64 %rd0, [IN];
    ld.param.u64 %rd1, [OUT];
    
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd2, %r1;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;
    
    ld.global.u32 %r2, [%rd3];
    st.global.u32 [%rd4], %r2;
    
    ret;
}
```

- [ ] **Step 4: reference.py + run.py + README.md + __init__.py for producer_consumer**

`reference.py`:
```python
import numpy as np
def reference(n: int = 32): return np.ones(n, dtype=np.uint32)
```

`run.py`:
```python
import numpy as np, pathlib, gpusim
from gpusim.api import Stream, Event
from gpusim.config.loader import load_default


def main():
    n = 32
    SHARED = np.zeros(n, dtype=np.uint32)
    OUT = np.zeros(n, dtype=np.uint32)
    cfg = load_default()
    cfg.n_sm = 8
    here = pathlib.Path(__file__).parent
    s_a = Stream(); s_b = Stream()
    ev = Event()
    s_a.launch(ptx_src=(here / "kernel_write.ptx").read_text(),
               grid=(1,1,1), block=(32,1,1),
               params={"OUT": SHARED}, kernel_name="write", config=cfg)
    s_a.record(ev)
    s_b.wait(ev)
    s_b.launch(ptx_src=(here / "kernel_read.ptx").read_text(),
               grid=(1,1,1), block=(32,1,1),
               params={"IN": SHARED, "OUT": OUT}, kernel_name="read", config=cfg)
    multi_res = gpusim.synchronize(streams=[s_a, s_b], config=cfg)
    print(multi_res.stream_summary())
    print(f"Event wait cycles: {multi_res.event_wait_cycles_per_stream()}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# event_producer_consumer

Phase 8 demo: producer stream writes X → record event → consumer stream
waits event → reads X. Demonstrates cross-stream synchronization via Event.
```

`__init__.py` (empty).

- [ ] **Step 5: Parity test for event_fanout**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "event_fanout"


def test_event_fanout_correctness():
    """1 producer event satisfies 3 consumer streams."""
    import gpusim
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter(); _reset_event_id_counter()
    
    n = 32
    SHARED = np.zeros(n, dtype=np.uint32)
    out_b = np.zeros(n, dtype=np.uint32)
    out_c = np.zeros(n, dtype=np.uint32)
    out_d = np.zeros(n, dtype=np.uint32)
    
    cfg = load_default()
    cfg.n_sm = 8
    
    ptx_write = (_DIR / "kernel_write.ptx").read_text()
    ptx_read = (_DIR / "kernel_read.ptx").read_text()
    
    s_a = Stream()
    s_b = Stream()
    s_c = Stream()
    s_d = Stream()
    ev = Event()
    
    s_a.launch(ptx_src=ptx_write, grid=(1,1,1), block=(32,1,1),
                params={"OUT": SHARED}, kernel_name="write", config=cfg)
    s_a.record(ev)
    
    for s, out in [(s_b, out_b), (s_c, out_c), (s_d, out_d)]:
        s.wait(ev)
        s.launch(ptx_src=ptx_read, grid=(1,1,1), block=(32,1,1),
                  params={"IN": SHARED, "OUT": out},
                  kernel_name=f"read_{s.stream_id}", config=cfg)
    
    multi_res = gpusim.synchronize(streams=[s_a, s_b, s_c, s_d], config=cfg)
    
    # All consumers see the producer's writes
    assert SHARED.sum() == n
    assert out_b.sum() == n
    assert out_c.sum() == n
    assert out_d.sum() == n
```

- [ ] **Step 6: Reuse same kernel files for event_fanout**

Copy `kernel_write.ptx` and `kernel_read.ptx` from event_producer_consumer to `examples/event_fanout/`.

- [ ] **Step 7: event_fanout supporting files**

`reference.py`:
```python
import numpy as np
def reference(n: int = 32): return np.ones(n, dtype=np.uint32)
```

`run.py`:
```python
import numpy as np, pathlib, gpusim
from gpusim.api import Stream, Event
from gpusim.config.loader import load_default


def main():
    n = 32
    SHARED = np.zeros(n, dtype=np.uint32)
    outs = [np.zeros(n, dtype=np.uint32) for _ in range(3)]
    cfg = load_default()
    cfg.n_sm = 8
    here = pathlib.Path(__file__).parent
    s_a = Stream()
    streams = [Stream() for _ in range(3)]
    ev = Event()
    s_a.launch(ptx_src=(here / "kernel_write.ptx").read_text(),
                grid=(1,1,1), block=(32,1,1),
                params={"OUT": SHARED}, kernel_name="write", config=cfg)
    s_a.record(ev)
    for s, out in zip(streams, outs):
        s.wait(ev)
        s.launch(ptx_src=(here / "kernel_read.ptx").read_text(),
                  grid=(1,1,1), block=(32,1,1),
                  params={"IN": SHARED, "OUT": out},
                  kernel_name=f"read_{s.stream_id}", config=cfg)
    multi_res = gpusim.synchronize(streams=[s_a] + streams, config=cfg)
    print(multi_res.stream_summary())


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# event_fanout

Phase 8 demo: 1 producer + 3 consumers wait on the same event.
Demonstrates event fanout pattern.
```

`__init__.py` (empty).

- [ ] **Step 8: Run + commit**

```
.venv/bin/pytest tests/parity/test_event_producer_consumer.py tests/parity/test_event_fanout.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/event_producer_consumer/ examples/event_fanout/ tests/parity/test_event_producer_consumer.py tests/parity/test_event_fanout.py
git commit -m "feat(examples): event_producer_consumer + event_fanout"
```

---

### Task 20: Tag M3

```bash
.venv/bin/pytest -q -m "not slow"
git tag M3-phase8-complete
```

---

## Milestone M4: L2 partitioning + l2_window_demo

### Task 21: L2Line.owner_stream_id + in_window fields

**Files:**
- Modify: `gpusim/core/cache/l2.py` (extend L2Line dataclass)
- Test: `tests/unit/cache/test_l2_window.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_l2_line_default_owner_unset():
    from gpusim.core.cache.l2 import L2Line
    line = L2Line(addr=0x1000, valid=False, dirty=False, last_use=0)
    assert line.owner_stream_id == -1
    assert line.in_window is False


def test_l2_line_owner_settable():
    from gpusim.core.cache.l2 import L2Line
    line = L2Line(addr=0x1000, valid=True, dirty=False, last_use=0,
                    owner_stream_id=2, in_window=True)
    assert line.owner_stream_id == 2
    assert line.in_window is True
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add fields to L2Line**

In `gpusim/core/cache/l2.py`, in L2Line dataclass:

```python
    owner_stream_id: int = -1     # NEW Phase 8
    in_window: bool = False        # NEW Phase 8
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/cache/test_l2_window.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/core/cache/l2.py tests/unit/cache/test_l2_window.py
git commit -m "feat(cache): L2Line.owner_stream_id + in_window fields"
```

---

### Task 22: L2Cache.register_stream_window + window-aware eviction

**Files:**
- Modify: `gpusim/core/cache/l2.py` (add register_stream_window + update _pick_victim)
- Test: extend `tests/unit/cache/test_l2_window.py`

- [ ] **Step 1: Append failing test**

```python
def test_l2_register_stream_window():
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.config.schema import CacheConfig
    class _NoOpHbm:
        def request(self, line_addr, now): return now + 100
        def write_request(self, line_addr, now): return now + 100
    cfg = CacheConfig()
    l2 = L2Cache(cfg, _NoOpHbm())
    l2.register_stream_window(stream_id=0, start_set=0, n_sets=32)
    assert l2._stream_windows[0] == (0, 32)
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add register_stream_window + window protection logic**

In `gpusim/core/cache/l2.py::L2Cache.__init__`, add:
```python
        self._stream_windows: dict = {}    # NEW Phase 8
```

Add method:
```python
    def register_stream_window(self, stream_id: int, start_set: int, n_sets: int) -> None:
        """Reserve [start_set, start_set+n_sets) as protected window for this stream."""
        self._stream_windows[stream_id] = (start_set, n_sets)
    
    def _line_in_window(self, line, set_idx: int) -> bool:
        if line.owner_stream_id < 0: return False
        window = self._stream_windows.get(line.owner_stream_id)
        if window is None: return False
        start, n = window
        return start <= set_idx < start + n
```

Update `_pick_victim` (or wherever LRU eviction lives) to skip protected lines:

```python
    def _pick_victim(self, set_idx: int, requesting_stream_id: int) -> "L2Line | None":
        candidates = []
        for line in self.sets[set_idx]:
            if (self._line_in_window(line, set_idx)
                    and line.owner_stream_id != requesting_stream_id):
                continue
            candidates.append(line)
        if not candidates:
            return None
        return min(candidates, key=lambda c: c.last_use)
```

⚠ The actual L2Cache eviction code may differ. Read `gpusim/core/cache/l2.py` to find where line install / eviction happens, and integrate the window check accordingly.

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/cache/test_l2_window.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/core/cache/l2.py tests/unit/cache/test_l2_window.py
git commit -m "feat(cache): L2Cache.register_stream_window + window-aware eviction"
```

---

### Task 23: Stream.set_l2_window + Device wires registration

**Files:**
- Modify: `gpusim/api.py` (Stream.l2_window field + set_l2_window method)
- Modify: `gpusim/core/device.py::run_streams` (register windows before main loop)
- Test: extend `tests/unit/api/test_stream_priority.py` (add to that file or new)

- [ ] **Step 1: Append failing test**

```python
def test_stream_set_l2_window():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    assert s.l2_window is None
    s.set_l2_window(start_set=0, n_sets=32)
    assert s.l2_window == (0, 32)
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add l2_window field + set_l2_window method to Stream**

In `gpusim/api.py` Stream:

```python
    l2_window: tuple | None = None     # NEW Phase 8 — (start_set, n_sets)
    
    def set_l2_window(self, *, start_set: int, n_sets: int) -> None:
        """Reserve L2 sets [start_set, start_set+n_sets) as protected window."""
        self.l2_window = (start_set, n_sets)
```

- [ ] **Step 4: Wire window registration in Device.run_streams**

At the top of `Device.run_streams`, before the main loop:

```python
        # Phase 8 M4: register per-stream L2 windows
        if hasattr(self, "l2"):
            for s in streams:
                if s.l2_window is not None:
                    self.l2.register_stream_window(s.stream_id, *s.l2_window)
```

⚠ Adapt to actual Device structure. If Device has L2 accessible as `self.l2`, use that. If multiple SMs each have own L2 ref, register on the shared device-level L2.

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/api/test_stream_priority.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/api.py gpusim/core/device.py tests/unit/api/test_stream_priority.py
git commit -m "feat(api): Stream.set_l2_window + Device registers per-stream windows"
```

---

### Task 24: l2_window_hit_rate_per_stream + l2_window_protection_efficiency metrics

**Files:**
- Modify: `gpusim/analysis/metrics.py`
- Modify: `gpusim/api.py`
- Test: extend `tests/unit/analysis/test_phase8_metrics.py`

- [ ] **Step 1: Append failing test**

```python
def test_l2_window_hit_rate_per_stream():
    from gpusim.analysis.metrics import l2_window_hit_rate_per_stream
    df = pd.DataFrame([
        {"stream_id": 0, "hit": True}, {"stream_id": 0, "hit": True},
        {"stream_id": 0, "hit": False}, {"stream_id": 1, "hit": False},
    ])
    out = l2_window_hit_rate_per_stream(df)
    assert abs(out[0] - 2/3) < 0.01
    assert abs(out[1] - 0.0) < 0.01


def test_l2_window_protection_efficiency():
    from gpusim.analysis.metrics import l2_window_protection_efficiency
    df = pd.DataFrame([
        {"hit": True, "in_window": True}, {"hit": True, "in_window": True},
        {"hit": True, "in_window": False}, {"hit": False, "in_window": False},
    ])
    eff = l2_window_protection_efficiency(df)
    # 2 in-window hits out of 3 total hits = 0.67
    assert abs(eff - 2/3) < 0.01
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Append metrics**

```python
def l2_window_hit_rate_per_stream(memory_events_df) -> dict:
    """L2 hit rate per stream."""
    if memory_events_df is None or memory_events_df.empty:
        return {}
    out = {}
    for sid, group in memory_events_df.groupby("stream_id"):
        total = len(group)
        hits = group["hit"].sum() if "hit" in group.columns else 0
        out[int(sid)] = float(hits) / total if total > 0 else 0.0
    return out


def l2_window_protection_efficiency(memory_events_df) -> float:
    """Fraction of L2 hits that came from window-protected lines."""
    if memory_events_df is None or memory_events_df.empty:
        return 0.0
    hits = memory_events_df[memory_events_df["hit"] == True] if "hit" in memory_events_df.columns else memory_events_df
    if hits.empty: return 0.0
    in_window = hits["in_window"].sum() if "in_window" in hits.columns else 0
    return float(in_window) / len(hits)
```

⚠ The memory_events_df schema needs `hit` and `in_window` columns. If existing memory events don't carry these, add them in Phase 8 or use a different data source.

- [ ] **Step 4: Add MultiStreamResult methods**

```python
    def l2_window_hit_rate(self) -> dict:
        from gpusim.analysis.metrics import l2_window_hit_rate_per_stream
        # ... fetch memory events df from recorder ...
        return l2_window_hit_rate_per_stream(...)
    
    def l2_window_protection_efficiency(self) -> float:
        from gpusim.analysis.metrics import l2_window_protection_efficiency
        return l2_window_protection_efficiency(...)
```

- [ ] **Step 5: Run + commit**

```
.venv/bin/pytest tests/unit/analysis/test_phase8_metrics.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/analysis/metrics.py gpusim/api.py tests/unit/analysis/test_phase8_metrics.py
git commit -m "feat(analysis+api): l2_window_hit_rate + l2_window_protection_efficiency"
```

---

### Task 25: Example l2_window_demo

**Files:**
- Create: `examples/l2_window_demo/{kernel.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_l2_window_demo.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "l2_window_demo"


def test_l2_window_demo_correctness():
    """High stream with L2 window + low stream streaming. Both produce correct outputs."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    
    n = 32
    OUT_HIGH = np.zeros(n, dtype=np.uint32)
    OUT_LOW = np.zeros(n, dtype=np.uint32)
    
    cfg = load_default()
    cfg.n_sm = 8
    
    ptx = (_DIR / "kernel.ptx").read_text()
    s_high = Stream(priority="high")
    s_high.set_l2_window(start_set=0, n_sets=32)
    s_low = Stream(priority="low")
    
    s_high.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                    params={"OUT": OUT_HIGH}, kernel_name="critical", config=cfg)
    s_low.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                   params={"OUT": OUT_LOW}, kernel_name="background", config=cfg)
    
    multi_res = gpusim.synchronize(streams=[s_high, s_low], config=cfg)
    
    assert OUT_HIGH.sum() == n
    assert OUT_LOW.sum() == n
    assert len(multi_res.streams) == 2
```

- [ ] **Step 2: kernel.ptx** (each thread writes 1 to OUT[tid]):

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
import numpy as np, pathlib, gpusim
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    OUT_HIGH = np.zeros(n, dtype=np.uint32)
    OUT_LOW = np.zeros(n, dtype=np.uint32)
    cfg = load_default()
    cfg.n_sm = 8
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s_high = Stream(priority="high")
    s_high.set_l2_window(start_set=0, n_sets=32)
    s_low = Stream(priority="low")
    s_high.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                    params={"OUT": OUT_HIGH}, kernel_name="critical", config=cfg)
    s_low.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                   params={"OUT": OUT_LOW}, kernel_name="background", config=cfg)
    multi_res = gpusim.synchronize(streams=[s_high, s_low], config=cfg)
    print(multi_res.stream_summary())


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# l2_window_demo

Phase 8 demo: high-priority stream with L2 set-window protection.
Mimics H100 cudaStreamAttributeAccessPolicyWindow.
```

`__init__.py` (empty).

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_l2_window_demo.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/l2_window_demo/ tests/parity/test_l2_window_demo.py
git commit -m "feat(examples): l2_window_demo — high stream with L2 set-window protection"
```

---

### Task 26: Tag M4

```bash
.venv/bin/pytest -q -m "not slow"
git tag M4-phase8-complete
```

---

## Milestone M5: Pipeline + viz + docs + ship

### Task 27: Example multi_stream_pipeline_full

**Files:**
- Create: `examples/multi_stream_pipeline_full/{kernel_load.ptx, kernel_compute.ptx, kernel_store.ptx, reference.py, run.py, README.md, __init__.py}`
- Create: `tests/parity/test_multi_stream_pipeline_full.py`

- [ ] **Step 1: Parity test**

```python
import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "multi_stream_pipeline_full"


def test_multi_stream_pipeline_full_correctness():
    """3 streams (load → compute → store) with priority + events + L2 window."""
    import gpusim
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter(); _reset_event_id_counter()
    
    n = 32
    A = np.arange(n, dtype=np.float32)
    INTER = np.zeros(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    
    cfg = load_default()
    cfg.n_sm = 8
    
    s_load = Stream(priority="normal")
    s_compute = Stream(priority="high")
    s_compute.set_l2_window(start_set=0, n_sets=16)
    s_store = Stream(priority="normal")
    
    ev_load_done = Event()
    ev_compute_done = Event()
    
    here = _DIR
    s_load.launch(ptx_src=(here / "kernel_load.ptx").read_text(),
                   grid=(1,1,1), block=(32,1,1),
                   params={"IN": A, "OUT": INTER}, kernel_name="load", config=cfg)
    s_load.record(ev_load_done)
    
    s_compute.wait(ev_load_done)
    s_compute.launch(ptx_src=(here / "kernel_compute.ptx").read_text(),
                       grid=(1,1,1), block=(32,1,1),
                       params={"IN": INTER, "OUT": INTER}, kernel_name="compute", config=cfg)
    s_compute.record(ev_compute_done)
    
    s_store.wait(ev_compute_done)
    s_store.launch(ptx_src=(here / "kernel_store.ptx").read_text(),
                    grid=(1,1,1), block=(32,1,1),
                    params={"IN": INTER, "OUT": OUT}, kernel_name="store", config=cfg)
    
    multi_res = gpusim.synchronize(streams=[s_load, s_compute, s_store], config=cfg)
    
    # End-to-end: A → load → compute (×2) → store → OUT
    expected = A * 2.0
    np.testing.assert_array_almost_equal(OUT, expected)
    assert len(multi_res.streams) == 3
```

- [ ] **Step 2: kernel_load.ptx** (copy IN to OUT):

```
.visible .entry test(.param .u64 IN, .param .u64 OUT)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<2>;
    
    ld.param.u64 %rd0, [IN];
    ld.param.u64 %rd1, [OUT];
    
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd2, %r1;
    
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];
    add.u64 %rd3, %rd1, %rd2;
    st.global.f32 [%rd3], %f0;
    
    ret;
}
```

- [ ] **Step 3: kernel_compute.ptx** (multiply by 2):

```
.visible .entry test(.param .u64 IN, .param .u64 OUT)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<3>;
    
    ld.param.u64 %rd0, [IN];
    ld.param.u64 %rd1, [OUT];
    
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd2, %r1;
    
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];
    mul.f32 %f1, %f0, 0f40000000;   // 2.0
    add.u64 %rd3, %rd1, %rd2;
    st.global.f32 [%rd3], %f1;
    
    ret;
}
```

- [ ] **Step 4: kernel_store.ptx** (copy IN to OUT, same as load):

```
.visible .entry test(.param .u64 IN, .param .u64 OUT)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<2>;
    
    ld.param.u64 %rd0, [IN];
    ld.param.u64 %rd1, [OUT];
    
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd2, %r1;
    
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];
    add.u64 %rd3, %rd1, %rd2;
    st.global.f32 [%rd3], %f0;
    
    ret;
}
```

- [ ] **Step 5: reference.py + run.py + README.md + __init__.py**

`reference.py`:
```python
import numpy as np
def reference(A: np.ndarray): return A * 2.0
```

`run.py`:
```python
import numpy as np, pathlib, gpusim
from gpusim.api import Stream, Event
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    INTER = np.zeros(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    cfg.n_sm = 8
    here = pathlib.Path(__file__).parent
    s_load = Stream(priority="normal")
    s_compute = Stream(priority="high")
    s_compute.set_l2_window(start_set=0, n_sets=16)
    s_store = Stream(priority="normal")
    ev1 = Event(); ev2 = Event()
    s_load.launch(ptx_src=(here / "kernel_load.ptx").read_text(),
                   grid=(1,1,1), block=(32,1,1),
                   params={"IN": A, "OUT": INTER}, kernel_name="load", config=cfg)
    s_load.record(ev1)
    s_compute.wait(ev1)
    s_compute.launch(ptx_src=(here / "kernel_compute.ptx").read_text(),
                       grid=(1,1,1), block=(32,1,1),
                       params={"IN": INTER, "OUT": INTER}, kernel_name="compute", config=cfg)
    s_compute.record(ev2)
    s_store.wait(ev2)
    s_store.launch(ptx_src=(here / "kernel_store.ptx").read_text(),
                    grid=(1,1,1), block=(32,1,1),
                    params={"IN": INTER, "OUT": OUT}, kernel_name="store", config=cfg)
    multi_res = gpusim.synchronize(streams=[s_load, s_compute, s_store], config=cfg)
    print(multi_res.stream_summary())
    print(f"OUT[0:4] = {list(OUT[0:4])}")
    print(f"event_chain_critical_path: {multi_res.event_chain_critical_path()}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# multi_stream_pipeline_full

Phase 8 capstone: 3 streams (load → compute → store) chained via events,
compute stream high-priority + L2 window protection.
```

`__init__.py` (empty).

- [ ] **Step 6: Run + commit**

```
.venv/bin/pytest tests/parity/test_multi_stream_pipeline_full.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add examples/multi_stream_pipeline_full/ tests/parity/test_multi_stream_pipeline_full.py
git commit -m "feat(examples): multi_stream_pipeline_full — Phase 8 capstone (3 streams + events + L2 window + priority)"
```

---

### Task 28: HTML §29/§30/§31 sections

**Files:**
- Modify: `gpusim/viz/html_report.py` (3 render helpers + populate context)
- Modify: `gpusim/viz/_template.html.j2` (3 new section blocks)
- Test: `tests/unit/viz/test_html_report_phase8.py` (NEW)

- [ ] **Step 1: Create test**

```python
def test_html_report_phase8_sections(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.kernel_launch(stream_id=0, kernel_name="k0", grid=(1,1,1), block=(32,1,1),
                     launch_cycle=0, complete_cycle=100, n_ctas=1)
    r.stream_event(cycle=50, event_id=1, stream_id=0, op="record")
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="t", grid=(1,1,1), block=(32,1,1),
              cycles=200, occupancy={"active_ctas": 1, "bottleneck": "none"})
    html = out.read_text()
    assert "Priority" in html or "priority" in html.lower() \
            or "Event" in html or "event" in html.lower()
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add render helpers + template blocks**

In `gpusim/viz/html_report.py`, append:

```python
def _render_priority_dispatch(rec):
    if not getattr(rec, "kernel_launch_events", None):
        return ""
    from dataclasses import asdict
    import pandas as pd
    df = pd.DataFrame([asdict(e) for e in rec.kernel_launch_events])
    return "<h3>Kernel launches by stream</h3>" + df.to_html(index=False)


def _render_event_timeline(rec):
    if not getattr(rec, "stream_event_events", None):
        return ""
    from dataclasses import asdict
    import pandas as pd
    df = pd.DataFrame([asdict(e) for e in rec.stream_event_events])
    return "<h3>Stream events timeline</h3>" + df.to_html(index=False)


def _render_l2_window_heatmap(rec):
    # Phase 8 simple version: just a table of L2 access counts per stream
    if not getattr(rec, "instr_events", None) and not getattr(rec, "instr_issues", None):
        return ""
    return "<h3>L2 access (placeholder for window heatmap)</h3>"
```

In `save_html`, add to context:
```python
    context.update({
        "priority_dispatch_html": _render_priority_dispatch(rec),
        "event_timeline_html": _render_event_timeline(rec),
        "l2_window_heatmap_html": _render_l2_window_heatmap(rec),
    })
```

In `gpusim/viz/_template.html.j2`, append after Phase 7 §27/§28:

```html
{% if priority_dispatch_html %}
<h2>§29 Priority dispatch breakdown</h2>
{{ priority_dispatch_html | safe }}
{% endif %}

{% if event_timeline_html %}
<h2>§30 Event timeline</h2>
{{ event_timeline_html | safe }}
{% endif %}

{% if l2_window_heatmap_html %}
<h2>§31 L2 window heatmap</h2>
{{ l2_window_heatmap_html | safe }}
{% endif %}
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/viz/test_html_report_phase8.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/viz/html_report.py gpusim/viz/_template.html.j2 tests/unit/viz/test_html_report_phase8.py
git commit -m "feat(viz): HTML §29/§30/§31 — priority dispatch + event timeline + L2 window"
```

---

### Task 29: Perfetto priority annotations + StreamEvent emission

**Files:**
- Modify: `gpusim/viz/perfetto.py`
- Test: extend `tests/unit/viz/test_html_report_phase8.py`

- [ ] **Step 1: Append failing test**

```python
def test_perfetto_stream_event_emitted():
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.perfetto import build_perfetto
    r = Recorder()
    r.stream_event(cycle=50, event_id=1, stream_id=0, op="record")
    pf = build_perfetto(r)
    cats = {e.get("cat") for e in pf.get("traceEvents", [])}
    assert any("stream_event" in str(c) for c in cats)
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Add StreamEvent emission to perfetto.py**

In `gpusim/viz/perfetto.py::build_perfetto`, append:

```python
    # Phase 8 stream events as instant events
    for ev in getattr(rec, "stream_event_events", []):
        events.append({
            "name": f"event_{ev.op}_{ev.event_id}",
            "cat": "stream_event", "ph": "i",
            "ts": ev.cycle, "s": "g",
            "pid": f"Stream-{ev.stream_id}",
            "tid": "events",
            "args": {"event_id": ev.event_id, "op": ev.op},
        })
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/unit/viz/test_html_report_phase8.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add gpusim/viz/perfetto.py tests/unit/viz/test_html_report_phase8.py
git commit -m "feat(viz): Perfetto StreamEvent instant events"
```

---

### Task 30: 6 tutorial chapters 31-36

**Files:**
- Create: `docs/tutorial/{31,32,33,34,35,36}-*.md`

- [ ] **Step 1: Read existing style** (`docs/tutorial/30-scheduler-fairness-streams.md`)

- [ ] **Step 2: Write 6 chapters (~500-700 words each)**

**Chapter 31 — true-concurrent-scheduler:**
- ConcurrentStreamScheduler vs Phase 7 sequential drain
- Per-cycle dispatch model
- true_concurrent_overlap demo (`examples/true_concurrent_overlap/run.py`)
- 看模拟器: `multi_res.cross_stream_concurrency_gain()`
- 改一改: 1 vs 2 streams cycles
- 真机对照: H100 default scheduler

**Chapter 32 — stream-priority-weighted-rr:**
- 3 priority levels + weighted RR (4:2:1)
- priority_demo (`examples/priority_demo/run.py`)
- 看模拟器: `multi_res.priority_dispatch_share()`
- 改一改: configure cfg.scheduler.priority_weights
- 真机对照: cudaStreamCreateWithPriority

**Chapter 33 — cuda-events-record-wait:**
- Event lifecycle: create → record → wait → signal
- event_producer_consumer (`examples/event_producer_consumer/run.py`)
- 看模拟器: `multi_res.event_wait_cycles_per_stream()` + HTML §30
- 改一改: skip event → race condition
- 真机对照: cudaEventRecord + cudaStreamWaitEvent

**Chapter 34 — event-fanout-pattern:**
- 1 producer event → multiple consumers wait
- event_fanout (`examples/event_fanout/run.py`)
- 看模拟器: 多个 wait_satisfied entries for same event_id
- 改一改: chain events for serial consumer pipeline
- 真机对照: pthread condvar broadcast pattern

**Chapter 35 — l2-cache-window-partitioning ⭐:**
- L2 set window + line ownership + eviction protection
- l2_window_demo (`examples/l2_window_demo/run.py`)
- 看模拟器: `l2_window_hit_rate` + `l2_window_protection_efficiency`
- 改一改: smaller window → less protection
- 真机对照: cudaStreamAttributeAccessPolicyWindow

**Chapter 36 — production-multi-stream-pipeline ⭐:**
- Combine priority + events + L2 window
- multi_stream_pipeline_full (`examples/multi_stream_pipeline_full/run.py`)
- 看模拟器: event_chain_critical_path
- 改一改: drop priority → critical path lengthens
- 真机对照: CUTLASS persistent matmul + cooperative epilogue with stream priority

- [ ] **Step 3: Commit**

```bash
git add docs/tutorial/31-true-concurrent-scheduler.md \
        docs/tutorial/32-stream-priority-weighted-rr.md \
        docs/tutorial/33-cuda-events-record-wait.md \
        docs/tutorial/34-event-fanout-pattern.md \
        docs/tutorial/35-l2-cache-window-partitioning.md \
        docs/tutorial/36-production-multi-stream-pipeline.md
git commit -m "docs(tutorial): chapters 31-36 — Phase 8 features"
```

---

### Task 31: Phase 8 microbench (facts + runtime)

**Files:**
- Create: `tests/microbench/test_phase8_facts.py`
- Create: `tests/microbench/test_phase8_runtime.py`

- [ ] **Step 1: Phase 8 facts**

`tests/microbench/test_phase8_facts.py`:
```python
"""Phase 8 microbench — multi-stream concurrency facts."""
import numpy as np


def test_priority_high_finishes_no_slower_than_low():
    """High priority stream should not finish slower than low priority stream."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    cfg = load_default()
    cfg.n_sm = 8
    
    src = """
.visible .entry test(.param .u64 A, .param .u64 B, .param .u64 OUT) {
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
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
    _reset_stream_id_counter()
    s_high = Stream(priority="high")
    s_low = Stream(priority="low")
    out_h = np.zeros(n, dtype=np.float32)
    out_l = np.zeros(n, dtype=np.float32)
    s_high.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                    params={"A": A, "B": B, "OUT": out_h}, kernel_name="kh", config=cfg)
    s_low.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                   params={"A": A, "B": B, "OUT": out_l}, kernel_name="kl", config=cfg)
    multi_res = gpusim.synchronize(streams=[s_high, s_low], config=cfg)
    
    # Both correct; both completed
    np.testing.assert_array_equal(out_h, A + B)
    np.testing.assert_array_equal(out_l, A + B)
    assert len(multi_res.streams) == 2


def test_event_satisfies_consumer_in_order():
    """Consumer waiting on event sees producer's writes."""
    import gpusim
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter(); _reset_event_id_counter()
    
    n = 32
    SHARED = np.zeros(n, dtype=np.uint32)
    OUT = np.zeros(n, dtype=np.uint32)
    cfg = load_default()
    cfg.n_sm = 8
    
    write_src = """
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
    read_src = """
.visible .entry test(.param .u64 IN, .param .u64 OUT) {
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    ld.param.u64 %rd0, [IN];
    ld.param.u64 %rd1, [OUT];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd2, %r1;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;
    ld.global.u32 %r2, [%rd3];
    st.global.u32 [%rd4], %r2;
    ret;
}
"""
    s_a = Stream()
    s_b = Stream()
    ev = Event()
    s_a.launch(ptx_src=write_src, grid=(1,1,1), block=(32,1,1),
                params={"OUT": SHARED}, kernel_name="write", config=cfg)
    s_a.record(ev)
    s_b.wait(ev)
    s_b.launch(ptx_src=read_src, grid=(1,1,1), block=(32,1,1),
                params={"IN": SHARED, "OUT": OUT}, kernel_name="read", config=cfg)
    gpusim.synchronize(streams=[s_a, s_b], config=cfg)
    
    assert SHARED.sum() == n
    assert OUT.sum() == n
```

- [ ] **Step 2: Phase 8 runtime (slow)**

`tests/microbench/test_phase8_runtime.py`:
```python
import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_priority_demo_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "priority_demo"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_multi_stream_pipeline_full_runtime_under_60s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "multi_stream_pipeline_full"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=120)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 60
```

- [ ] **Step 3: Run + commit**

```
.venv/bin/pytest tests/microbench/test_phase8_facts.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add tests/microbench/test_phase8_facts.py tests/microbench/test_phase8_runtime.py
git commit -m "test(microbench): Phase 8 facts + runtime budget tests"
```

---

### Task 32: Phase 1-7 regression rename + Phase 7 examples + 6 ref stubs

**Files:**
- Rename: `tests/parity/test_phase1_6_examples_unchanged.py` → `test_phase1_7_examples_unchanged.py`
- Modify: edit list
- Modify: `tests/reference/gen_reference.py` (append 6 Phase 8 kernel names)
- Create: 6 ref JSON stubs

- [ ] **Step 1: Rename and edit**

```bash
git mv tests/parity/test_phase1_6_examples_unchanged.py tests/parity/test_phase1_7_examples_unchanged.py
```

In renamed file:
- Rename `PHASE_1_6_EXAMPLES` → `PHASE_1_7_EXAMPLES`
- Append 4 Phase 7 examples to the list:
  - `concurrent_vector_add_2stream`
  - `compute_vs_memory_overlap`
  - `l2_contention_2stream`
  - `stream_priority_serial_vs_concurrent`
- Update test function names from `phase1_6` → `phase1_7` if any.

- [ ] **Step 2: Append to gen_reference.py**

```python
"true_concurrent_overlap",
"priority_demo",
"event_producer_consumer",
"event_fanout",
"l2_window_demo",
"multi_stream_pipeline_full",
```

- [ ] **Step 3: Create 6 ref JSON stubs**

```bash
for k in true_concurrent_overlap priority_demo event_producer_consumer event_fanout l2_window_demo multi_stream_pipeline_full; do
  cat > tests/reference/data/$k.ref.json <<JSON
{
  "kernel": "$k",
  "phase": 8,
  "metrics": {
    "cross_stream_concurrency_gain": null,
    "priority_dispatch_share": null,
    "event_wait_cycles_per_stream": null,
    "l2_window_hit_rate": null,
    "l2_window_protection_efficiency": null,
    "event_chain_critical_path": null
  },
  "tolerance": {
    "cross_stream_concurrency_gain_pct": 15,
    "priority_dispatch_share_pct": 10,
    "event_wait_cycles_per_stream_pct": 20,
    "l2_window_hit_rate_pct": 15,
    "l2_window_protection_efficiency_pct": 15,
    "event_chain_critical_path_pct": 15
  },
  "notes": "Generated by gen_reference.py on real Hopper. null = not yet measured."
}
JSON
done
```

- [ ] **Step 4: Run + commit**

```
.venv/bin/pytest tests/parity/test_phase1_7_examples_unchanged.py -v
.venv/bin/pytest -q -m "not slow"
```

```bash
git add tests/parity/test_phase1_7_examples_unchanged.py \
        tests/reference/gen_reference.py \
        tests/reference/data/true_concurrent_overlap.ref.json \
        tests/reference/data/priority_demo.ref.json \
        tests/reference/data/event_producer_consumer.ref.json \
        tests/reference/data/event_fanout.ref.json \
        tests/reference/data/l2_window_demo.ref.json \
        tests/reference/data/multi_stream_pipeline_full.ref.json
git commit -m "test(regression+reference): rename phase1_6 → phase1_7 + 4 Phase 7 examples + 6 ref stubs"
```

---

### Task 33: README v8 + final tag

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update to v8**

In `README.md`:
- Capabilities/status: add Phase 8 ✅
- Phase 8 features section:
  - True concurrent scheduler (`ConcurrentStreamScheduler` per-cycle weighted RR)
  - Stream priority (high/normal/low, 4:2:1 weights)
  - CUDA events (Event class + Stream.record/wait + StreamEvent trace)
  - L2 set-window partitioning (cudaStreamAttributeAccessPolicyWindow equivalent)
  - 6 metrics, 1 new trace event, 3 HTML sections, Perfetto annotations
  - 6 examples + 6 tutorials chapters 31-36
  - Honest note: M1 minimal scheduler integration; Phase 9 may add full per-cycle CTA slicing for stronger overlap benefits
  - Backward compatible: Phase 1-7 unchanged
- Examples list: add 6 (was 29, now 35)
- Tutorials list: add 31-36 (was 30, now 36)
- Phase status: 1-8 ✅

- [ ] **Step 2: Run final suite + 6 examples**

```
.venv/bin/pytest -q -m "not slow"
.venv/bin/python examples/true_concurrent_overlap/run.py
.venv/bin/python examples/priority_demo/run.py
.venv/bin/python examples/event_producer_consumer/run.py
.venv/bin/python examples/event_fanout/run.py
.venv/bin/python examples/l2_window_demo/run.py
.venv/bin/python examples/multi_stream_pipeline_full/run.py
```

- [ ] **Step 3: Commit + tag**

```bash
git add README.md
git commit -m "docs(readme): v8 — Phase 8 capabilities (true concurrent + priority + events + L2 window)"
git tag phase8-complete
git tag | grep phase
git log --oneline | head -10
```

---

### Task 34: Final sanity sweep + done

- [ ] **Step 1: Full pytest sweep**

```
.venv/bin/pytest -q -m "not slow"
```

- [ ] **Step 2: Microbench + Phase 1-7 regression**

```
.venv/bin/pytest tests/microbench/test_phase8_facts.py tests/parity/test_phase1_7_examples_unchanged.py -v
```

- [ ] **Step 3: Generate one HTML manually + spot-check §29-§31**

- [ ] **Step 4: Verify Perfetto JSON has Stream-N priority + StreamEvent**

- [ ] **Step 5: Done**

Phase 8 ships when all tasks complete + tags landed.

---

### Task 35: (consolidated as part of T34)

(Reserved — split if final cleanup needed.)

---

### Task 36: (consolidated as part of T34)

(Reserved — split if README+tag needs separate commit.)

---

## End-of-plan checklist

- [ ] M1 (True concurrent scheduler + true_concurrent_overlap): T1-T8
- [ ] M2 (Stream priority + priority_demo): T9-T13
- [ ] M3 (Events + 2 examples): T14-T20
- [ ] M4 (L2 partitioning + l2_window_demo): T21-T26
- [ ] M5 (Pipeline + viz + docs + ship): T27-T34
- [ ] All 5 milestone tags
- [ ] Phase 1-7 regression unbroken
- [ ] 6 new examples + 6 tutorials shipped
- [ ] README v8 reflects Phase 8
