# gpusim Phase 7 — Multi-stream / Multi-kernel Concurrency

> **Status:** Brainstormed 2026-05-09. Implementation TBD.

## 1. Goals & Non-goals

### Goals
- Add **multi-stream / multi-kernel concurrent execution** to gpusim.
- Provide a CUDA-stream-like API: `gpusim.Stream()` + `Stream.launch(...)` + `gpusim.synchronize()`.
- Reuse Phase 4 multi-SM device + L2/HBM sharing — multi-stream tags every event with `stream_id` so contention/fairness can be analyzed.
- Schedule CTAs **round-robin across active streams**, fair at CTA granularity (Hopper default).
- Ship 4 examples + 4 tutorial chapters demonstrating: basic concurrency, compute/memory overlap, L2 contention, serial vs concurrent fairness.
- Full trace (1 new event + `stream_id` propagated to all 11 existing events) + 4 analysis metrics + 2 HTML sections + per-stream Perfetto swimlanes.
- 100% backward compatible: `gpusim.run(...)` single-kernel path is unchanged; it executes implicitly on a default stream.

### Non-goals (deferred to Phase 8+)
- **Stream priorities** (`cudaStreamCreateWithPriority`).
- **Events / cross-stream sync** (`cudaEventRecord` / `cudaStreamWaitEvent`).
- **CUDA Graphs** (DAG of kernels with explicit dependencies).
- **L2 cache partitioning per stream** (`cudaStreamAttribute_L2WindowSize`).
- **Persistent kernels** / dynamic parallelism.
- **HBM channel arbitration tracking per stream** (analytics only — already correct cycles).

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  gpusim.Stream(0)        gpusim.Stream(1)        ...        │
│  ├─ launch(ptx, grid, block, params)                        │
│  ├─ pending_grids: deque[GridLaunch]   (launch order)       │
│  └─ completed: list[Result]                                 │
└─────────────────┬─────────────────────┬─────────────────────┘
                  │                     │
                  ▼                     ▼
        ┌─────────────────────────────────────┐
        │  Device.run_streams(streams)        │
        │  ┌─ MultiStreamScheduler ──────┐    │
        │  │ Each cycle:                  │    │
        │  │   for stream in RR_order:    │    │
        │  │     pick next CTA            │    │
        │  │     find SM with capacity    │    │
        │  │     dispatch (tag stream_id) │    │
        │  └──────────────────────────────┘    │
        └─────────────────────────────────────┘
                  │
                  ▼
        ┌─────────────────────────────────────┐
        │  N SMs (Phase 4) — share L2 + HBM   │
        │  Each CTA tagged with stream_id     │
        │  Trace events all carry stream_id   │
        └─────────────────────────────────────┘
```

### Key invariants
- **Single device, N SMs** (Phase 4 unchanged) — multi-stream shares the same hardware.
- **CTA dispatch is the binding moment** — when scheduler dispatches a CTA, that CTA's `stream_id` is recorded; all subsequent events from this CTA carry that `stream_id`.
- **Stream-internal ordering** — within a single Stream, multiple `launch(...)` calls execute strictly in launch order (next grid does not start dispatching until prior grid is fully retired).
- **Cross-stream concurrency** — different streams' CTAs interleave freely on SMs.
- **No device-wide barrier** — `gpusim.synchronize()` simply drains all streams to completion.
- **Phase 1-6 backward compat**: `gpusim.run(...)` is internally a single-stream `run_streams([implicit_stream])`.

---

## 3. Data model

### 3.1 New `gpusim.Stream` class (`gpusim/api.py`)

```python
@dataclass
class GridLaunch:
    ptx_src: str
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    params: dict
    config: Config | None = None     # if None, inherit from synchronize()


@dataclass
class Stream:
    stream_id: int                   # auto-assigned (monotonic) on construction
    pending: deque[GridLaunch]       # launches not yet started dispatching
    inflight: GridLaunch | None      # currently dispatching grid (None = idle)
    completed: list[Result]          # one Result per finished launch

    def launch(self, ptx_src: str, grid: tuple, block: tuple, params: dict, *,
               kernel_name: str = "<unnamed>", config=None) -> None:
        """Append a GridLaunch to pending. Async — returns immediately."""
        ...

    def is_idle(self) -> bool:
        return self.inflight is None and not self.pending
```

Stream IDs are globally unique across the process lifetime (incremented on every `Stream()` call). The default implicit stream used by `gpusim.run(...)` has `stream_id = 0`.

### 3.2 New `KernelLaunch` trace event (`gpusim/trace/events.py`)

```python
@dataclass(frozen=True)
class KernelLaunch:
    stream_id: int
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    launch_cycle: int        # cycle when first CTA was dispatched
    complete_cycle: int      # cycle when last CTA retired
    n_ctas: int
```

### 3.3 `stream_id` field on existing events

The following 11 trace events all gain `stream_id: int = 0` (default 0 = implicit single stream, fully backward-compatible):

`InstrIssue`, `MemoryAccess`, `BarrierEvent`, `MmaEvent`, `BulkLoadEvent`, `BulkStoreEvent`, `ClusterDispatch`, `ClusterBarrier`, `CtaDispatch`, `L2MshrEvent`, `AtomicEvent`.

Recorder methods that record these events also gain `stream_id` keyword argument with default 0.

### 3.4 `Result.stream_id`

```python
@dataclass
class Result:
    ...
    stream_id: int = 0       # NEW Phase 7
```

---

## 4. Scheduler design

### 4.1 New class `MultiStreamScheduler` (`gpusim/core/scheduler.py`)

```python
class MultiStreamScheduler:
    """RR scheduler over multiple streams; intra-stream FIFO over grids."""

    def __init__(self, streams: list[Stream], policy: str = "rr"):
        self.streams = streams
        self.policy = policy
        self.cursor = 0                                 # RR pointer
        self._cta_iters: dict[int, CtaIterator] = {}    # per-stream current grid's CTA iter

    def next_dispatch(self, available_sms: list[SM]) -> tuple[Stream, CTA, SM] | None:
        """Try each stream in RR order; first one with a dispatchable CTA wins."""
        for _ in range(len(self.streams)):
            s = self.streams[self.cursor]
            self.cursor = (self.cursor + 1) % len(self.streams)
            cta = self._next_cta_for_stream(s)
            if cta is None:
                continue
            sm = self._find_sm_with_capacity(cta, available_sms)
            if sm is not None:
                return (s, cta, sm)
        return None

    def _next_cta_for_stream(self, s: Stream) -> CTA | None:
        """Return next CTA for stream's current in-flight grid.
           If current grid is fully dispatched and retired, pop next from pending."""
        ...
```

### 4.2 Stream-internal grid sequencing
- A grid's next CTA is only available after current grid has CTAs left to dispatch.
- A new grid does not begin dispatching until prior grid in same stream has **all CTAs retired** (occupancy contributing back to 0). This is verified each cycle.

### 4.3 `Device.run_streams` main loop (`gpusim/core/device.py`)

```python
def run_streams(streams: list[Stream], cfg) -> MultiStreamResult:
    sched = MultiStreamScheduler(streams)
    cycle = 0
    while not all(s.is_idle() and not s_in_flight(s) for s in streams):
        # 1. Dispatch up to N CTAs this cycle
        while True:
            choice = sched.next_dispatch(available_sms_list())
            if choice is None: break
            stream, cta, sm = choice
            sm.dispatch_cta(cta, stream_id=stream.stream_id)
        # 2. Tick everything
        for sm in device.sms: sm.tick(cycle)
        device.l2.tick(cycle)
        device.hbm.tick(cycle)
        cycle += 1
    return aggregate_per_stream_results(streams, recorder, cycle)
```

### 4.4 Backward compat: single-stream path
`Device.run(...)` (Phase 4 single-grid) is preserved unchanged. It internally constructs a single anonymous Stream and routes through `run_streams` with that one stream — but bypassing the multi-stream scheduler in favor of the existing single-grid scheduler when there is only 1 stream (perf preservation).

Alternatively, `run(...)` may simply construct an implicit Stream and invoke `run_streams`. The choice affects performance only; behavior is identical.

---

## 5. API

### 5.1 Single kernel (Phase 1-6, unchanged)

```python
res = gpusim.run(ptx_src=..., grid=..., block=..., params=..., mode="timing", config=cfg)
# res is Result with stream_id=0
```

### 5.2 Multi-stream (new)

```python
import gpusim
from gpusim.config.loader import load_default

cfg = load_default()
cfg.n_sm = 8

s1 = gpusim.Stream()                     # stream_id=0
s2 = gpusim.Stream()                     # stream_id=1
s1.launch(ptx_a, grid=(8,1,1), block=(32,1,1), params={"OUT": out_a},
          kernel_name="vec_add_a")
s2.launch(ptx_b, grid=(8,1,1), block=(32,1,1), params={"OUT": out_b},
          kernel_name="vec_add_b")

multi_res = gpusim.synchronize(config=cfg)   # MultiStreamResult
```

### 5.3 Same-stream multi-launch (pipeline-style)

```python
s = gpusim.Stream()
s.launch(ptx_load, ...)         # serial in same stream
s.launch(ptx_compute, ...)
s.launch(ptx_store, ...)
multi_res = gpusim.synchronize()
```

### 5.4 New `MultiStreamResult` class

```python
class MultiStreamResult:
    streams: dict[int, list[Result]]    # per-stream Results in launch order
    total_cycles: int                    # device-level: max stream end_cycle
    _recorder: Recorder                  # global recorder

    @property
    def kernel_launch_events_df(self) -> pd.DataFrame: ...
    @property
    def per_stream_events_df(self) -> dict[int, dict[str, pd.DataFrame]]: ...
    @property
    def stream_metrics(self) -> dict[int, dict]:
        """Per-stream: cycles, ctas, l2_share, hbm_share, dispatch_latency_avg."""
        ...

    def stream_summary(self) -> str:
        """Stream 0: vec_add_a, 8 CTAs, 1500 cycles
           Stream 1: vec_add_b, 8 CTAs, 1500 cycles
           Concurrency factor: 1.85×
           Compute/memory overlap: 67%"""
        ...

    def fairness(self) -> float:                  # Jain's index on CTA dispatch
    def overlap_ratio(self) -> float:             # compute vs memory overlap
```

### 5.5 `gpusim.synchronize` signature

```python
def synchronize(stream: Stream | None = None, *,
                config: Config | None = None) -> MultiStreamResult:
    """Drain all streams (or specified stream); return aggregated MultiStreamResult.

    If config not provided, uses each launch's config (which may be inherited
    from gpusim.load_default())."""
```

---

## 6. Trace + Analysis

### 6.1 Trace plumbing

- **`gpusim/trace/events.py`**: add `KernelLaunch` dataclass; add `stream_id: int = 0` field to 11 existing events.
- **`gpusim/trace/recorder.py`**: add `kernel_launch(...)` method; add `stream_id` kw param to all 11 record methods. Add `self.kernel_launch_events: list = []` to `__init__`.
- **`gpusim/trace/writer.py`**: add `kernel_launch.parquet` writer; existing parquet schemas auto-include the new column.

### 6.2 New analysis metrics (`gpusim/analysis/metrics.py`)

```python
def stream_concurrency_factor(kernel_launch_df, total_cycles: int) -> float:
    """Average number of streams active per cycle, over the device run.
       1.0 = serial; up to N for full overlap."""

def compute_memory_overlap(events_dfs: dict[str, pd.DataFrame]) -> float:
    """Fraction of compute-event cycles that overlap with memory-event cycles
       on different streams. High = good kernel co-locating."""

def l2_bandwidth_per_stream(memory_events_df) -> dict[int, float]:
    """Fraction of L2 requests originating from each stream."""

def stream_fairness_jain(cta_dispatch_df) -> float:
    """Jain's fairness index over per-stream CTA dispatch counts:
       (Σ x_i)² / (n · Σ x_i²)   where x_i = CTAs dispatched for stream i.
       1.0 = perfectly fair; 1/n = worst case."""
```

### 6.3 `Result` / `MultiStreamResult` extensions

```python
# Result (per-launch)
result.stream_id              # int

# MultiStreamResult
multi_res.kernel_launch_events_df    # pd.DataFrame
multi_res.stream_metrics              # dict[int, dict]
multi_res.stream_summary()            # str
multi_res.per_stream_events_df        # dict[int, dict[str, pd.DataFrame]]
multi_res.fairness()                  # float
multi_res.overlap_ratio()             # float
```

---

## 7. Viz

### 7.1 HTML report (`gpusim/viz/html_report.py` + `_template.html.j2`)

Two new sections:

- **§27 Stream concurrency timeline**
  - Gantt chart: per-stream row showing in-flight cycle range + KernelLaunch markers
  - Right-side table: 4 metrics (concurrency_factor / overlap / l2_bandwidth split / fairness)

- **§28 Per-stream resource breakdown**
  - Stacked bar: L2 requests, HBM requests, SM occupancy by stream_id
  - Annotates dominant kernel per stream

### 7.2 Perfetto JSON (`gpusim/viz/perfetto.py`)

- All existing events emitted by `build_perfetto` get `args.stream_id = ev.stream_id`.
- New `pid="Stream-0"`, `pid="Stream-1"`, ... swimlanes — each emits `KernelLaunch` events as `ph="X"` slices showing the kernel's lifetime.
- Stream swimlane background color rotates by `stream_id % 6` for visual distinction.

---

## 8. Examples (4)

### 8.1 `concurrent_vector_add_2stream/`
- Two streams each launch `vector_add` (grid=(8,1,1), block=(32,1,1)).
- Stream 0: A+B → C; Stream 1: D+E → F (independent gmem regions).
- **Verifies:** both outputs correct; cycles ≈ 1.2× single-vector_add (not 2×, proving overlap).
- Files: `kernel.ptx`, `reference.py`, `run.py`, `README.md`, `__init__.py`
- Parity test: `tests/parity/test_concurrent_vector_add_2stream.py`

### 8.2 `compute_vs_memory_overlap/` ⭐ Core teaching
- Stream 0: wgmma matmul (compute-bound, low HBM traffic).
- Stream 1: vector_add (memory-bound, high HBM traffic).
- **Verifies:** total cycles ≈ max(t_wgmma, t_vec_add) (not sum); `compute_memory_overlap` > 0.5.
- Demonstrates the real CUDA optimization: pair compute kernels with memory kernels.

### 8.3 `l2_contention_2stream/`
- Two streams simultaneously read/write **overlapping** gmem regions (same L2 line set).
- **Verifies:** cycles noticeably higher than separated-region version (real contention cost); `l2_bandwidth_per_stream` ≈ 50/50.

### 8.4 `stream_priority_serial_vs_concurrent/`
- Same total work (4 vec_add grids):
  - Config 1: all 4 in same stream (serial).
  - Config 2: 4 separate streams (concurrent).
- **Verifies:** `stream_concurrency_factor` differs significantly; concurrent cycles ≪ serial cycles.
- Note: no real "priority" yet (Phase 8); this is serial-vs-concurrent comparison.

---

## 9. Tutorials

`docs/tutorial/`, ~500-700 words each, English body + Chinese subheadings (`看模拟器` / `改一改` / `真机对照`):

- **27-multi-stream-concurrency-basics.md** — example 1
- **28-compute-memory-overlap.md** — example 2 ⭐ core
- **29-l2-hbm-contention-streams.md** — example 3
- **30-scheduler-fairness-streams.md** — example 4

Style matches chapters 22-26 (Phase 6) and 19-21 (Phase 5).

---

## 10. Testing strategy

### Unit tests (~12 new)
- `tests/unit/api/test_stream.py` — Stream construction, launch, pending queue, is_idle
- `tests/unit/core/test_multistream_scheduler.py` — RR fairness; intra-stream grid sequencing; cross-stream CTA interleaving on a 2-SM device with 2 streams; corner case: one stream idle
- `tests/unit/trace/test_kernel_launch_event.py` — recorder + parquet writer
- `tests/unit/trace/test_per_event_stream_id.py` — all 11 events correctly carry stream_id when set
- `tests/unit/analysis/test_phase7_metrics.py` — 4 metrics, each with 1+ test

### Parity tests (~4)
- One per example: correctness + cycles range assertion

### Microbench
- `tests/microbench/test_phase7_facts.py` (fast): concurrent vec_add ≥ 1.5× faster than serial vec_add (independent work, same total)
- `tests/microbench/test_phase7_runtime.py` (slow): each example runs under 60s

### Regression
- Rename `tests/parity/test_phase1_5_examples_unchanged.py` → `test_phase1_6_examples_unchanged.py`
- Add 5 Phase 6 examples to the regression list

### Test count target
438 (Phase 6 baseline) → ~458 (+20).

---

## 11. Milestones

| Milestone | Scope | Tag |
|---|---|---|
| **M1** Trace plumbing + Stream/launch API | KernelLaunch event + stream_id propagation to 11 events + Stream class + Result.stream_id | `M1-phase7-complete` |
| **M2** MultiStreamScheduler + Device.run_streams | RR scheduler + grid sequencing + CTA dispatch tagging + 1 demo: concurrent_vector_add_2stream | `M2-phase7-complete` |
| **M3** Compute/memory overlap example + 4 metrics | compute_vs_memory_overlap example + 4 metrics + MultiStreamResult API | `M3-phase7-complete` |
| **M4** Contention + fairness examples | l2_contention_2stream + stream_priority_serial_vs_concurrent | `M4-phase7-complete` |
| **M5** Viz + docs + ship | HTML §27/§28 + Perfetto stream swimlane + 4 tutorials + microbench + Phase 1-6 regression rename + README v7 + final tag | `phase7-complete` |

Estimated 25-28 tasks total.

---

## 12. File list

### New files
```
gpusim/core/scheduler.py                      # extend: MultiStreamScheduler
gpusim/api.py                                 # extend: Stream, MultiStreamResult, synchronize
examples/concurrent_vector_add_2stream/       # 5 files
examples/compute_vs_memory_overlap/           # 5 files
examples/l2_contention_2stream/               # 5 files
examples/stream_priority_serial_vs_concurrent/ # 5 files
docs/tutorial/27-multi-stream-concurrency-basics.md
docs/tutorial/28-compute-memory-overlap.md
docs/tutorial/29-l2-hbm-contention-streams.md
docs/tutorial/30-scheduler-fairness-streams.md
tests/unit/api/test_stream.py
tests/unit/core/test_multistream_scheduler.py
tests/unit/trace/test_kernel_launch_event.py
tests/unit/trace/test_per_event_stream_id.py
tests/unit/analysis/test_phase7_metrics.py
tests/parity/test_concurrent_vector_add_2stream.py
tests/parity/test_compute_vs_memory_overlap.py
tests/parity/test_l2_contention_2stream.py
tests/parity/test_stream_priority_serial_vs_concurrent.py
tests/microbench/test_phase7_facts.py
tests/microbench/test_phase7_runtime.py
tests/reference/data/{4 example names}.ref.json
```

### Modified files
```
gpusim/trace/events.py                        # +KernelLaunch + stream_id field on 11 events
gpusim/trace/recorder.py                      # +kernel_launch method + stream_id kwarg on 11 methods
gpusim/trace/writer.py                        # +kernel_launch.parquet
gpusim/core/device.py                         # +run_streams + tag CTA dispatch with stream_id
gpusim/core/sm.py                             # propagate stream_id from CTA to events
gpusim/core/sub_core.py                       # propagate stream_id from CTA to events
gpusim/api.py                                 # Result.stream_id; new Stream / MultiStreamResult / synchronize
gpusim/analysis/metrics.py                    # +4 metrics
gpusim/viz/notebook.py                        # +kernel_launch_events_dataframe + per_stream_events
gpusim/viz/html_report.py                     # +§27/§28 render helpers
gpusim/viz/_template.html.j2                  # +§27/§28 blocks
gpusim/viz/perfetto.py                        # +Stream swimlanes + stream_id args
tests/parity/test_phase1_5_examples_unchanged.py → test_phase1_6_examples_unchanged.py
tests/reference/gen_reference.py              # +4 kernel names
README.md                                     # v7 — Phase 7 capabilities
```

---

## 13. Backward compatibility

- `gpusim.run(...)` — **unchanged** behavior. Returns `Result` with `stream_id=0`.
- `Result` — gains `stream_id: int = 0` field; existing access patterns unchanged.
- All 11 trace events — new `stream_id: int = 0` field appended; default 0 means "not set" / "single-stream mode".
- All recorder methods — `stream_id` keyword argument with default 0.
- Existing examples (vector_add, reduction_smem, etc.) — no changes; they continue to use the single-kernel path.
- Existing tests — no changes required. Phase 1-6 regression test (renamed) runs unchanged.

---

## 14. Open questions / future work

- **Stream priorities** — Phase 8: `Stream(priority="high"|"normal"|"low")`. Scheduler weights RR slots by priority.
- **Cross-stream events** — Phase 8: `cudaEvent`-equivalent with record/wait semantics.
- **CUDA Graphs** — Phase 9: pre-recorded DAG of kernels.
- **L2 partitioning** — Phase 8: per-stream L2 set affinity.
- **Persistent kernels** — Phase 9: `Stream.persistent_launch(...)` — kernel waits for work in a queue rather than retiring.
- **Dynamic parallelism** — Phase 10: kernel launches kernel.
- **Sub-CTA dispatch fairness** — current scheduler is CTA-granular; could go finer (warp-level).

---

## 15. Acceptance criteria

Phase 7 ships when:

- [ ] All 5 milestone tags present (`M1-phase7-complete` ... `M4-phase7-complete`, `phase7-complete`)
- [ ] All 4 examples run cleanly (`python examples/<name>/run.py`)
- [ ] All 4 parity tests pass
- [ ] Microbench `test_phase7_facts.py::test_concurrent_faster_than_serial` passes (concurrent ≥ 1.5× serial)
- [ ] `MultiStreamResult.stream_summary()` produces meaningful output for each example
- [ ] HTML report shows §27 + §28 when multi-stream events present
- [ ] Perfetto JSON has per-stream swimlane
- [ ] Phase 1-6 regression test (renamed) passes: all prior examples unchanged
- [ ] Test count: 438 → ~458 (+20)
- [ ] README v7 documents Phase 7 capabilities
