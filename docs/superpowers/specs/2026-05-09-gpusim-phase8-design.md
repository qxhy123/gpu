# gpusim Phase 8 — True Concurrent Scheduler + Stream Priority + Events + L2 Partitioning

> **Status:** Brainstormed 2026-05-09. Implementation TBD.

## 1. Goals & Non-goals

### Goals
- **Replace Phase 7's sequential drain with true per-cycle interleaved CTA dispatch**, fixing the core limitation that `compute_vs_memory_overlap` could not actually save cycles.
- Add **stream priority** (`high` / `normal` / `low`) with weighted round-robin scheduling (4:2:1 default tokens).
- Add **CUDA events** (`Event.record(stream)` + `Stream.wait(event)`) for cross-stream sync; record + wait state machine; event-blocked stream awareness in scheduler.
- Add **L2 cache set-window partitioning** per stream (mimicking H100 `cudaStreamAttributeAccessPolicyWindow`) — protects high-priority stream's hot data from eviction by background streams.
- Ship 6 examples + 6 tutorial chapters demonstrating each feature in isolation, plus a final `multi_stream_pipeline_full` integration demo.
- Full trace (1 new `StreamEvent` event + per-line owner_stream_id on L2 lines) + 6 new analysis metrics + 3 HTML sections + Perfetto priority/event annotations.
- 100% backward compatible: Phase 1-7 examples and tests pass unchanged. Single-kernel `gpusim.run()` unchanged. Multi-stream API from Phase 7 (`Stream.launch`, `gpusim.synchronize`) unchanged at signature level — internal scheduler is the only change.

### Non-goals (deferred to Phase 9+)
- **CUDA Graphs** — pre-recorded DAG of kernels with explicit topological dependencies.
- **Persistent kernels** — long-running kernels that pop work from a queue.
- **Numeric stream priorities** (`cudaStreamCreateWithPriority(stream, flags, priority)` with -5..0 range). Phase 8 uses string `high|normal|low` only.
- **Dynamic parallelism** (kernel launches kernel).
- **Stream callbacks** (`cudaStreamAddCallback`).
- **Multi-GPU / NVLink / NCCL**.
- **Multi-device events** (events spanning multiple GPUs).
- **L2 way-isolation** (alternative partitioning mode); Phase 8 uses set-window only.

---

## 2. Architecture

```
Phase 7 baseline:                       Phase 8:
┌───────────────────┐                   ┌─────────────────────────────────┐
│ Stream(0/1/...)   │                   │ Stream(priority="high|normal|low")│
│ launch → pending  │                   │ launch → pending + event_blocked  │
└─────────┬─────────┘                   └─────────────┬───────────────────┘
          ▼                                           ▼
┌───────────────────┐                   ┌──────────────────────────────────┐
│ run_streams       │                   │ ConcurrentStreamScheduler:       │
│ sequential drain  │                   │  per-cycle weighted RR           │
│ (1 launch at time)│        →          │  + event-block check             │
└─────────┬─────────┘                   │  + priority-weighted slots       │
          ▼                             └──────────┬───────────────────────┘
                                                   ▼
                                        ┌──────────────────────────────────┐
                                        │ N SMs share L2 + HBM             │
                                        │  L2 with set-window partitioning │
                                        │  per-stream cache eviction policy│
                                        └──────────────────────────────────┘
```

### Key invariants
- **Per-cycle dispatch**: Each cycle, scheduler can dispatch CTAs from multiple streams simultaneously. SM occupancy permitting, several streams may have CTAs running on the same SM at the same cycle.
- **Weighted RR by priority**: `high=4`, `normal=2`, `low=1` token slots per cycle (configurable). High-priority streams get more dispatch attempts per cycle.
- **Event-block state**: A stream waiting on an unsignaled event is skipped in the scheduler's per-cycle pass. Once event signals, stream resumes next cycle.
- **L2 set-window protection**: A stream may register a `(start_set, n_sets)` window. Lines in those sets owned by the registering stream are evict-protected from other streams.
- **Phase 7 sequential drain is removed**, not kept as fallback. Single-stream behavior is preserved by ConcurrentStreamScheduler degenerating to single-stream behavior with no priority/event/window features used.
- **Phase 1-7 backward compat**: All existing tests pass. `gpusim.run()` and Phase 7 multi-stream API signatures unchanged.

---

## 3. Data model

### 3.1 Stream extensions (`gpusim/api.py`)

```python
@dataclass
class Stream:
    stream_id: int = field(default_factory=_next_stream_id)
    pending: deque = field(default_factory=deque)
    inflight: GridLaunch | None = None
    completed: list = field(default_factory=list)
    
    # NEW Phase 8
    priority: str = "normal"                            # "high" | "normal" | "low"
    event_waits: list = field(default_factory=list)     # Events this stream is waiting on
    in_flight_ctas: int = 0                              # Count of CTAs not yet retired
    l2_window: tuple | None = None                       # (start_set, n_sets) — for L2 partitioning

    def __post_init__(self):
        if self.priority not in ("high", "normal", "low"):
            raise ValueError(f"priority must be high/normal/low, got {self.priority!r}")

    def record(self, ev: "Event") -> None:
        """Append a record-marker to pending; signals ev when prior pending+inflight retire."""
        self.pending.append(_RecordMarker(event=ev))

    def wait(self, ev: "Event") -> None:
        """Block this stream's future launches until ev is signaled."""
        self.event_waits.append(ev)

    def set_l2_window(self, *, start_set: int, n_sets: int) -> None:
        """Reserve L2 sets [start_set, start_set+n_sets) as evict-protected window."""
        self.l2_window = (start_set, n_sets)
```

### 3.2 New `gpusim.Event` class

```python
_EVENT_ID_COUNTER = 0


def _next_event_id() -> int:
    global _EVENT_ID_COUNTER
    eid = _EVENT_ID_COUNTER
    _EVENT_ID_COUNTER += 1
    return eid


@dataclass
class Event:
    event_id: int = field(default_factory=_next_event_id)
    
    recorded_in_stream: Stream | None = None
    record_cycle: int | None = None         # cycle when scheduler reaches the record point
    signaled_at_cycle: int | None = None    # cycle when all prior CTAs in recorded_in_stream retired
    
    def is_signaled(self, current_cycle: int) -> bool:
        return (self.signaled_at_cycle is not None 
                and self.signaled_at_cycle <= current_cycle)
```

### 3.3 New `_RecordMarker` (internal sentinel)

A zero-CTA pseudo-grid that scheduler treats specially:
```python
@dataclass
class _RecordMarker:
    event: Event
    grid: tuple = (0, 0, 0)   # zero-CTA so iter terminates immediately
```

When `_ensure_inflight(s)` pops a `_RecordMarker`, it sets `event.recorded_in_stream = s` and `event.record_cycle = current_cycle`, then immediately marks the marker as "retired" (no CTAs to dispatch).

### 3.4 New `StreamEvent` trace event (`gpusim/trace/events.py`)

```python
@dataclass(frozen=True)
class StreamEvent:
    cycle: int
    event_id: int
    stream_id: int
    op: str               # "record" | "wait_start" | "wait_satisfied"
```

3 ops:
- `record`: when scheduler processes a record-marker (event has its `record_cycle` set).
- `wait_start`: when stream first becomes event-blocked (entered `wait()` and event not yet signaled).
- `wait_satisfied`: when blocked stream unblocks (event becomes signaled).

### 3.5 L2Line extensions (`gpusim/core/cache/l2.py`)

```python
@dataclass
class L2Line:
    addr: int
    valid: bool
    dirty: bool
    last_use: int
    
    # NEW Phase 8
    owner_stream_id: int = -1     # which stream installed this line
    in_window: bool = False        # if owner has a set window claiming this set
```

---

## 4. True concurrent scheduler

### 4.1 New class `ConcurrentStreamScheduler` (`gpusim/core/scheduler.py`)

Replaces Phase 7's `MultiStreamScheduler`. Phase 7's class can be deleted, or kept as a deprecated alias.

```python
class ConcurrentStreamScheduler:
    """Per-cycle weighted RR over multiple streams, with event-block awareness.
    
    Each cycle:
      1. For each stream in priority-weighted RR order:
         - Skip if event-blocked (waiting on an unsignaled event)
         - Skip if no CTAs to dispatch
         - Try to dispatch up to its priority weight (high=4, normal=2, low=1)
      2. Tick all SMs + L2 + HBM
      3. Check for grid retire → release event blocks
    """
    
    def __init__(self, streams: list, priority_weights: dict | None = None):
        self.streams = streams
        self.cursor = 0
        self._cta_iters: dict[int, _CtaIter] = {}
        self._priority_weights = priority_weights or {"high": 4, "normal": 2, "low": 1}
    
    def stream_weight(self, s: Stream) -> int:
        return self._priority_weights.get(s.priority, 2)
    
    def is_event_blocked(self, s: Stream, current_cycle: int) -> bool:
        for ev in s.event_waits:
            if not ev.is_signaled(current_cycle): return True
        return False
    
    def _ensure_inflight(self, s):
        if s.inflight is None and s.pending:
            head = s.pending.popleft()
            if isinstance(head, _RecordMarker):
                # Process record marker: signal event when prior CTAs retire
                head.event.recorded_in_stream = s
                # signaled_at_cycle set later by retire-check
                return False
            s.inflight = head
            self._cta_iters[s.stream_id] = _CtaIter(head.grid)
            s.in_flight_ctas = head.grid[0] * head.grid[1] * head.grid[2]
        return s.inflight is not None
    
    def step(self, available_sms, current_cycle: int) -> list[tuple]:
        """Returns list of (stream, cta, sm) dispatch decisions for this cycle."""
        decisions = []
        for s in self.streams:
            if s.is_idle() and not s.in_flight_ctas: continue
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

### 4.2 New `Device.run_streams` main loop

```python
def run_streams(self, streams: list, *, events: list = None) -> MultiStreamResult:
    sched = ConcurrentStreamScheduler(streams)
    cycle = 0
    while not all(s.is_idle() and s.in_flight_ctas == 0 for s in streams):
        # 1. Per-cycle dispatch (weighted RR + event-block check)
        decisions = sched.step(self.available_sms(), cycle)
        for stream, cta, sm in decisions:
            sm.dispatch_cta(cta, stream_id=stream.stream_id)
        
        # 2. Tick everything ONE cycle
        for sm in self.sms: sm.tick(cycle)
        self.l2.tick(cycle)
        self.hbm.tick(cycle)
        
        # 3. Check grid retire + event signaling
        for s in streams:
            if s.in_flight_ctas == 0 and s.inflight is not None:
                self._signal_kernel_complete(s, cycle)
                sched.mark_grid_retired(s)
            # Check pending record-markers: if all prior CTAs retired, signal event
            self._check_event_signals(s, cycle)
        
        cycle += 1
    return self._build_result(streams, sched, cycle)
```

### 4.3 Event signaling logic

```python
def _check_event_signals(self, s: Stream, current_cycle: int) -> None:
    """If stream is currently idle and the head of pending is a _RecordMarker,
    pop it and signal the event."""
    while (s.inflight is None and s.pending 
            and isinstance(s.pending[0], _RecordMarker)
            and s.in_flight_ctas == 0):
        marker = s.pending.popleft()
        marker.event.signaled_at_cycle = current_cycle
        if self._recorder:
            self._recorder.stream_event(cycle=current_cycle,
                                          event_id=marker.event.event_id,
                                          stream_id=s.stream_id, op="record")
```

---

## 5. Priority API + weighted RR

### 5.1 Stream priority field (already covered in §3.1)

### 5.2 Weighted RR allocation

Each cycle, scheduler iterates `streams` in insertion order; each stream gets up to `_priority_weights[s.priority]` dispatch tokens. Default: `{"high": 4, "normal": 2, "low": 1}`. Configurable via `cfg.scheduler.priority_weights`.

If SM occupancy exhausts before tokens used → tokens lost (do not roll over).

### 5.3 Configuration extension (`gpusim/config/schema.py`)

```python
@dataclass
class SchedulerConfig:
    cta_scheduler: str = "rr"                    # existing Phase 4
    priority_weights: dict = field(default_factory=lambda: {"high": 4, "normal": 2, "low": 1})
```

`cfg.scheduler.priority_weights["high"] = 8` overrides for stronger preference.

### 5.4 Result API (`gpusim/api.py::MultiStreamResult`)

```python
def priority_dispatch_share(self) -> dict:
    """Fraction of CTA dispatches per priority level."""
    if self._recorder is None: return {}
    counts = {"high": 0, "normal": 0, "low": 0}
    stream_priority = {s.stream_id: getattr(s, "priority", "normal")
                         for sid, results in self.streams.items()
                         for s in self._streams_for_id(sid)}
    for ev in getattr(self._recorder, "cta_dispatch_events", []):
        p = stream_priority.get(ev.stream_id, "normal")
        counts[p] += 1
    total = max(sum(counts.values()), 1)
    return {p: c / total for p, c in counts.items()}
```

(Internal `_streams_for_id` looks up the actual Stream object retained for priority access.)

### 5.5 API example

```python
s_high = gpusim.Stream(priority="high")
s_low = gpusim.Stream(priority="low")
s_high.launch(ptx, ..., kernel_name="critical")
s_low.launch(ptx, ..., kernel_name="background")
res = gpusim.synchronize(streams=[s_high, s_low], config=cfg)
print(res.priority_dispatch_share())
# Expected: {"high": ~0.8, "normal": 0.0, "low": ~0.2}  (4 vs 1 weight)
```

---

## 6. Events API

### 6.1 Stream.record / wait (covered in §3.1)

### 6.2 Event lifecycle

1. `Event()` created → `event_id` assigned; `recorded_in_stream`, `record_cycle`, `signaled_at_cycle` all None.
2. `Stream.record(ev)` → appends `_RecordMarker(event=ev)` to that stream's pending queue.
3. Scheduler processes record-marker → sets `event.record_cycle = current_cycle`. Marker dropped (no CTAs).
4. When all prior in-flight CTAs in `recorded_in_stream` retire → `event.signaled_at_cycle = current_cycle` (in `_check_event_signals`).
5. `Stream.wait(ev)` adds to `event_waits[]`. Scheduler skips this stream while `not ev.is_signaled(current_cycle)`.
6. Once event signals, dependent streams unblock and resume next cycle.

### 6.3 Result API

```python
def event_wait_cycles_per_stream(self) -> dict[int, int]:
    """Total cycles each stream spent event-blocked."""
    if self._recorder is None: return {}
    waits = getattr(self._recorder, "stream_event_events", [])
    out = {}
    pending_starts = {}     # (stream_id, event_id) -> wait_start_cycle
    for ev in sorted(waits, key=lambda x: x.cycle):
        key = (ev.stream_id, ev.event_id)
        if ev.op == "wait_start":
            pending_starts[key] = ev.cycle
        elif ev.op == "wait_satisfied" and key in pending_starts:
            cycles = ev.cycle - pending_starts.pop(key)
            out[ev.stream_id] = out.get(ev.stream_id, 0) + cycles
    return out


def event_chain_critical_path(self) -> int:
    """Longest event-mediated dependency chain (in cycles)."""
    # ... DAG analysis on stream_event_events + kernel_launch_events ...
```

### 6.4 API example

```python
ev = gpusim.Event()
s_a = gpusim.Stream()
s_b = gpusim.Stream()

s_a.launch(ptx_write, ...)
s_a.record(ev)
s_b.wait(ev)
s_b.launch(ptx_read, ...)

res = gpusim.synchronize(streams=[s_a, s_b], config=cfg)
print(res.event_wait_cycles_per_stream())   # {1: 250} — s_b waited 250 cycles
```

---

## 7. L2 partitioning (set-window)

### 7.1 Stream.set_l2_window (covered in §3.1)

### 7.2 L2Line extensions (covered in §3.5)

### 7.3 L2Cache changes (`gpusim/core/cache/l2.py`)

```python
class L2Cache:
    def __init__(self, cfg, hbm):
        # ... existing init ...
        # NEW Phase 8
        self._stream_windows: dict[int, tuple] = {}    # stream_id -> (start_set, n_sets)
    
    def register_stream_window(self, stream_id: int, start_set: int, n_sets: int) -> None:
        """Reserve [start_set, start_set+n_sets) as protected window for this stream."""
        self._stream_windows[stream_id] = (start_set, n_sets)
    
    def _line_in_window(self, line: L2Line, set_idx: int) -> bool:
        if line.owner_stream_id < 0: return False
        window = self._stream_windows.get(line.owner_stream_id)
        if window is None: return False
        start, n = window
        return start <= set_idx < start + n
    
    def _pick_victim(self, set_idx: int, requesting_stream_id: int) -> L2Line | None:
        """LRU with window protection.
        Among lines in this set, exclude lines that are window-protected by
        other streams. Pick LRU among remaining."""
        candidates = []
        for line in self.sets[set_idx]:
            if (self._line_in_window(line, set_idx) 
                    and line.owner_stream_id != requesting_stream_id):
                continue   # protected — skip
            candidates.append(line)
        if not candidates:
            return None     # all lines protected → no install possible (caller treats as miss-only)
        return min(candidates, key=lambda c: c.last_use)
    
    def _install_line(self, addr: int, set_idx: int, requesting_stream_id: int) -> None:
        victim = self._pick_victim(set_idx, requesting_stream_id)
        if victim is None: return    # cannot install (all protected)
        # ... write back if dirty ...
        victim.addr = addr
        victim.valid = True
        victim.last_use = self._now
        victim.owner_stream_id = requesting_stream_id
        victim.in_window = self._line_in_window(victim, set_idx)
```

### 7.4 Device wires Stream.l2_window into L2

```python
# In Device.run_streams, before the main loop:
for s in streams:
    if s.l2_window is not None:
        self.l2.register_stream_window(s.stream_id, *s.l2_window)
```

### 7.5 API example

```python
s_critical = gpusim.Stream(priority="high")
s_critical.set_l2_window(start_set=0, n_sets=32)
s_bg = gpusim.Stream(priority="low")     # no window

s_critical.launch(ptx_loop_hot_tile, ..., kernel_name="critical")
s_bg.launch(ptx_streaming_huge_array, ..., kernel_name="background")
res = gpusim.synchronize(streams=[s_critical, s_bg], config=cfg)
print(res.l2_window_hit_rate())
# Expected: {0: 0.95, 1: 0.10}  (s_critical's hot tile preserved)
```

---

## 8. Trace + Analysis

### 8.1 New trace event `StreamEvent` (covered in §3.4)

### 8.2 Recorder methods

```python
class Recorder:
    def __init__(self):
        # ... existing ...
        self.stream_event_events: list = []     # NEW Phase 8
    
    def stream_event(self, *, cycle: int, event_id: int, stream_id: int, op: str) -> None:
        from gpusim.trace.events import StreamEvent
        self.stream_event_events.append(StreamEvent(
            cycle=cycle, event_id=event_id, stream_id=stream_id, op=op,
        ))
```

### 8.3 Writer adds `stream_event.parquet`

### 8.4 6 new analysis metrics (`gpusim/analysis/metrics.py`)

```python
def priority_dispatch_share(cta_dispatch_df, stream_priority_lookup) -> dict:
    """Fraction of CTA dispatches per priority level (high/normal/low)."""

def event_wait_cycles_per_stream(stream_event_df) -> dict:
    """Total cycles each stream spent event-blocked.
    Pairs wait_start with wait_satisfied per (stream_id, event_id)."""

def l2_window_hit_rate_per_stream(memory_events_df, stream_window_config) -> dict:
    """L2 hit rate per stream, separated by stream_id.
    For windowed streams: rate within their window."""

def l2_window_protection_efficiency(memory_events_df, l2_line_owner_df) -> float:
    """Fraction of L2 line hits that came from window-protected lines.
    High = window is providing real benefit."""

def cross_stream_concurrency_gain(kernel_launch_df, total_cycles) -> float:
    """Speedup over sequential drain baseline.
    Computed as: sum(per-launch cycles) / total_cycles.
    1.0 = no overlap; > 1.0 = concurrent benefit."""

def event_chain_critical_path(stream_event_df, kernel_launch_df) -> int:
    """Longest event-mediated dependency chain in cycles.
    DAG analysis: trace launches that depend on events that depend on prior launches."""
```

### 8.5 Result + MultiStreamResult extensions

```python
class MultiStreamResult:
    # ... existing ...
    
    @property
    def stream_event_events_df(self): ...
    
    def priority_dispatch_share(self) -> dict: ...
    def event_wait_cycles_per_stream(self) -> dict: ...
    def l2_window_hit_rate(self) -> dict: ...
    def l2_window_protection_efficiency(self) -> float: ...
    def cross_stream_concurrency_gain(self) -> float: ...
    def event_chain_critical_path(self) -> int: ...
```

---

## 9. Viz

### 9.1 HTML new sections (`gpusim/viz/html_report.py` + `_template.html.j2`)

- **§29 Priority dispatch breakdown** — bar chart showing CTAs dispatched by priority class; shows `priority_dispatch_share()`.
- **§30 Event timeline** — gantt of events: per-stream rows showing record/wait_start/wait_satisfied markers; arrows linking record → wait pairs.
- **§31 L2 window heatmap** — N×M grid (sets × time-buckets); cell color = dominant `owner_stream_id`. Visualizes per-set ownership over time.

### 9.2 Perfetto annotations (`gpusim/viz/perfetto.py`)

- All KernelLaunch events: `args.priority = stream.priority` (color-code by priority level).
- New `StreamEvent` events emitted as Perfetto instant events (`ph: "i"`):
  ```python
  {"name": f"event_{op}_{event_id}", "cat": "stream_event",
   "ph": "i", "ts": cycle, "pid": f"Stream-{stream_id}",
   "tid": "events", "s": "g", "args": {...}}
  ```
- Optional: emit Perfetto async events (ph: "s" + "f") to draw record→wait arrows.

---

## 10. Examples (6)

### 10.1 `true_concurrent_overlap/`
- Compute-heavy stream + memory-heavy stream — same as Phase 7's compute_vs_memory_overlap, but on Phase 8 scheduler.
- **Verifies**: `total_cycles ≤ max(compute, memory) × 1.2` (proving real overlap, vs Phase 7 sequential drain).
- Files: 6 (kernel_compute.ptx, kernel_memory.ptx, reference.py, run.py, README.md, __init__.py).
- Parity test: cycles range + correctness + `multi_res.cross_stream_concurrency_gain() > 1.3`.

### 10.2 `priority_demo/`
- 3 streams (high, normal, low), each with 10 vec_add launches.
- **Verifies**: high stream finishes ~3× faster than low; `priority_dispatch_share() ≈ {"high": 0.57, "normal": 0.29, "low": 0.14}`.

### 10.3 `event_producer_consumer/`
- Stream A writes X → record(ev) → Stream B wait(ev) → reads X.
- **Verifies**: data dependency satisfied; B sees A's writes; `event_wait_cycles_per_stream` shows B waited.

### 10.4 `event_fanout/`
- Stream A → record(ev) → Streams B, C, D each wait(ev) → all read.
- **Verifies**: 1 event satisfies multiple consumers; all 3 consumers wait for the single producer.

### 10.5 `l2_window_demo/`
- High stream + L2 window on hot tile; low stream streaming through huge array.
- **Verifies**: high stream's `l2_window_hit_rate ≥ 0.8`; low stream's hit rate unaffected by high's window.

### 10.6 `multi_stream_pipeline_full/` ⭐ Capstone
- 3 streams: load (memory) → compute (wgmma, high priority + L2 window) → store (memory).
- Event chain: load → ev1 → compute waits ev1 → ev2 → store waits ev2.
- **Verifies**: end-to-end correctness; `event_chain_critical_path` matches expected depth; combined feature interactions.

Each example: 5 files (kernel.ptx + reference.py + run.py + README.md + __init__.py) plus a parity test in `tests/parity/`.

---

## 11. Tutorials

`docs/tutorial/`, ~500-700 words each, English body + Chinese subheadings (`看模拟器` / `改一改` / `真机对照`):

- **31-true-concurrent-scheduler.md** — example 1
- **32-stream-priority-weighted-rr.md** — example 2
- **33-cuda-events-record-wait.md** — example 3
- **34-event-fanout-pattern.md** — example 4
- **35-l2-cache-window-partitioning.md** — example 5 ⭐ core
- **36-production-multi-stream-pipeline.md** — example 6 ⭐ capstone

Match style of chapters 27-30 (Phase 7).

---

## 12. Testing strategy

### Unit tests (~18 new)
- `tests/unit/api/test_stream_priority.py` — Stream(priority=...) validation; weight lookup
- `tests/unit/api/test_event.py` — Event lifecycle; record/wait state
- `tests/unit/core/test_concurrent_scheduler.py` — per-cycle interleave; weighted RR; event-block; replaces test_multistream_scheduler
- `tests/unit/cache/test_l2_window.py` — set window registration; eviction protection; cross-stream isolation
- `tests/unit/trace/test_stream_event.py` — StreamEvent recorder + parquet
- `tests/unit/analysis/test_phase8_metrics.py` — 6 new metrics

### Parity tests (~6) — one per example

### Microbench
- `tests/microbench/test_phase8_facts.py` (fast):
  - True concurrent: 2-stream overlap ≤ 0.8× sequential drain baseline (concurrency benefit)
  - Priority: high finishes ≥ 2× faster than low
  - L2 window: windowed stream hit rate ≥ 1.3× unwindowed baseline
- `tests/microbench/test_phase8_runtime.py` (slow): 6 examples each under 60s

### Regression
- Rename `tests/parity/test_phase1_6_examples_unchanged.py` → `test_phase1_7_examples_unchanged.py`
- Add 4 Phase 7 examples to the regression list

### Test count target
473 (Phase 7 baseline) → ~510 (+37).

---

## 13. Milestones

| Milestone | Scope | Tag |
|---|---|---|
| **M1** True concurrent scheduler + true_concurrent_overlap | ConcurrentStreamScheduler + Device.run_streams rewrite + Stream.in_flight_ctas + 1 example | `M1-phase8-complete` |
| **M2** Stream priority + priority_demo | Stream.priority + weighted RR + priority_dispatch_share + 1 example | `M2-phase8-complete` |
| **M3** Events + 2 examples | Event class + Stream.record/wait + StreamEvent trace + event metrics + event_producer_consumer + event_fanout | `M3-phase8-complete` |
| **M4** L2 partitioning + l2_window_demo | L2Line.owner_stream_id + window registration + eviction policy + 2 metrics + 1 example | `M4-phase8-complete` |
| **M5** Pipeline + viz + docs + ship | multi_stream_pipeline_full + 6 chapters + HTML §29-§31 + Perfetto priority/event annotations + microbench + Phase 1-7 regression rename + README v8 | `phase8-complete` |

Estimated 36 tasks total.

---

## 14. File list

### New files
```
examples/true_concurrent_overlap/                # 6 files (compute + memory ptx)
examples/priority_demo/                          # 5 files
examples/event_producer_consumer/                # 5 files
examples/event_fanout/                           # 5 files
examples/l2_window_demo/                         # 5 files
examples/multi_stream_pipeline_full/             # 6+ files (3 ptx kernels)
docs/tutorial/31-true-concurrent-scheduler.md
docs/tutorial/32-stream-priority-weighted-rr.md
docs/tutorial/33-cuda-events-record-wait.md
docs/tutorial/34-event-fanout-pattern.md
docs/tutorial/35-l2-cache-window-partitioning.md
docs/tutorial/36-production-multi-stream-pipeline.md
tests/unit/api/test_stream_priority.py
tests/unit/api/test_event.py
tests/unit/core/test_concurrent_scheduler.py    # replaces test_multistream_scheduler.py (rename)
tests/unit/cache/test_l2_window.py
tests/unit/trace/test_stream_event.py
tests/unit/analysis/test_phase8_metrics.py
tests/parity/test_true_concurrent_overlap.py
tests/parity/test_priority_demo.py
tests/parity/test_event_producer_consumer.py
tests/parity/test_event_fanout.py
tests/parity/test_l2_window_demo.py
tests/parity/test_multi_stream_pipeline_full.py
tests/microbench/test_phase8_facts.py
tests/microbench/test_phase8_runtime.py
tests/reference/data/{6 example names}.ref.json
```

### Modified files
```
gpusim/api.py                                   # Stream priority+event_waits+l2_window+record/wait/set_l2_window;
                                                # Event class; _RecordMarker; MultiStreamResult new metrics
gpusim/core/scheduler.py                        # ConcurrentStreamScheduler replaces MultiStreamScheduler
gpusim/core/device.py                           # run_streams rewrite (per-cycle main loop)
gpusim/core/cache/l2.py                         # L2Line owner_stream_id+in_window; register_stream_window;
                                                # _pick_victim with window protection
gpusim/config/schema.py                         # SchedulerConfig.priority_weights
gpusim/trace/events.py                          # +StreamEvent
gpusim/trace/recorder.py                        # +stream_event method
gpusim/trace/writer.py                          # +stream_event.parquet
gpusim/analysis/metrics.py                      # +6 metrics
gpusim/viz/notebook.py                          # +stream_event_events_dataframe
gpusim/viz/html_report.py                       # +§29/§30/§31 render helpers
gpusim/viz/_template.html.j2                    # +§29/§30/§31 blocks
gpusim/viz/perfetto.py                          # +priority args + StreamEvent emission + record→wait arrows
tests/parity/test_phase1_6_examples_unchanged.py → test_phase1_7_examples_unchanged.py
tests/reference/gen_reference.py                # +6 kernel names
README.md                                       # v8 — Phase 8 capabilities
```

---

## 15. Backward compatibility

- `gpusim.run(...)` — unchanged behavior. Returns `Result` with `stream_id=0`.
- `gpusim.synchronize(streams)` — unchanged signature; internal scheduler change is invisible.
- Phase 7 examples (concurrent_vector_add_2stream, compute_vs_memory_overlap, l2_contention_2stream, stream_priority_serial_vs_concurrent) — pass unchanged.
- Phase 1-6 examples — pass unchanged (single-kernel path).
- Existing tests — no signature changes required.
- Phase 7 `MultiStreamScheduler` class — replaced by `ConcurrentStreamScheduler`. The Phase 7 class is removed since its only user is `Device.run_streams` which is rewritten. If any tests reference `MultiStreamScheduler` directly, those tests are updated to use `ConcurrentStreamScheduler`.

---

## 16. Open questions / future work

- **Numeric stream priorities** — Phase 9: `Stream(priority=-5)` matching `cudaStreamCreateWithPriority` exactly.
- **CUDA Graphs** — Phase 9: `gpusim.Graph()` with explicit DAG of launches and dependencies.
- **Stream-stream wait sugar** — `s_b.wait_stream(s_a)` shorthand for `ev = Event(); s_a.record(ev); s_b.wait(ev)`.
- **Persistent kernels** — Phase 10: `Stream.persistent_launch(ptx, queue=...)` — kernel waits for new work in a queue instead of retiring.
- **L2 way-isolation alternative** — explore as alternative partitioning model.
- **Multi-event wait** — `Stream.wait_all([ev1, ev2])` for joining multiple producers.
- **Event timing** — `Event.elapsed_time(start_ev, end_ev)` mimicking cudaEventElapsedTime.

---

## 17. Acceptance criteria

Phase 8 ships when:

- [ ] All 5 milestone tags present (`M1-phase8-complete` ... `M4-phase8-complete`, `phase8-complete`)
- [ ] All 6 examples run cleanly (`python examples/<name>/run.py`)
- [ ] All 6 parity tests pass
- [ ] Microbench `test_phase8_facts.py::test_concurrent_faster_than_sequential` passes (Phase 8 ≤ 0.8× Phase 7 baseline cycles for compute/memory overlap)
- [ ] Microbench `test_phase8_facts.py::test_priority_high_finishes_faster_than_low` passes
- [ ] Microbench `test_phase8_facts.py::test_l2_window_protects_hit_rate` passes
- [ ] HTML report shows §29 + §30 + §31 when multi-stream + event + window events present
- [ ] Perfetto JSON has priority annotations + StreamEvent instant events
- [ ] Phase 1-7 regression test (renamed) passes: all prior examples unchanged
- [ ] Test count: 473 → ~510 (+37)
- [ ] README v8 documents Phase 8 capabilities + honest sequential-drain-removed migration note
