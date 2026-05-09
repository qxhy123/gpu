# gpusim Phase 9 — Per-cycle Slicing + L2 Eviction + Multi-event Wait + Event.elapsed_time

> **Status:** Brainstormed 2026-05-09. Implementation TBD.

## 1. Goals & Non-goals

### Goals
- **Rewrite `Device.run_streams` as a single per-cycle main loop**, fixing Phase 8's per-launch nesting limitation. Multiple streams' CTAs from different grids can now run in the same cycle, sharing SM/L2/HBM. The `compute_vs_memory_overlap`-style examples will show real cycle savings.
- **Wire L2 set-window protection into actual eviction.** Phase 8 added `register_stream_window` + `_line_in_window` helper but did not modify `CacheSet.install()`. Phase 9 changes the install path to skip evicting another stream's protected lines, and propagates `stream_id` through the gmem event recording so `l2_window_hit_rate` returns real numbers.
- **Add `Stream.wait_all(events)` multi-event wait** for fan-in patterns (consumer waits for multiple producers).
- **Add `Event.elapsed_time(start, end)` cycle-delta utility** mirroring `cudaEventElapsedTime` (returns `int` since cycles are discrete).
- Ship 3 examples + 3 tutorial chapters demonstrating the four refinements.
- 2 new analysis metrics + 1 HTML section + Perfetto record→wait arrows.
- 100% backward compatible: Phase 1-8 examples and tests pass unchanged.

### Non-goals (deferred to Phase 10+)
- CUDA Graphs / DAG of kernels.
- Multi-GPU / NVLink / NCCL.
- Numeric stream priority (still 3-level).
- `Stream.wait_any(events)` (OR-semantics fan-in not in CUDA).
- L2 way-isolation alternative.
- Stream callbacks.
- `cudaEventQuery` (poll without block).

---

## 2. Architecture

```
Phase 8 baseline:                    Phase 9:
Device.run_streams                   Device.run_streams
├── for stream:                      ├── one per-cycle main loop
│   ├── Device.run(grid)             │   ├── sched.step() → all streams concurrent
│   │   └── full-grid execution      │   ├── for sm in sms: sm.tick(cycle)
│   └── mark_grid_retired            │   ├── l2.tick(cycle); hbm.tick(cycle)
└── (sequential per launch)          │   ├── check grid retire + signal events
                                     │   └── advance cycle
                                     └── true cross-grid CTA interleave
```

### Key changes
- **`Device.run_streams` owns the cycle counter** and the per-cycle main loop.
- **`Device.run` (single-launch path) becomes a thin wrapper** that constructs an implicit Stream + GridLaunch and delegates to `run_streams`.
- **CTA dispatch is decoupled from grid execution**: scheduler dispatches CTAs to SMs cycle-by-cycle; SMs already support per-cycle tick.
- **L2Cache eviction install path** consults `_line_in_window` and skips protected lines from other streams.
- **gmem event recording** carries `hit` and `in_window` columns enabling `l2_window_hit_rate` metric.
- **Phase 8 ConcurrentStreamScheduler** unchanged at API level; the cycle main loop now actually invokes its `step()` per cycle.

---

## 3. Per-cycle Device.run_streams rewrite

### 3.1 New main loop in `gpusim/core/device.py`

```python
def run_streams(self, streams: list, *, events: list = None) -> "MultiStreamResult":
    """Phase 9: per-cycle main loop. Multiple streams' CTAs can run in same cycle."""
    from gpusim.core.scheduler import ConcurrentStreamScheduler
    from gpusim.api import MultiStreamResult, _RecordMarker
    
    # Phase 8 M4: register per-stream L2 windows
    for s in streams:
        if s.l2_window is not None:
            self.l2.register_stream_window(s.stream_id, *s.l2_window)
    
    weights = getattr(getattr(self.cfg, "scheduler", None), "priority_weights", None)
    sched = ConcurrentStreamScheduler(streams, priority_weights=weights)
    
    # Pre-instantiate SMs/L2/HBM (already done in __init__)
    cycle = 0
    max_cycles = 1_000_000  # safety cap
    
    while not all(s.is_idle() and s.in_flight_ctas == 0 for s in streams):
        if cycle >= max_cycles:
            raise RuntimeError(f"run_streams exceeded {max_cycles} cycles")
        
        # 1. Per-cycle dispatch
        decisions = sched.step(self._available_sms(), cycle)
        for stream, cta_idx, sm in decisions:
            self._dispatch_cta_to_sm(sm, stream, cta_idx, cycle)
        
        # 2. Tick everything
        for sm in self.sms: sm.tick(cycle)
        if hasattr(self, "l2"): self.l2.tick(cycle)
        if hasattr(self, "hbm"): self.hbm.tick(cycle)
        
        # 3. Check grid retire + event signaling
        for s in streams:
            if s.inflight is not None and self._stream_grid_retired(s):
                self._on_grid_retire(s, cycle, sched)
            self._check_event_signals(s, cycle)
        
        cycle += 1
    
    return self._build_multistream_result(streams, sched, cycle)
```

### 3.2 Helpers
- `_available_sms()` returns SMs with capacity for at least one more CTA (uses Phase 4 occupancy logic).
- `_dispatch_cta_to_sm(sm, stream, cta_idx, cycle)` calls existing SM dispatch path with `stream_id=stream.stream_id`.
- `_stream_grid_retired(s)` checks if all CTAs of `s.inflight` have completed (uses SM warp-state inspection or a per-stream completion counter).
- `_on_grid_retire(s, cycle, sched)` builds a Result for the completed grid, appends to `s.completed`, calls `sched.mark_grid_retired(s)`, and signals any `_RecordMarker` events tied to this stream's flush point.
- `_check_event_signals(s, cycle)` iterates pending record-markers and signals events when prior CTAs are done.
- `_build_multistream_result(...)` aggregates per-stream Results + total_cycles.

### 3.3 `Device.run` becomes thin wrapper
```python
def run(self, *, ptx_src=None, kernel=None, grid, block, params,
         stream_id: int = 0, kernel_name: str = "<unnamed>", **kwargs):
    """Single-launch path. Phase 9: implicit-stream wrapper around run_streams."""
    from gpusim.api import Stream, GridLaunch, _reset_stream_id_counter
    s = Stream()
    s.launch(ptx_src=ptx_src or "", grid=grid, block=block,
              params=params, kernel_name=kernel_name)
    multi_res = self.run_streams([s])
    return multi_res.streams[s.stream_id][0]
```

⚠ Implementation reality check: existing `Device.run` may already handle `kernel` (parsed) directly; keep that path for backward compat. The wrapper version is the long-term target but Phase 9 may keep both with `run_streams` calling `_run_single_grid_at_cycle(...)` internally — depending on how deeply current Device.run is wired into per-cycle SM tick. Pick whichever lets `run_streams` cleanly drive cycles.

---

## 4. L2 eviction integration

### 4.1 Modify `CacheSet.install()` in `gpusim/core/cache/line.py` (or wherever set lives)

```python
def install(self, addr: int, *, requesting_stream_id: int = -1,
              line_in_window_check=None) -> "CacheLine | None":
    """Install addr into this set. Returns the line, or None if no install possible
    (all candidates protected by other streams' windows).
    
    Args:
        requesting_stream_id: stream installing this line (for window protection)
        line_in_window_check: optional callback (line, set_idx) -> bool
    """
    # 1. Hit?
    for line in self.lines:
        if line.valid and line.addr == addr:
            line.last_use = self._now
            return line
    
    # 2. Pick victim — skip lines protected by other streams
    candidates = []
    for line in self.lines:
        if (line_in_window_check is not None
                and line_in_window_check(line, self.set_idx)
                and line.owner_stream_id != requesting_stream_id):
            continue   # protected
        candidates.append(line)
    
    if not candidates:
        return None    # cannot install
    
    victim = min(candidates, key=lambda c: c.last_use)
    # ... write back if dirty ...
    victim.addr = addr
    victim.valid = True
    victim.last_use = self._now
    victim.owner_stream_id = requesting_stream_id
    if line_in_window_check is not None:
        victim.in_window = line_in_window_check(victim, self.set_idx)
    return victim
```

### 4.2 `L2Cache.lookup()` passes `requesting_stream_id` + window-check callback

```python
def lookup(self, addr: int, requesting_stream_id: int = -1):
    set_idx = self._set_index(addr)
    cs = self.sets[set_idx]
    line = cs.install(addr, requesting_stream_id=requesting_stream_id,
                       line_in_window_check=self._line_in_window)
    if line is None:
        # Caller treats as miss without caching — penalty is HBM round-trip
        return _MissNoInstall()
    return line
```

### 4.3 Caller sites pass `stream_id`
Where SubCore/SM call into L2 (gmem load/store paths), pass the current CTA's `stream_id` as `requesting_stream_id`. The gmem event recording also picks up `hit` (whether `lookup` returned an existing line vs new install) and `in_window` (whether line is in protected window).

### 4.4 GmemEvent gains `hit` + `in_window` fields (Phase 8 had stream_id)
```python
@dataclass(frozen=True)
class GmemEvent:
    # ... existing fields ...
    hit: bool = False           # NEW Phase 9 — L2 hit flag
    in_window: bool = False      # NEW Phase 9 — line was in protected window
```

Recorder.gmem_access gains `hit` + `in_window` kwargs (default False for backward compat).

This unblocks the Phase 8 `l2_window_hit_rate_per_stream` and `l2_window_protection_efficiency` metrics — they now have real data.

---

## 5. Multi-event wait

### 5.1 New `Stream.wait_all(events)` method

```python
class Stream:
    def wait_all(self, events: list["Event"]) -> None:
        """Block this stream's future launches until ALL events are signaled."""
        for ev in events:
            self.event_waits.append(ev)
```

Functionally identical to calling `wait()` repeatedly (Phase 8 already supports multi-event in `event_waits` list); added as explicit API for clarity.

### 5.2 Scheduler unchanged — already AND-semantics
`ConcurrentStreamScheduler.is_event_blocked(s)` already returns True if ANY event in `s.event_waits` is unsignaled. So multi-event "all" semantics is free.

### 5.3 API example
```python
ev_a = gpusim.Event()
ev_b = gpusim.Event()
s_consumer.wait_all([ev_a, ev_b])
s_consumer.launch(...)   # waits for both producers
```

---

## 6. Event.elapsed_time

### 6.1 New method on Event class

```python
@dataclass
class Event:
    # ... existing fields ...
    
    @staticmethod
    def elapsed_time(start: "Event", end: "Event") -> int:
        """Cycles between two signaled events. Both must be signaled.
        Returns int (cycles) — discrete, not float ms like real CUDA."""
        if start.signaled_at_cycle is None:
            raise RuntimeError(f"start event {start.event_id} not signaled")
        if end.signaled_at_cycle is None:
            raise RuntimeError(f"end event {end.event_id} not signaled")
        return end.signaled_at_cycle - start.signaled_at_cycle
```

### 6.2 API example
```python
ev_start = gpusim.Event()
ev_end = gpusim.Event()
s.record(ev_start)
s.launch(...)
s.record(ev_end)
gpusim.synchronize(streams=[s], config=cfg)
print(f"Kernel took {gpusim.Event.elapsed_time(ev_start, ev_end)} cycles")
```

---

## 7. Trace + Analysis

### 7.1 GmemEvent gains `hit` + `in_window` (covered in §4.4)

### 7.2 2 new metrics (`gpusim/analysis/metrics.py`)

```python
def actual_cross_grid_overlap_cycles(kernel_launch_df, total_cycles: int) -> int:
    """Cycles where ≥2 launches were in-flight simultaneously.
    Higher = more cross-grid concurrency realized."""
    if kernel_launch_df is None or kernel_launch_df.empty:
        return 0
    overlap = 0
    for cycle in range(total_cycles):
        active = sum(1 for _, row in kernel_launch_df.iterrows()
                       if row["launch_cycle"] <= cycle <= row["complete_cycle"])
        if active >= 2:
            overlap += 1
    return overlap


def l2_eviction_protected_count(gmem_events_df) -> dict:
    """Per-stream count of L2 evictions blocked by another stream's window protection.
    Inferred from misses where the line could not be installed."""
    if gmem_events_df is None or gmem_events_df.empty:
        return {}
    # Caller reports "miss without install" via a special hit=False, in_window=False marker;
    # for Phase 9 simplicity, we count misses where requesting stream had no window
    # and the address lands in a set with another stream's window.
    # Implementation TBD via gmem event schema.
    ...
```

### 7.3 MultiStreamResult methods
```python
def actual_cross_grid_overlap_cycles(self) -> int: ...
def l2_eviction_protected_count(self) -> dict: ...
```

---

## 8. Viz

### 8.1 HTML §32 — Combined overlap visualization

`gpusim/viz/html_report.py` adds `_render_combined_overlap(rec)` that produces a single gantt chart annotating per-stream timelines with priority badges (color), event record/wait markers (icons), and L2 window highlights (background tint).

Template addition in `_template.html.j2`:
```html
{% if combined_overlap_html %}
<h2>§32 Combined overlap — priority + events + L2 window</h2>
{{ combined_overlap_html | safe }}
{% endif %}
```

### 8.2 Perfetto record→wait arrows

`gpusim/viz/perfetto.py` adds async event pairs (`ph: "s"` start + `ph: "f"` finish) for each `record` → `wait_satisfied` pair in the same `event_id`, drawing arrows in Perfetto UI.

---

## 9. Examples (3)

### 9.1 `phase8_overlap_real/`
- Same compute-heavy + memory-heavy kernels as Phase 8 `compute_vs_memory_overlap`.
- **Verifies on Phase 9**: total_cycles ≤ 0.7× sum(per-launch cycles), proving real overlap; `actual_cross_grid_overlap_cycles() > 0`.
- Files: 6 (kernel_compute.ptx, kernel_memory.ptx, reference.py, run.py, README.md, __init__.py).
- Parity test asserts cycles range + `cross_stream_concurrency_gain() ≥ 1.4`.

### 9.2 `multi_event_fan_in/`
- 2 producers (s_a, s_b) → 1 consumer (s_c) using `s_c.wait_all([ev_a, ev_b])`.
- **Verifies**: consumer sees BOTH producers' writes; doesn't start until both events signal.
- Parity test asserts data correctness + `event_chain_critical_path()` matches expected depth.

### 9.3 `event_timing_benchmark/`
- 3 launches with `record(ev_i)` between each.
- Use `Event.elapsed_time(ev1, ev2)` to time each phase.
- **Verifies**: `elapsed_time` returns positive int matching expected cycles.
- Demonstrates the timing pattern programmers use to profile kernels.

---

## 10. Tutorials

`docs/tutorial/`, ~500-700 words each:

- **37-per-cycle-scheduler-and-real-overlap.md** — example 1 ⭐
- **38-multi-event-fan-in-pattern.md** — example 2
- **39-event-timing-and-profiling.md** — example 3

---

## 11. Testing strategy

### Unit tests (~10 new)
- `tests/unit/api/test_event_elapsed_time.py` — Event.elapsed_time validation + edge cases
- `tests/unit/api/test_stream_wait_all.py` — Stream.wait_all + multi-event blocking
- `tests/unit/core/test_device_per_cycle_loop.py` — per-cycle main loop correctness; backward compat with single-stream
- `tests/unit/cache/test_l2_eviction_window_protection.py` — install() respects window; cross-stream isolation; hit/in_window columns
- `tests/unit/analysis/test_phase9_metrics.py` — 2 new metrics

### Parity tests (~3) — one per example

### Microbench
- `test_phase9_facts.py` (fast):
  - Phase 9 cycles ≤ 0.7× Phase 8 baseline for compute_vs_memory_overlap
  - L2 window protection: windowed stream hit rate ≥ 1.3× unwindowed
  - Multi-event wait: consumer block cycles ≥ longest producer
- `test_phase9_runtime.py` (slow): 3 examples each under 60s

### Regression
- Rename `tests/parity/test_phase1_7_examples_unchanged.py` → `test_phase1_8_examples_unchanged.py`
- Add 6 Phase 8 examples to the regression list

### Test count target
520 (Phase 8 baseline) → ~540 (+20).

---

## 12. Milestones

| Milestone | Scope | Tag |
|---|---|---|
| **M1** Per-cycle Device.run_streams main loop + phase8_overlap_real | Rewrite run_streams; verify cross-grid concurrency; parity test passes | `M1-phase9-complete` |
| **M2** L2 eviction integration | CacheSet.install window protection + GmemEvent.hit/in_window + l2_window_hit_rate returns real data | `M2-phase9-complete` |
| **M3** Multi-event wait + Event.elapsed_time + 2 examples | Stream.wait_all + Event.elapsed_time + multi_event_fan_in + event_timing_benchmark | `M3-phase9-complete` |
| **M4** 2 metrics + HTML §32 + Perfetto arrows | actual_cross_grid_overlap_cycles + l2_eviction_protected_count + viz | `M4-phase9-complete` |
| **M5** Tutorials + microbench + Phase 1-8 regression rename + README v9 + ship | 3 chapters + microbench + 3 ref stubs + README | `phase9-complete` |

Estimated 24 tasks total.

---

## 13. File list

### New files
```
examples/phase8_overlap_real/                   # 6 files (M1)
examples/multi_event_fan_in/                    # 5 files (M3)
examples/event_timing_benchmark/                # 5 files (M3)
docs/tutorial/37-per-cycle-scheduler-and-real-overlap.md
docs/tutorial/38-multi-event-fan-in-pattern.md
docs/tutorial/39-event-timing-and-profiling.md
tests/unit/api/test_event_elapsed_time.py
tests/unit/api/test_stream_wait_all.py
tests/unit/core/test_device_per_cycle_loop.py
tests/unit/cache/test_l2_eviction_window_protection.py
tests/unit/analysis/test_phase9_metrics.py
tests/parity/test_phase8_overlap_real.py
tests/parity/test_multi_event_fan_in.py
tests/parity/test_event_timing_benchmark.py
tests/microbench/test_phase9_facts.py
tests/microbench/test_phase9_runtime.py
tests/reference/data/{3 example names}.ref.json
```

### Modified files
```
gpusim/api.py                                   # Stream.wait_all + Event.elapsed_time (staticmethod)
gpusim/core/device.py                           # run_streams rewritten as per-cycle main loop;
                                                # run() becomes thin wrapper
gpusim/core/cache/line.py (or l2.py)            # CacheSet.install accepts requesting_stream_id +
                                                # line_in_window_check; LRU skips protected lines
gpusim/core/cache/l2.py                         # lookup() passes requesting_stream_id;
                                                # plumb to gmem event hit/in_window
gpusim/core/sub_core.py                         # gmem load/store path passes stream_id +
                                                # records hit/in_window in GmemEvent
gpusim/trace/events.py                          # GmemEvent + hit + in_window fields
gpusim/trace/recorder.py                        # gmem_access accepts hit + in_window kwargs
gpusim/analysis/metrics.py                      # +2 metrics
gpusim/api.py                                   # MultiStreamResult.actual_cross_grid_overlap_cycles + l2_eviction_protected_count
gpusim/viz/html_report.py                       # +§32 combined overlap helper
gpusim/viz/_template.html.j2                    # +§32 block
gpusim/viz/perfetto.py                          # record→wait async arrows
tests/parity/test_phase1_7_examples_unchanged.py → test_phase1_8_examples_unchanged.py
tests/reference/gen_reference.py                # +3 kernel names
README.md                                       # v9 — Phase 9 capabilities + honest notes update
```

---

## 14. Backward compatibility

- `gpusim.run(...)` — unchanged behavior, returns Result with stream_id=0.
- Phase 8 multi-stream API (`Stream.launch`, `gpusim.synchronize`) — unchanged.
- Phase 8 `Stream.wait(ev)` — unchanged; `wait_all` is additive.
- Phase 8 `Event` class — gains `elapsed_time` staticmethod; no breaking change.
- Phase 1-8 examples + tests — pass unchanged.
- `MultiStreamScheduler` alias — kept (still points to `ConcurrentStreamScheduler`).
- The internal `Device.run` rewrite is invisible at API level: caller still gets a Result.

---

## 15. Open questions / future work

- **`Stream.wait_any([ev1, ev2])`** — OR-semantics; not in CUDA, low priority.
- **`Event.query()`** — non-blocking poll; useful for advanced scheduling.
- **L2 way-isolation alternative** — alternative partitioning model (Phase 8 had set-affinity).
- **Real-time priority weights** — runtime adjustment of priority_weights per-launch.
- **Float-precision elapsed_time** — currently int cycles; could derive float ms using simulated clock rate.

---

## 16. Acceptance criteria

Phase 9 ships when:

- [ ] All 5 milestone tags present (`M1-phase9-complete` ... `M4-phase9-complete`, `phase9-complete`)
- [ ] All 3 examples run cleanly (`python examples/<name>/run.py`)
- [ ] All 3 parity tests pass
- [ ] Microbench `test_phase9_facts.py::test_real_overlap_savings` passes (Phase 9 ≤ 0.7× Phase 8 baseline cycles for compute/memory overlap)
- [ ] L2 window hit-rate metric returns non-zero for windowed streams in `l2_window_demo`
- [ ] HTML report shows §32 when multi-stream + event + window events present
- [ ] Perfetto JSON has record→wait async arrows
- [ ] Phase 1-8 regression test (renamed) passes: all prior examples unchanged
- [ ] Test count: 520 → ~540 (+20)
- [ ] README v9 documents Phase 9 capabilities + removes "Phase 8 limitation" notes that are now resolved
