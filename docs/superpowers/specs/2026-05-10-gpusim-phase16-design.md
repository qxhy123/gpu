# gpusim Phase 16 — Memory Pools + Stream-Ordered Async Allocator

> **Status:** Brainstormed 2026-05-10. Implementation TBD.

## 1. Goals & Non-goals

### Goals
- Add `gpusim.mempool` package with `MemoryPool` class.
- Stream-ordered semantics: `pool.malloc_async(stream, n_bytes)` and `pool.free_async(stream, allocation)`. Free on stream A makes the block reusable on stream A immediately; reuse on stream B requires explicit event-based sync (record/wait via Phase 8).
- `Allocation` handle wraps a numpy ndarray view (`alloc.buf`) so existing kernel-launch APIs accept pool buffers without change.
- Pool grows lazily by allocating new `bytearray` slabs when no free block fits; tracks high-water-mark.
- `pool.trim_to(release_threshold_bytes)` releases free blocks above the threshold back to "OS" (drops the bytearray).
- 4 examples + 4 tutorial chapters (62-65).
- 4 metrics: `pool_high_water_mark`, `pool_reuse_rate`, `pool_alloc_count`, `pool_release_total_bytes`.
- 4 trace events: `PoolAllocate`, `PoolFree`, `PoolGrow`, `PoolTrim`.
- 100% backward compatible: Phase 1-15 unchanged.

### Non-goals (deferred to Phase 17+)
- IPC mempool (`cudaMemPoolExportToShareableHandle`).
- Multi-GPU pool sharing.
- Default device pool (`cudaDeviceGetDefaultMemPool`) — Phase 16 requires an explicit `MemoryPool` instance.
- Pool attributes other than `release_threshold` (e.g., `cudaMemPoolReuseAllowOpportunistic`, `cudaMemPoolReuseAllowInternalDependencies`).
- Memory advice / unified memory (`cudaMemAdvise`, `cudaMemPrefetchAsync`).
- Coalescing of adjacent free blocks within a slab.
- Real CUDA's "header" overhead modeling.

---

## 2. Architecture

```
gpusim.mempool (NEW package):
├── allocation.py
│   └── Allocation                   # ptr_id, n_bytes, buf (ndarray view), pool ref, alloc_stream
├── pool.py
│   ├── MemoryPool                   # the pool
│   ├── .malloc_async(stream, n_bytes, dtype=np.uint8) -> Allocation
│   ├── .free_async(stream, allocation) -> None
│   ├── .trim_to(release_threshold_bytes) -> int   # returns bytes released
│   ├── .release_threshold (attr, settable; default 2**30)
│   └── internal state: free_blocks_by_stream, in_flight_bytes, high_water_mark, slabs, _pool_id
└── __init__.py                       # exports MemoryPool, Allocation

gpusim.api (Stream extension):
├── Stream.malloc_async(pool, n_bytes, dtype=np.uint8) -> Allocation     # forwards to pool
└── Stream.free_async(pool, allocation) -> None                            # forwards to pool

gpusim.trace.events (NEW dataclasses):
├── PoolAllocate(pool_id, stream_id, n_bytes, ptr_id, reused: bool, cycle)
├── PoolFree(pool_id, stream_id, ptr_id, n_bytes, cycle)
├── PoolGrow(pool_id, n_bytes_added, cycle)
└── PoolTrim(pool_id, n_bytes_released, cycle)

gpusim.trace.recorder (NEW methods):
├── recorder.pool_allocate(...)
├── recorder.pool_free(...)
├── recorder.pool_grow(...)
└── recorder.pool_trim(...)

gpusim.analysis.metrics (NEW functions):
├── pool_high_water_mark(recorder, pool_id) -> int
├── pool_reuse_rate(recorder, pool_id) -> float
├── pool_alloc_count(recorder, pool_id) -> int
└── pool_release_total_bytes(recorder, pool_id) -> int
```

### Key invariants

- **Same-stream reuse is immediate.** `free_async` on stream A pushes the block onto `free_blocks_by_stream[A.stream_id]`. The next `malloc_async` on stream A scans this list first.
- **Cross-stream reuse requires event sync.** Block released on stream A is NOT reachable from stream B unless B has called `wait(ev)` for an event recorded after the free on A. Phase 16 implements this conservatively: cross-stream reuse only happens for blocks the pool has explicitly "promoted" to the cross-stream free pool. A block is promoted when its `alloc_stream` has gone idle (`stream.is_idle()`) AND a synchronization event has been recorded after the free. We approximate this by: when `pool.free_async(stream, alloc)` is called, the block is staged on per-stream free list; calling `pool.synchronize_stream(stream)` (called by `Stream.record(event)` automatically when capture is not in flight) promotes all per-stream free blocks on that stream into the cross-stream pool.
- **Block reuse policy: best-fit.** Among free blocks (same-stream first, then cross-stream pool), pick the smallest block that's `>= n_bytes`. Tie-break by oldest free time. If none fits, allocate a new slab of exactly `n_bytes` (no over-allocation; YAGNI for slab-size tuning).
- **Cycle costs:**
  - Fresh allocation (slab grow): 100 cycles
  - Reuse from free list: 5 cycles
  - Free: 20 cycles
  - Trim: 50 cycles per slab released
- **Pool ID:** auto-assigned via a module-level counter, exposed as `pool._pool_id` (read-only).
- **Buffer dtype:** caller passes `dtype=np.uint8` by default. Pool stores raw bytes; `Allocation.buf` is a typed view via `np.frombuffer(slab, dtype=dtype, count=n_elements, offset=byte_offset)` where `n_elements = n_bytes // dtype().itemsize`.
- **Allocation handle is opaque to outside callers.** They use `alloc.buf` (the ndarray view) wherever a kernel param expects a buffer.

---

## 3. Data model

### 3.1 `Allocation` (`gpusim/mempool/allocation.py`)

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Allocation:
    ptr_id: int                 # unique within pool
    n_bytes: int
    buf: object                 # numpy ndarray view (typed)
    pool: object                # MemoryPool reference
    alloc_stream_id: int        # stream that allocated this block
    _slab_index: int            # which backing slab in pool.slabs
    _byte_offset: int           # offset within slab
```

### 3.2 `MemoryPool` (`gpusim/mempool/pool.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


_pool_id_counter = 0
def _next_pool_id() -> int:
    global _pool_id_counter
    _pool_id_counter += 1
    return _pool_id_counter


@dataclass
class _FreeBlock:
    n_bytes: int
    slab_index: int
    byte_offset: int
    freed_at_count: int          # monotonic counter for tiebreak


@dataclass
class MemoryPool:
    release_threshold: int = 2**30
    _pool_id: int = field(default_factory=_next_pool_id)

    # backing storage: list of bytearray slabs
    slabs: list = field(default_factory=list)               # list[bytearray]
    slab_n_bytes: list = field(default_factory=list)        # list[int]
    slab_in_use_bytes: list = field(default_factory=list)   # list[int]

    # free blocks keyed by alloc_stream_id; cross-stream pool keyed under -1
    free_blocks_by_stream: dict = field(default_factory=lambda: defaultdict(list))

    in_flight_bytes: int = 0
    high_water_mark: int = 0
    _free_counter: int = 0
    _next_ptr_id: int = 0
    _recorder: object | None = None

    def malloc_async(self, stream, n_bytes: int, dtype=None) -> "Allocation":
        from gpusim.mempool.allocation import Allocation
        if dtype is None:
            dtype = np.uint8

        sid = stream.stream_id
        # 1. Try same-stream free list
        block, reused = self._pop_best_fit(self.free_blocks_by_stream[sid], n_bytes)
        # 2. Fall back to cross-stream pool
        if block is None:
            block, reused = self._pop_best_fit(self.free_blocks_by_stream[-1], n_bytes)
        # 3. Grow
        if block is None:
            slab_idx = self._grow(n_bytes)
            block = _FreeBlock(n_bytes=n_bytes, slab_index=slab_idx,
                               byte_offset=0, freed_at_count=-1)
            self.slab_in_use_bytes[slab_idx] = n_bytes
            reused = False

        n_elements = n_bytes // np.dtype(dtype).itemsize
        view = np.frombuffer(self.slabs[block.slab_index],
                             dtype=dtype, count=n_elements,
                             offset=block.byte_offset)

        ptr_id = self._next_ptr_id; self._next_ptr_id += 1
        alloc = Allocation(ptr_id=ptr_id, n_bytes=block.n_bytes,
                            buf=view, pool=self, alloc_stream_id=sid,
                            _slab_index=block.slab_index,
                            _byte_offset=block.byte_offset)

        self.in_flight_bytes += block.n_bytes
        if self.in_flight_bytes > self.high_water_mark:
            self.high_water_mark = self.in_flight_bytes

        if self._recorder is not None:
            self._recorder.pool_allocate(
                pool_id=self._pool_id, stream_id=sid,
                n_bytes=block.n_bytes, ptr_id=ptr_id, reused=reused,
                cycle=0,
            )
        return alloc

    def free_async(self, stream, allocation: "Allocation") -> None:
        sid = stream.stream_id
        self._free_counter += 1
        self.free_blocks_by_stream[sid].append(_FreeBlock(
            n_bytes=allocation.n_bytes,
            slab_index=allocation._slab_index,
            byte_offset=allocation._byte_offset,
            freed_at_count=self._free_counter,
        ))
        self.in_flight_bytes -= allocation.n_bytes
        if self._recorder is not None:
            self._recorder.pool_free(
                pool_id=self._pool_id, stream_id=sid,
                ptr_id=allocation.ptr_id, n_bytes=allocation.n_bytes,
                cycle=0,
            )

    def synchronize_stream(self, stream) -> None:
        """Promote all per-stream free blocks for this stream into the cross-stream pool.
        Called explicitly by user code or by Stream.record(event) when no capture is active."""
        sid = stream.stream_id
        promoted = self.free_blocks_by_stream.pop(sid, [])
        if promoted:
            self.free_blocks_by_stream[-1].extend(promoted)

    def trim_to(self, release_threshold_bytes: int) -> int:
        """Release fully-free slabs whose total free bytes exceed the threshold.
        Returns total bytes released."""
        # Aggregate free bytes per slab.
        per_slab_free = defaultdict(int)
        per_slab_blocks = defaultdict(list)   # slab_idx -> list[(stream_key, list_idx)]
        for stream_key, blocks in self.free_blocks_by_stream.items():
            for i, b in enumerate(blocks):
                per_slab_free[b.slab_index] += b.n_bytes
                per_slab_blocks[b.slab_index].append((stream_key, i))

        # A slab is "fully free" if per_slab_free[slab_idx] == self.slab_n_bytes[slab_idx].
        released = 0
        slabs_to_release = []
        for slab_idx, free_bytes in per_slab_free.items():
            if free_bytes != self.slab_n_bytes[slab_idx]:
                continue
            # Only release if we are above release_threshold_bytes after release.
            current_total = sum(self.slab_n_bytes)
            if current_total - self.slab_n_bytes[slab_idx] >= release_threshold_bytes:
                slabs_to_release.append(slab_idx)
                released += self.slab_n_bytes[slab_idx]

        if slabs_to_release:
            # Remove free blocks pointing to released slabs
            for stream_key in list(self.free_blocks_by_stream.keys()):
                self.free_blocks_by_stream[stream_key] = [
                    b for b in self.free_blocks_by_stream[stream_key]
                    if b.slab_index not in slabs_to_release
                ]
            # Mark slabs as released (set to None so indices stay stable)
            for slab_idx in slabs_to_release:
                self.slabs[slab_idx] = None
                self.slab_n_bytes[slab_idx] = 0
                self.slab_in_use_bytes[slab_idx] = 0

        if self._recorder is not None and released > 0:
            self._recorder.pool_trim(
                pool_id=self._pool_id, n_bytes_released=released, cycle=0,
            )
        return released

    # ---- private helpers ----

    def _pop_best_fit(self, blocks: list, n_bytes: int):
        """Find smallest block >= n_bytes. Tiebreak by oldest free."""
        candidates = [(i, b) for i, b in enumerate(blocks) if b.n_bytes >= n_bytes]
        if not candidates:
            return None, False
        candidates.sort(key=lambda ib: (ib[1].n_bytes, ib[1].freed_at_count))
        i, block = candidates[0]
        blocks.pop(i)
        return block, True

    def _grow(self, n_bytes: int) -> int:
        slab_idx = len(self.slabs)
        self.slabs.append(bytearray(n_bytes))
        self.slab_n_bytes.append(n_bytes)
        self.slab_in_use_bytes.append(0)
        if self._recorder is not None:
            self._recorder.pool_grow(
                pool_id=self._pool_id, n_bytes_added=n_bytes, cycle=0,
            )
        return slab_idx
```

### 3.3 `Stream` extensions (`gpusim/api.py`)

Add two thin convenience methods on `Stream`:

```python
    def malloc_async(self, pool, n_bytes: int, dtype=None) -> "Allocation":
        """Allocate from pool on this stream (Phase 16)."""
        return pool.malloc_async(self, n_bytes, dtype=dtype)

    def free_async(self, pool, allocation) -> None:
        """Return allocation to pool on this stream (Phase 16)."""
        pool.free_async(self, allocation)
```

`Stream.record(event)` is **not** modified by Phase 16. The cross-stream reuse promotion is done explicitly via `pool.synchronize_stream(stream)` in user code (multi-stream example demonstrates this). Keeping it explicit avoids hidden coupling between `gpusim.api` and `gpusim.mempool`.

---

## 4. Trace + Analysis

### 4.1 Trace events (`gpusim/trace/events.py`)

```python
@dataclass(frozen=True)
class PoolAllocate:
    pool_id: int
    stream_id: int
    n_bytes: int
    ptr_id: int
    reused: bool
    cycle: int

@dataclass(frozen=True)
class PoolFree:
    pool_id: int
    stream_id: int
    ptr_id: int
    n_bytes: int
    cycle: int

@dataclass(frozen=True)
class PoolGrow:
    pool_id: int
    n_bytes_added: int
    cycle: int

@dataclass(frozen=True)
class PoolTrim:
    pool_id: int
    n_bytes_released: int
    cycle: int
```

### 4.2 4 metrics (`gpusim/analysis/metrics.py`)

```python
def pool_high_water_mark(recorder, pool_id: int) -> int:
    """Phase 16: peak in_flight bytes for a pool over the trace.
    Computed by walking PoolAllocate/PoolFree events in cycle order."""
    events = []
    for ev in getattr(recorder, "pool_allocate_events", []):
        if ev.pool_id == pool_id:
            events.append((ev.cycle, +ev.n_bytes))
    for ev in getattr(recorder, "pool_free_events", []):
        if ev.pool_id == pool_id:
            events.append((ev.cycle, -ev.n_bytes))
    events.sort()
    in_flight = 0
    high = 0
    for _, delta in events:
        in_flight += delta
        if in_flight > high:
            high = in_flight
    return high


def pool_reuse_rate(recorder, pool_id: int) -> float:
    """Phase 16: fraction of allocations that reused a previously-freed block."""
    events = [ev for ev in getattr(recorder, "pool_allocate_events", [])
              if ev.pool_id == pool_id]
    if not events:
        return 0.0
    reused = sum(1 for ev in events if ev.reused)
    return reused / len(events)


def pool_alloc_count(recorder, pool_id: int) -> int:
    """Phase 16: total malloc_async calls on a pool."""
    return sum(1 for ev in getattr(recorder, "pool_allocate_events", [])
               if ev.pool_id == pool_id)


def pool_release_total_bytes(recorder, pool_id: int) -> int:
    """Phase 16: total bytes released by trim_to calls on a pool."""
    return sum(ev.n_bytes_released
               for ev in getattr(recorder, "pool_trim_events", [])
               if ev.pool_id == pool_id)
```

---

## 5. Viz

Reuse Phase 11 HTML report. Phase 16 adds **§36 Memory Pool Stats** (per-pool: high-water-mark, reuse rate, alloc count, total released bytes) — a small table appended to the existing HTML structure.

No new Perfetto swimlane; pool events are reported in §36 only.

---

## 6. Examples (4)

### 6.1 `mempool_basic/`
- 1 stream, 1 pool. Loop 4 times: `alloc → free`. Verify alloc 1 grows the pool, alloc 2-4 reuse. Reuse rate ≥ 0.75.

### 6.2 `mempool_fragmentation/`
- Allocate sizes [1024, 2048, 4096, 1024, 2048] in order, free all, then re-alloc [1024, 2048] → expect both reused. Show that best-fit finds the right blocks.

### 6.3 `mempool_multi_stream/`
- 2 streams. sA allocs+frees a block; without `pool.synchronize_stream(sA)`, sB malloc must grow. After `synchronize_stream`, sB malloc reuses the promoted block.

### 6.4 `mempool_train_step/`
- Toy "training step" loop: each iteration allocs activation (8KB) + grad (8KB), runs a kernel that touches both, frees both, then `synchronize_stream(stream)`. Verify high_water_mark stabilizes after iteration 1 (~16KB) and reuse rate approaches 1.0.

---

## 7. Tutorials

`docs/tutorial/` chapters 62-65:
- **62-mempool-basic.md** — example 1
- **63-mempool-fragmentation.md** — example 2
- **64-mempool-multi-stream.md** — example 3
- **65-mempool-train-step.md** — example 4

---

## 8. Testing strategy

### Unit tests (~14 new)
- `tests/unit/mempool/test_pool_basic.py` — single-stream malloc/free/reuse, grow on first alloc, in_flight tracking
- `tests/unit/mempool/test_pool_best_fit.py` — best-fit selection, tiebreak by oldest free
- `tests/unit/mempool/test_pool_cross_stream.py` — synchronize_stream promotes blocks; without it, cross-stream malloc grows
- `tests/unit/mempool/test_pool_trim.py` — trim_to releases fully-free slabs, respects release_threshold
- `tests/unit/mempool/test_allocation_view.py` — Allocation.buf is a numpy view that kernels can read/write
- `tests/unit/mempool/test_stream_convenience.py` — Stream.malloc_async / free_async forward correctly
- `tests/unit/analysis/test_phase16_metrics.py` — 4 new metrics

### Parity tests (~4)
One per example.

### Microbench
- `test_phase16_facts.py` (fast):
  - First alloc records `reused=False`; second-after-free records `reused=True`
  - Cross-stream alloc without sync grows the pool
  - trim_to releases fully-free slabs only
  - high_water_mark monotone non-decreasing
- `test_phase16_runtime.py` (slow): 4 examples each under 30s

### Regression
- Rename `tests/parity/test_phase1_14_examples_unchanged.py` → `test_phase1_15_examples_unchanged.py`
- Add 4 Phase 15 examples (`stream_capture_basic`, `stream_capture_multi_stream`, `graph_conditional_branch`, `graph_while_loop`) to the regression list

### Test count target
770 (Phase 15 baseline) → ~800 (+30).

---

## 9. Milestones

| Milestone | Scope | Tag |
|---|---|---|
| **M1** Pool core + basic example | MemoryPool single-stream malloc/free/reuse, Allocation, basic trace events, mempool_basic example | `M1-phase16-complete` |
| **M2** Cross-stream + multi_stream example | synchronize_stream + cross-stream pool + Stream convenience methods + mempool_multi_stream | `M2-phase16-complete` |
| **M3** trim_to + fragmentation example | trim_to + release_threshold + PoolTrim event + mempool_fragmentation | `M3-phase16-complete` |
| **M4** 4 metrics + train_step example | pool_high_water_mark + pool_reuse_rate + pool_alloc_count + pool_release_total_bytes + mempool_train_step | `M4-phase16-complete` |
| **M5** Tutorials + microbench + regression rename + README v16 + ship | 4 chapters + microbench + Phase 1-15 regression rename + README | `phase16-complete` |

Estimated 22 tasks total.

---

## 10. File list

### New files
```
gpusim/mempool/__init__.py
gpusim/mempool/allocation.py
gpusim/mempool/pool.py
examples/mempool_basic/                # 5 files (M1)
examples/mempool_multi_stream/         # 5 files (M2)
examples/mempool_fragmentation/        # 5 files (M3)
examples/mempool_train_step/           # 5 files (M4)
docs/tutorial/62-mempool-basic.md
docs/tutorial/63-mempool-fragmentation.md
docs/tutorial/64-mempool-multi-stream.md
docs/tutorial/65-mempool-train-step.md
tests/unit/mempool/__init__.py
tests/unit/mempool/test_pool_basic.py
tests/unit/mempool/test_pool_best_fit.py
tests/unit/mempool/test_pool_cross_stream.py
tests/unit/mempool/test_pool_trim.py
tests/unit/mempool/test_allocation_view.py
tests/unit/mempool/test_stream_convenience.py
tests/unit/analysis/test_phase16_metrics.py
tests/parity/test_mempool_basic.py
tests/parity/test_mempool_multi_stream.py
tests/parity/test_mempool_fragmentation.py
tests/parity/test_mempool_train_step.py
tests/microbench/test_phase16_facts.py
tests/microbench/test_phase16_runtime.py
```

### Modified files
```
gpusim/api.py                       # +Stream.malloc_async +Stream.free_async
gpusim/trace/events.py              # +PoolAllocate +PoolFree +PoolGrow +PoolTrim
gpusim/trace/recorder.py            # +4 recorder methods + 4 list slots
gpusim/analysis/metrics.py          # +4 metrics
tests/parity/test_phase1_14_examples_unchanged.py → test_phase1_15_examples_unchanged.py
README.md                           # v16 — Phase 16 capabilities
```

---

## 11. Backward compatibility

- All Phase 1-15 examples + tests pass unchanged.
- Pool API is opt-in: existing examples use raw numpy buffers as before.
- `Stream.malloc_async` / `free_async` are new methods on `Stream`; no existing attributes touched.
- `Stream.record` is unchanged — Phase 16 does NOT auto-promote on record. User must call `pool.synchronize_stream(stream)` explicitly.

---

## 12. Acceptance criteria

Phase 16 ships when:

- [ ] All 5 milestone tags present (`M1-phase16-complete` ... `M4-phase16-complete`, `phase16-complete`)
- [ ] All 4 examples run cleanly via `python run.py`
- [ ] All 4 parity tests pass
- [ ] Microbench facts: reuse vs fresh-alloc flag correct, cross-stream grows without sync, trim respects release_threshold, high_water_mark monotone
- [ ] Phase 1-15 regression test (renamed) passes
- [ ] Test count: 770 → ~800 (+30)
- [ ] README v16 documents Phase 16 capabilities
