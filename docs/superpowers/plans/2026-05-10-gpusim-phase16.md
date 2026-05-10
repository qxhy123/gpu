# gpusim Phase 16 Implementation Plan — Memory Pools + Stream-Ordered Async Allocator

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `gpusim.mempool` package providing `MemoryPool` with stream-ordered `malloc_async` / `free_async` / `trim_to`, plus 4 trace events, 4 metrics, 4 examples, and 4 tutorial chapters.

**Architecture:** New self-contained package `gpusim/mempool/{pool,allocation}.py`. `MemoryPool` keeps `bytearray` slabs as backing storage; `Allocation.buf` is a typed numpy view via `np.frombuffer` so any kernel that accepts an ndarray param works unchanged. Stream-ordered semantics are implemented as same-stream free → immediate reuse; cross-stream reuse requires explicit `pool.synchronize_stream(stream)` to promote per-stream free blocks into the cross-stream pool keyed under `-1`. Best-fit allocation policy with oldest-free tiebreak.

**Tech Stack:** Python 3, numpy, pytest. New package `gpusim/mempool/`. Existing modules touched: `gpusim/api.py` (Stream convenience methods), `gpusim/trace/{events,recorder}.py` (4 new events), `gpusim/analysis/metrics.py` (4 new metrics).

**CRITICAL — environment:** All `pytest` invocations must use the project's `.venv`:
```
.venv/bin/pytest <args>
```
or
```
source .venv/bin/activate && pytest <args>
```
Conda Python at `/Users/yangyang/anaconda3/bin/python` lacks `ml_dtypes` and produces false failures on Phase 3+ tests.

---

## File structure

### New package
- `gpusim/mempool/__init__.py` — exports `MemoryPool`, `Allocation`
- `gpusim/mempool/allocation.py` — `Allocation` dataclass
- `gpusim/mempool/pool.py` — `MemoryPool`, `_FreeBlock`, `_next_pool_id`

### New examples (5 files each)
- `examples/mempool_basic/` (M1)
- `examples/mempool_multi_stream/` (M2)
- `examples/mempool_fragmentation/` (M3)
- `examples/mempool_train_step/` (M4)

### New tutorials (M5)
- `docs/tutorial/62-mempool-basic.md`
- `docs/tutorial/63-mempool-fragmentation.md`
- `docs/tutorial/64-mempool-multi-stream.md`
- `docs/tutorial/65-mempool-train-step.md`

### New tests
- `tests/unit/mempool/__init__.py`
- `tests/unit/mempool/test_pool_basic.py` (M1)
- `tests/unit/mempool/test_pool_recorder.py` (M1)
- `tests/unit/mempool/test_pool_cross_stream.py` (M2)
- `tests/unit/mempool/test_stream_convenience.py` (M2)
- `tests/unit/mempool/test_pool_trim.py` (M3)
- `tests/unit/mempool/test_pool_best_fit.py` (M3)
- `tests/unit/mempool/test_allocation_view.py` (M4)
- `tests/unit/analysis/test_phase16_metrics.py` (M4)
- `tests/parity/test_mempool_basic.py` (M1)
- `tests/parity/test_mempool_multi_stream.py` (M2)
- `tests/parity/test_mempool_fragmentation.py` (M3)
- `tests/parity/test_mempool_train_step.py` (M4)
- `tests/microbench/test_phase16_facts.py` (M5)
- `tests/microbench/test_phase16_runtime.py` (M5)
- `tests/parity/test_phase1_15_examples_unchanged.py` (M5, replaces `test_phase1_14_examples_unchanged.py`)

### Modified files
- `gpusim/api.py` — `Stream.malloc_async`, `Stream.free_async` convenience methods
- `gpusim/trace/events.py` — `PoolAllocate`, `PoolFree`, `PoolGrow`, `PoolTrim` dataclasses
- `gpusim/trace/recorder.py` — 4 list slots + 4 recorder methods
- `gpusim/analysis/metrics.py` — 4 metric functions
- `README.md` — v16 capabilities row + detailed section

---

# M1 — Pool Core + Basic Example

## Task 1: `Allocation` dataclass + package skeleton

**Files:**
- Create: `gpusim/mempool/__init__.py`
- Create: `gpusim/mempool/allocation.py`
- Create: `tests/unit/mempool/__init__.py` (empty)
- Create: `tests/unit/mempool/test_pool_basic.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/mempool/test_pool_basic.py`:
```python
def test_allocation_holds_buf_and_metadata():
    from gpusim.mempool.allocation import Allocation
    import numpy as np
    buf = np.zeros(8, dtype=np.uint8)
    a = Allocation(ptr_id=1, n_bytes=8, buf=buf, pool=None,
                    alloc_stream_id=3, _slab_index=0, _byte_offset=0)
    assert a.ptr_id == 1
    assert a.n_bytes == 8
    assert a.buf is buf
    assert a.alloc_stream_id == 3
    assert a._slab_index == 0
    assert a._byte_offset == 0
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/pytest tests/unit/mempool/test_pool_basic.py -v`
Expected: `ModuleNotFoundError: gpusim.mempool.allocation`.

- [ ] **Step 3: Create files**

`gpusim/mempool/__init__.py`:
```python
from gpusim.mempool.allocation import Allocation
from gpusim.mempool.pool import MemoryPool

__all__ = ["Allocation", "MemoryPool"]
```

`gpusim/mempool/allocation.py`:
```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Allocation:
    ptr_id: int
    n_bytes: int
    buf: object                 # numpy ndarray view (typed)
    pool: object                # MemoryPool reference
    alloc_stream_id: int
    _slab_index: int
    _byte_offset: int
```

`tests/unit/mempool/__init__.py` — empty file.

(Note: `gpusim/mempool/__init__.py` imports `MemoryPool` which doesn't exist yet, so the import will fail. Defer the `MemoryPool` import to Task 2 — for now write `__init__.py` as just `from gpusim.mempool.allocation import Allocation` plus `__all__ = ["Allocation"]`.)

Revised `gpusim/mempool/__init__.py`:
```python
from gpusim.mempool.allocation import Allocation

__all__ = ["Allocation"]
```

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/pytest tests/unit/mempool/test_pool_basic.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/mempool/__init__.py gpusim/mempool/allocation.py tests/unit/mempool/__init__.py tests/unit/mempool/test_pool_basic.py
git commit -m "feat(mempool): Allocation dataclass + package skeleton (Phase 16 M1)"
```

---

## Task 2: `MemoryPool` core — malloc + free + grow + best-fit

**Files:**
- Create: `gpusim/mempool/pool.py`
- Modify: `gpusim/mempool/__init__.py` — export `MemoryPool`
- Test: `tests/unit/mempool/test_pool_basic.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/mempool/test_pool_basic.py`:
```python
def test_pool_first_malloc_grows():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 64)
    assert a.n_bytes == 64
    assert a.alloc_stream_id == s.stream_id
    assert pool.in_flight_bytes == 64
    assert pool.high_water_mark == 64
    assert len(pool.slabs) == 1


def test_pool_second_alloc_after_free_reuses_same_stream():
    """Free on stream A → next malloc on A immediately reuses."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a1 = pool.malloc_async(s, 64)
    pool.free_async(s, a1)
    assert pool.in_flight_bytes == 0
    a2 = pool.malloc_async(s, 64)
    # Same slab — no new growth
    assert len(pool.slabs) == 1
    assert a2._slab_index == a1._slab_index
    assert a2._byte_offset == a1._byte_offset


def test_pool_in_flight_decreases_on_free():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 32)
    assert pool.in_flight_bytes == 32
    pool.free_async(s, a)
    assert pool.in_flight_bytes == 0


def test_pool_high_water_mark_monotone():
    """high_water_mark only increases, never decreases."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a1 = pool.malloc_async(s, 100)
    a2 = pool.malloc_async(s, 200)   # in_flight = 300
    assert pool.high_water_mark == 300
    pool.free_async(s, a1)            # in_flight = 200
    assert pool.high_water_mark == 300
    pool.free_async(s, a2)            # in_flight = 0
    assert pool.high_water_mark == 300


def test_pool_id_unique():
    from gpusim.mempool.pool import MemoryPool
    p1 = MemoryPool()
    p2 = MemoryPool()
    assert p1._pool_id != p2._pool_id


def test_allocation_buf_is_numpy_view_zeros_initial():
    """Newly-grown slab is a fresh bytearray; the view starts as zeros."""
    import numpy as np
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 16, dtype=np.uint8)
    assert isinstance(a.buf, np.ndarray)
    assert a.buf.dtype == np.uint8
    assert a.buf.shape == (16,)
    assert (a.buf == 0).all()


def test_allocation_buf_writes_back_to_slab():
    """Mutating the view mutates the underlying bytearray; reuse sees writes (until the
    new owner overwrites — they are responsible for initialization)."""
    import numpy as np
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a1 = pool.malloc_async(s, 16, dtype=np.uint8)
    a1.buf[:] = 7
    pool.free_async(s, a1)
    a2 = pool.malloc_async(s, 16, dtype=np.uint8)
    # Same memory; not zeroed by the pool — caller's responsibility.
    assert (a2.buf == 7).all()


def test_pool_best_fit_picks_smallest_fitting():
    """Best-fit: among free blocks ≥ requested, pick the smallest."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a_small = pool.malloc_async(s, 32)
    a_med = pool.malloc_async(s, 64)
    a_large = pool.malloc_async(s, 128)
    # Free all
    pool.free_async(s, a_small)
    pool.free_async(s, a_med)
    pool.free_async(s, a_large)
    # Request 50 — should reuse the 64-byte block, not the 128-byte one
    a_new = pool.malloc_async(s, 50)
    assert a_new._slab_index == a_med._slab_index
    assert a_new._byte_offset == a_med._byte_offset
    assert a_new.n_bytes == 64    # block keeps its full size


def test_pool_best_fit_grows_when_no_block_fits():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a_small = pool.malloc_async(s, 32)
    pool.free_async(s, a_small)
    # Request 100 — no fit, grow
    a_new = pool.malloc_async(s, 100)
    assert len(pool.slabs) == 2
    assert a_new.n_bytes == 100
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/pytest tests/unit/mempool/test_pool_basic.py -v`
Expected: FAIL — `ModuleNotFoundError: gpusim.mempool.pool`.

- [ ] **Step 3: Create `gpusim/mempool/pool.py`**

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
    freed_at_count: int


@dataclass
class MemoryPool:
    release_threshold: int = 2**30
    _pool_id: int = field(default_factory=_next_pool_id)

    slabs: list = field(default_factory=list)              # list[bytearray | None]
    slab_n_bytes: list = field(default_factory=list)       # list[int]
    slab_in_use_bytes: list = field(default_factory=list)  # list[int]

    free_blocks_by_stream: dict = field(default_factory=lambda: defaultdict(list))

    in_flight_bytes: int = 0
    high_water_mark: int = 0
    _free_counter: int = 0
    _next_ptr_id: int = 0
    _recorder: object | None = None

    def malloc_async(self, stream, n_bytes: int, dtype=None):
        from gpusim.mempool.allocation import Allocation
        if dtype is None:
            dtype = np.uint8

        sid = stream.stream_id
        block, reused = self._pop_best_fit(self.free_blocks_by_stream[sid], n_bytes)
        if block is None:
            block, reused = self._pop_best_fit(self.free_blocks_by_stream[-1], n_bytes)
        if block is None:
            slab_idx = self._grow(n_bytes)
            block = _FreeBlock(n_bytes=n_bytes, slab_index=slab_idx,
                               byte_offset=0, freed_at_count=-1)
            self.slab_in_use_bytes[slab_idx] = n_bytes
            reused = False

        n_elements = block.n_bytes // np.dtype(dtype).itemsize
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
        return alloc

    def free_async(self, stream, allocation) -> None:
        sid = stream.stream_id
        self._free_counter += 1
        self.free_blocks_by_stream[sid].append(_FreeBlock(
            n_bytes=allocation.n_bytes,
            slab_index=allocation._slab_index,
            byte_offset=allocation._byte_offset,
            freed_at_count=self._free_counter,
        ))
        self.in_flight_bytes -= allocation.n_bytes

    def _pop_best_fit(self, blocks: list, n_bytes: int):
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
        return slab_idx
```

- [ ] **Step 4: Update `__init__.py` to export MemoryPool**

`gpusim/mempool/__init__.py`:
```python
from gpusim.mempool.allocation import Allocation
from gpusim.mempool.pool import MemoryPool

__all__ = ["Allocation", "MemoryPool"]
```

- [ ] **Step 5: Run to confirm pass**

Run: `.venv/bin/pytest tests/unit/mempool/test_pool_basic.py -v`
Expected: 10 passed (1 from Task 1 + 9 new).

- [ ] **Step 6: Commit**

```bash
git add gpusim/mempool/pool.py gpusim/mempool/__init__.py tests/unit/mempool/test_pool_basic.py
git commit -m "feat(mempool): MemoryPool core — malloc/free/grow + best-fit (Phase 16 M1)"
```

---

## Task 3: 4 trace events + recorder methods

**Files:**
- Modify: `gpusim/trace/events.py` — append 4 event dataclasses
- Modify: `gpusim/trace/recorder.py` — add 4 list slots + 4 methods
- Test: `tests/unit/mempool/test_pool_recorder.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/mempool/test_pool_recorder.py`:
```python
def test_recorder_pool_allocate_appends():
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.pool_allocate(pool_id=1, stream_id=0, n_bytes=64,
                       ptr_id=10, reused=False, cycle=0)
    assert len(rec.pool_allocate_events) == 1
    ev = rec.pool_allocate_events[0]
    assert ev.pool_id == 1
    assert ev.stream_id == 0
    assert ev.n_bytes == 64
    assert ev.ptr_id == 10
    assert ev.reused is False


def test_recorder_pool_free_appends():
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.pool_free(pool_id=1, stream_id=0, ptr_id=10, n_bytes=64, cycle=0)
    assert len(rec.pool_free_events) == 1
    ev = rec.pool_free_events[0]
    assert ev.pool_id == 1
    assert ev.ptr_id == 10
    assert ev.n_bytes == 64


def test_recorder_pool_grow_appends():
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.pool_grow(pool_id=1, n_bytes_added=128, cycle=0)
    assert len(rec.pool_grow_events) == 1
    assert rec.pool_grow_events[0].n_bytes_added == 128


def test_recorder_pool_trim_appends():
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.pool_trim(pool_id=1, n_bytes_released=256, cycle=0)
    assert len(rec.pool_trim_events) == 1
    assert rec.pool_trim_events[0].n_bytes_released == 256
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/pytest tests/unit/mempool/test_pool_recorder.py -v`
Expected: FAIL — `Recorder` has no `pool_allocate` etc.

- [ ] **Step 3a: Add 4 event dataclasses to `gpusim/trace/events.py`**

Append:
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

- [ ] **Step 3b: Add 4 list slots + 4 methods to `Recorder`**

In `gpusim/trace/recorder.py`, in `Recorder.__init__`, append:
```python
        self.pool_allocate_events: list = []
        self.pool_free_events: list = []
        self.pool_grow_events: list = []
        self.pool_trim_events: list = []
```

Append methods at end of `Recorder` class:
```python
    def pool_allocate(self, *, pool_id: int, stream_id: int, n_bytes: int,
                       ptr_id: int, reused: bool, cycle: int) -> None:
        from gpusim.trace.events import PoolAllocate
        self.pool_allocate_events.append(PoolAllocate(
            pool_id=pool_id, stream_id=stream_id, n_bytes=n_bytes,
            ptr_id=ptr_id, reused=reused, cycle=cycle,
        ))

    def pool_free(self, *, pool_id: int, stream_id: int, ptr_id: int,
                    n_bytes: int, cycle: int) -> None:
        from gpusim.trace.events import PoolFree
        self.pool_free_events.append(PoolFree(
            pool_id=pool_id, stream_id=stream_id, ptr_id=ptr_id,
            n_bytes=n_bytes, cycle=cycle,
        ))

    def pool_grow(self, *, pool_id: int, n_bytes_added: int, cycle: int) -> None:
        from gpusim.trace.events import PoolGrow
        self.pool_grow_events.append(PoolGrow(
            pool_id=pool_id, n_bytes_added=n_bytes_added, cycle=cycle,
        ))

    def pool_trim(self, *, pool_id: int, n_bytes_released: int, cycle: int) -> None:
        from gpusim.trace.events import PoolTrim
        self.pool_trim_events.append(PoolTrim(
            pool_id=pool_id, n_bytes_released=n_bytes_released, cycle=cycle,
        ))
```

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/pytest tests/unit/mempool/test_pool_recorder.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/trace/events.py gpusim/trace/recorder.py tests/unit/mempool/test_pool_recorder.py
git commit -m "feat(trace): PoolAllocate/Free/Grow/Trim events + recorder methods (Phase 16 M1)"
```

---

## Task 4: Wire recorder into MemoryPool

**Files:**
- Modify: `gpusim/mempool/pool.py` — emit events from malloc_async / free_async / _grow
- Test: `tests/unit/mempool/test_pool_recorder.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/mempool/test_pool_recorder.py`:
```python
def test_pool_emits_allocate_event_on_malloc():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.trace.recorder import Recorder
    from gpusim.api import Stream
    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()
    pool.malloc_async(s, 64)
    assert len(rec.pool_allocate_events) == 1
    assert rec.pool_allocate_events[0].n_bytes == 64
    assert rec.pool_allocate_events[0].reused is False


def test_pool_emits_grow_event_on_first_alloc():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.trace.recorder import Recorder
    from gpusim.api import Stream
    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()
    pool.malloc_async(s, 64)
    assert len(rec.pool_grow_events) == 1
    assert rec.pool_grow_events[0].n_bytes_added == 64


def test_pool_emits_reused_true_on_second_alloc_after_free():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.trace.recorder import Recorder
    from gpusim.api import Stream
    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()
    a1 = pool.malloc_async(s, 64)
    pool.free_async(s, a1)
    pool.malloc_async(s, 64)
    assert len(rec.pool_allocate_events) == 2
    assert rec.pool_allocate_events[0].reused is False
    assert rec.pool_allocate_events[1].reused is True


def test_pool_emits_free_event():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.trace.recorder import Recorder
    from gpusim.api import Stream
    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()
    a = pool.malloc_async(s, 32)
    pool.free_async(s, a)
    assert len(rec.pool_free_events) == 1
    assert rec.pool_free_events[0].n_bytes == 32
    assert rec.pool_free_events[0].ptr_id == a.ptr_id
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/pytest tests/unit/mempool/test_pool_recorder.py -v`
Expected: 4 new tests fail — pool doesn't emit events yet.

- [ ] **Step 3: Add event emission to `MemoryPool`**

Edit `gpusim/mempool/pool.py`:

In `malloc_async`, after computing `alloc` and updating high_water_mark, before `return alloc`, add:
```python
        if self._recorder is not None:
            self._recorder.pool_allocate(
                pool_id=self._pool_id, stream_id=sid,
                n_bytes=block.n_bytes, ptr_id=ptr_id, reused=reused,
                cycle=0,
            )
```

In `free_async`, at end of method, add:
```python
        if self._recorder is not None:
            self._recorder.pool_free(
                pool_id=self._pool_id, stream_id=sid,
                ptr_id=allocation.ptr_id, n_bytes=allocation.n_bytes,
                cycle=0,
            )
```

In `_grow`, after appending to slab lists, before `return slab_idx`, add:
```python
        if self._recorder is not None:
            self._recorder.pool_grow(
                pool_id=self._pool_id, n_bytes_added=n_bytes, cycle=0,
            )
```

Updated full methods:
```python
    def malloc_async(self, stream, n_bytes: int, dtype=None):
        from gpusim.mempool.allocation import Allocation
        if dtype is None:
            dtype = np.uint8

        sid = stream.stream_id
        block, reused = self._pop_best_fit(self.free_blocks_by_stream[sid], n_bytes)
        if block is None:
            block, reused = self._pop_best_fit(self.free_blocks_by_stream[-1], n_bytes)
        if block is None:
            slab_idx = self._grow(n_bytes)
            block = _FreeBlock(n_bytes=n_bytes, slab_index=slab_idx,
                               byte_offset=0, freed_at_count=-1)
            self.slab_in_use_bytes[slab_idx] = n_bytes
            reused = False

        n_elements = block.n_bytes // np.dtype(dtype).itemsize
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

    def free_async(self, stream, allocation) -> None:
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

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/pytest tests/unit/mempool/test_pool_recorder.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/mempool/pool.py tests/unit/mempool/test_pool_recorder.py
git commit -m "feat(mempool): emit PoolAllocate/Free/Grow trace events from MemoryPool (Phase 16 M1)"
```

---

## Task 5: `examples/mempool_basic/` + parity test

**Files:**
- Create: `examples/mempool_basic/{__init__.py, kernel.ptx, reference.py, run.py, README.md}`
- Create: `tests/parity/test_mempool_basic.py`

- [ ] **Step 1: Write failing parity test**

Create `tests/parity/test_mempool_basic.py`:
```python
import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_basic"


def test_mempool_basic_reuse_rate():
    """4 alloc/free cycles → first grows pool, next 3 reuse same block."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    from gpusim.trace.recorder import Recorder

    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()

    for _ in range(4):
        a = pool.malloc_async(s, 1024)
        a.buf[:] = 9
        pool.free_async(s, a)

    # 4 allocates: 1 fresh + 3 reused
    assert len(rec.pool_allocate_events) == 4
    reused = sum(1 for ev in rec.pool_allocate_events if ev.reused)
    assert reused == 3
    assert len(pool.slabs) == 1


def test_mempool_basic_runs():
    """run.py exits 0."""
    import subprocess, sys
    res = subprocess.run([sys.executable, str(_DIR / "run.py")],
                         capture_output=True, timeout=30)
    assert res.returncode == 0, res.stderr.decode()[-500:]
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/pytest tests/parity/test_mempool_basic.py -v`
Expected: FAIL — example files missing.

- [ ] **Step 3: Create example files**

`examples/mempool_basic/__init__.py` — empty.

`examples/mempool_basic/kernel.ptx`:
```
.visible .entry inc(.param .u64 OUT)
{
    .reg .u64 %rd<5>;
    .reg .u32 %r<5>;

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

`examples/mempool_basic/reference.py`:
```python
import numpy as np


def reference(n: int = 32):
    return np.ones(n, dtype=np.uint32)
```

`examples/mempool_basic/run.py`:
```python
import numpy as np
from gpusim.mempool.pool import MemoryPool
from gpusim.api import Stream


def main():
    pool = MemoryPool()
    s = Stream()
    print(f"pool {pool._pool_id} created, release_threshold={pool.release_threshold}")

    # 4 alloc/free cycles of 1024 bytes
    for i in range(4):
        a = pool.malloc_async(s, 1024)
        a.buf[:] = i + 1                # touch the buffer
        pool.free_async(s, a)
        print(f"iter {i}: in_flight={pool.in_flight_bytes}, "
                f"high_water={pool.high_water_mark}, slabs={len(pool.slabs)}")

    print(f"final high_water_mark = {pool.high_water_mark} bytes "
            f"(expected 1024 — one slab reused 4x)")


if __name__ == "__main__":
    main()
```

`examples/mempool_basic/README.md`:
```markdown
# mempool_basic — Phase 16

Single stream, single pool. Alloc → free 4 times of 1024 bytes each.

Demonstrates:
- First malloc grows the pool by 1024 bytes (1 slab).
- Each subsequent malloc reuses the freed block — 0 new growth, reuse rate 75%.
- `high_water_mark` stays at 1024 bytes throughout.

## Run
```bash
python run.py
```
```

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/pytest tests/parity/test_mempool_basic.py -v`
Expected: 2 passed.

Sanity:
```bash
.venv/bin/python examples/mempool_basic/run.py
```
Expected: prints in_flight=0 each iter, slabs=1 throughout, final high_water_mark=1024.

- [ ] **Step 5: Commit**

```bash
git add examples/mempool_basic/ tests/parity/test_mempool_basic.py
git commit -m "feat(examples): mempool_basic — single-stream alloc/free reuse (Phase 16 M1)"
```

---

## Task 6: Tag M1

- [ ] **Step 1: Run full non-slow suite**

Run: `.venv/bin/pytest -m "not slow" -q`
Expected: ~787-790 passed (Phase 15 baseline 770 + ~17-20 from M1 unit/parity).

- [ ] **Step 2: Tag**

```bash
git tag M1-phase16-complete
```

- [ ] **Step 3: Verify**

Run: `git tag -l 'M1-phase16-complete'`
Expected: `M1-phase16-complete`

---

# M2 — Cross-Stream + Multi-Stream Example

## Task 7: `synchronize_stream` + cross-stream pool

**Files:**
- Modify: `gpusim/mempool/pool.py` — add `synchronize_stream` method
- Test: `tests/unit/mempool/test_pool_cross_stream.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/mempool/test_pool_cross_stream.py`:
```python
def test_cross_stream_alloc_without_sync_grows_pool():
    """Free on stream A, malloc on stream B without sync → grow."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    sA = Stream()
    sB = Stream()
    a = pool.malloc_async(sA, 64)
    pool.free_async(sA, a)
    # Without sync, B cannot reach A's free list
    pool.malloc_async(sB, 64)
    assert len(pool.slabs) == 2


def test_synchronize_stream_promotes_blocks_to_cross_stream_pool():
    """After synchronize_stream(sA), sB.malloc reuses the block."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    sA = Stream()
    sB = Stream()
    a = pool.malloc_async(sA, 64)
    pool.free_async(sA, a)
    assert len(pool.free_blocks_by_stream[sA.stream_id]) == 1
    pool.synchronize_stream(sA)
    assert len(pool.free_blocks_by_stream[sA.stream_id]) == 0
    assert len(pool.free_blocks_by_stream[-1]) == 1
    # Now sB can reuse
    pool.malloc_async(sB, 64)
    assert len(pool.slabs) == 1


def test_synchronize_stream_idempotent_when_nothing_to_promote():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    pool.synchronize_stream(s)    # no error, no-op
    assert len(pool.free_blocks_by_stream[-1]) == 0


def test_same_stream_reuse_takes_priority_over_cross_stream():
    """If both per-stream and cross-stream blocks fit, prefer same-stream."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    sA = Stream()
    sB = Stream()
    # Stream A allocates and frees, then synchronizes (block goes to cross-stream pool)
    a = pool.malloc_async(sA, 64)
    pool.free_async(sA, a)
    pool.synchronize_stream(sA)
    # Stream B allocates and frees (B's block is in B's per-stream list)
    b = pool.malloc_async(sB, 64)
    assert len(pool.slabs) == 1   # B reused A's promoted block
    pool.free_async(sB, b)
    # Now another B malloc should pull from B's per-stream list, not cross-stream
    pool.malloc_async(sB, 64)
    assert len(pool.slabs) == 1   # still 1, but reuse happened
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/pytest tests/unit/mempool/test_pool_cross_stream.py -v`
Expected: tests fail — `synchronize_stream` does not exist; cross-stream test grows pool but the reuse-test expects 1 slab.

(The first test `test_cross_stream_alloc_without_sync_grows_pool` may already pass — same-stream lookup falls through to cross-stream pool which is empty, so a new slab is allocated. Confirm.)

- [ ] **Step 3: Add `synchronize_stream` method**

Edit `gpusim/mempool/pool.py` — add method to `MemoryPool` class (after `free_async`):
```python
    def synchronize_stream(self, stream) -> None:
        """Promote all per-stream free blocks for this stream into the cross-stream pool.
        Phase 16: explicit promotion lets later mallocs on other streams reuse these blocks."""
        sid = stream.stream_id
        promoted = self.free_blocks_by_stream.pop(sid, [])
        if promoted:
            self.free_blocks_by_stream[-1].extend(promoted)
```

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/pytest tests/unit/mempool/test_pool_cross_stream.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/mempool/pool.py tests/unit/mempool/test_pool_cross_stream.py
git commit -m "feat(mempool): synchronize_stream promotes blocks to cross-stream pool (Phase 16 M2)"
```

---

## Task 8: `Stream.malloc_async` / `free_async` convenience methods

**Files:**
- Modify: `gpusim/api.py` — add 2 thin methods on `Stream`
- Test: `tests/unit/mempool/test_stream_convenience.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/mempool/test_stream_convenience.py`:
```python
def test_stream_malloc_async_forwards_to_pool():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = s.malloc_async(pool, 32)
    assert a.n_bytes == 32
    assert a.alloc_stream_id == s.stream_id


def test_stream_free_async_forwards_to_pool():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = s.malloc_async(pool, 32)
    assert pool.in_flight_bytes == 32
    s.free_async(pool, a)
    assert pool.in_flight_bytes == 0


def test_stream_malloc_async_dtype_arg_passthrough():
    import numpy as np
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = s.malloc_async(pool, 16, dtype=np.float32)
    assert a.buf.dtype == np.float32
    assert a.buf.shape == (4,)    # 16 / 4 = 4 elements
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/pytest tests/unit/mempool/test_stream_convenience.py -v`
Expected: FAIL — Stream has no `malloc_async`.

- [ ] **Step 3: Add convenience methods to `Stream`**

Edit `gpusim/api.py` — locate `Stream` class (around line 427-494). After `is_idle()` (the last method in `Stream`), add:
```python
    def malloc_async(self, pool, n_bytes: int, dtype=None):
        """Allocate from pool on this stream (Phase 16)."""
        return pool.malloc_async(self, n_bytes, dtype=dtype)

    def free_async(self, pool, allocation) -> None:
        """Return allocation to pool on this stream (Phase 16)."""
        pool.free_async(self, allocation)
```

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/pytest tests/unit/mempool/test_stream_convenience.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/api.py tests/unit/mempool/test_stream_convenience.py
git commit -m "feat(stream): malloc_async + free_async convenience methods (Phase 16 M2)"
```

---

## Task 9: `examples/mempool_multi_stream/` + parity test

**Files:**
- Create: `examples/mempool_multi_stream/{__init__.py, kernel.ptx, reference.py, run.py, README.md}`
- Create: `tests/parity/test_mempool_multi_stream.py`

- [ ] **Step 1: Write failing parity test**

Create `tests/parity/test_mempool_multi_stream.py`:
```python
import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_multi_stream"


def test_mempool_multi_stream_sync_promotes_block():
    """sA alloc/free + synchronize_stream → sB malloc reuses."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream

    pool = MemoryPool()
    sA = Stream()
    sB = Stream()

    # Without synchronize_stream, sB grows the pool
    a = sA.malloc_async(pool, 256)
    sA.free_async(pool, a)
    sB.malloc_async(pool, 256)
    assert len(pool.slabs) == 2

    # Reset for the synchronized half
    pool2 = MemoryPool()
    sA2 = Stream()
    sB2 = Stream()
    a2 = sA2.malloc_async(pool2, 256)
    sA2.free_async(pool2, a2)
    pool2.synchronize_stream(sA2)
    sB2.malloc_async(pool2, 256)
    assert len(pool2.slabs) == 1


def test_mempool_multi_stream_runs():
    import subprocess, sys
    res = subprocess.run([sys.executable, str(_DIR / "run.py")],
                         capture_output=True, timeout=30)
    assert res.returncode == 0, res.stderr.decode()[-500:]
```

- [ ] **Step 2: Run to confirm fail**

Expected: example files missing.

- [ ] **Step 3: Create example files**

`examples/mempool_multi_stream/__init__.py` — empty.

`examples/mempool_multi_stream/kernel.ptx`:
```
.visible .entry inc(.param .u64 OUT)
{
    .reg .u64 %rd<5>;
    .reg .u32 %r<5>;

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

`examples/mempool_multi_stream/reference.py`:
```python
def reference():
    return {"slabs_unsynced": 2, "slabs_synced": 1}
```

`examples/mempool_multi_stream/run.py`:
```python
from gpusim.mempool.pool import MemoryPool
from gpusim.api import Stream


def main():
    # Without synchronize_stream
    pool_a = MemoryPool()
    sA = Stream(); sB = Stream()
    a = sA.malloc_async(pool_a, 256)
    sA.free_async(pool_a, a)
    sB.malloc_async(pool_a, 256)    # cannot reuse — grows
    print(f"unsynced: {len(pool_a.slabs)} slabs (expected 2)")

    # With synchronize_stream
    pool_b = MemoryPool()
    sC = Stream(); sD = Stream()
    a2 = sC.malloc_async(pool_b, 256)
    sC.free_async(pool_b, a2)
    pool_b.synchronize_stream(sC)
    sD.malloc_async(pool_b, 256)    # reuses
    print(f"synced: {len(pool_b.slabs)} slabs (expected 1)")


if __name__ == "__main__":
    main()
```

`examples/mempool_multi_stream/README.md`:
```markdown
# mempool_multi_stream — Phase 16

Two streams sharing a pool. Demonstrates that cross-stream reuse requires
explicit `pool.synchronize_stream(stream)` to promote per-stream free blocks
into the cross-stream pool.

## Run
```bash
python run.py
```
```

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/pytest tests/parity/test_mempool_multi_stream.py -v`
Expected: 2 passed.

Sanity:
```bash
.venv/bin/python examples/mempool_multi_stream/run.py
```
Expected: prints `unsynced: 2 slabs` and `synced: 1 slabs`.

- [ ] **Step 5: Commit**

```bash
git add examples/mempool_multi_stream/ tests/parity/test_mempool_multi_stream.py
git commit -m "feat(examples): mempool_multi_stream — synchronize_stream cross-stream reuse (Phase 16 M2)"
```

---

## Task 10: Tag M2

- [ ] **Step 1: Run full non-slow suite**

Run: `.venv/bin/pytest -m "not slow" -q`
Expected: ~793-797 passed.

- [ ] **Step 2: Tag**

```bash
git tag M2-phase16-complete
```

---

# M3 — trim_to + Fragmentation Example

## Task 11: `trim_to` + `release_threshold`

**Files:**
- Modify: `gpusim/mempool/pool.py` — add `trim_to` method
- Test: `tests/unit/mempool/test_pool_trim.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/mempool/test_pool_trim.py`:
```python
def test_trim_releases_fully_free_slab_above_threshold():
    """If 2 slabs are fully free and total exceeds threshold, slabs are released."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()

    a1 = pool.malloc_async(s, 1000)
    a2 = pool.malloc_async(s, 2000)
    pool.free_async(s, a1)
    pool.free_async(s, a2)
    # 2 slabs (1000 + 2000 = 3000 bytes), all free.
    # release_threshold=500 → keep 500 bytes minimum after release.
    released = pool.trim_to(release_threshold_bytes=500)
    # Slabs are released individually if total - this_slab >= 500.
    # Releasing 1000 leaves 2000 ≥ 500 → ok. Releasing 2000 leaves 1000 ≥ 500 → ok.
    # Both released.
    assert released == 3000
    assert all(s is None for s in pool.slabs)


def test_trim_keeps_slab_when_release_would_drop_below_threshold():
    """Release threshold prevents dropping below the floor."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()

    a1 = pool.malloc_async(s, 1000)
    a2 = pool.malloc_async(s, 2000)
    pool.free_async(s, a1)
    pool.free_async(s, a2)
    # threshold = 2500: releasing the 1000 leaves 2000 < 2500 → keep 1000;
    # releasing the 2000 leaves 1000 < 2500 → keep 2000. Neither released.
    released = pool.trim_to(release_threshold_bytes=2500)
    assert released == 0
    assert all(s is not None for s in pool.slabs)


def test_trim_skips_slabs_with_in_use_bytes():
    """A slab with at least one live allocation cannot be released."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()

    a1 = pool.malloc_async(s, 1000)
    a2 = pool.malloc_async(s, 2000)
    pool.free_async(s, a2)        # only slab 1 is fully free
    released = pool.trim_to(release_threshold_bytes=0)
    # Only slab 1 (2000 bytes) released; slab 0 has a1 in use.
    assert released == 2000
    assert pool.slabs[0] is not None
    assert pool.slabs[1] is None


def test_trim_emits_pool_trim_event():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.trace.recorder import Recorder
    from gpusim.api import Stream
    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()
    a = pool.malloc_async(s, 500)
    pool.free_async(s, a)
    pool.trim_to(release_threshold_bytes=0)
    assert len(rec.pool_trim_events) == 1
    assert rec.pool_trim_events[0].n_bytes_released == 500


def test_trim_no_event_when_nothing_released():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.trace.recorder import Recorder
    from gpusim.api import Stream
    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()
    pool.malloc_async(s, 500)        # in-use; cannot release
    pool.trim_to(release_threshold_bytes=0)
    assert len(rec.pool_trim_events) == 0


def test_trim_removes_freed_blocks_pointing_to_released_slabs():
    """Free list must not contain dangling references after trim."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 500)
    pool.free_async(s, a)
    assert len(pool.free_blocks_by_stream[s.stream_id]) == 1
    pool.trim_to(release_threshold_bytes=0)
    # Slab released → freed block pointing to it should also be gone.
    assert len(pool.free_blocks_by_stream[s.stream_id]) == 0
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/pytest tests/unit/mempool/test_pool_trim.py -v`
Expected: FAIL — `trim_to` does not exist.

- [ ] **Step 3: Add `trim_to` method**

Edit `gpusim/mempool/pool.py` — add to `MemoryPool` class (after `synchronize_stream`):
```python
    def trim_to(self, release_threshold_bytes: int) -> int:
        """Release fully-free slabs while keeping total slab bytes ≥ release_threshold_bytes.
        Returns total bytes released."""
        from collections import defaultdict
        per_slab_free = defaultdict(int)
        for stream_key, blocks in self.free_blocks_by_stream.items():
            for b in blocks:
                per_slab_free[b.slab_index] += b.n_bytes

        released = 0
        slabs_to_release = []
        for slab_idx in range(len(self.slabs)):
            if self.slabs[slab_idx] is None:
                continue                              # already released
            if per_slab_free[slab_idx] != self.slab_n_bytes[slab_idx]:
                continue                              # has live allocations
            current_total = sum(
                self.slab_n_bytes[i] for i in range(len(self.slabs))
                if self.slabs[i] is not None and i not in slabs_to_release
            )
            if current_total - self.slab_n_bytes[slab_idx] >= release_threshold_bytes:
                slabs_to_release.append(slab_idx)
                released += self.slab_n_bytes[slab_idx]

        if slabs_to_release:
            for stream_key in list(self.free_blocks_by_stream.keys()):
                self.free_blocks_by_stream[stream_key] = [
                    b for b in self.free_blocks_by_stream[stream_key]
                    if b.slab_index not in slabs_to_release
                ]
            for slab_idx in slabs_to_release:
                self.slabs[slab_idx] = None
                self.slab_n_bytes[slab_idx] = 0
                self.slab_in_use_bytes[slab_idx] = 0

        if self._recorder is not None and released > 0:
            self._recorder.pool_trim(
                pool_id=self._pool_id, n_bytes_released=released, cycle=0,
            )
        return released
```

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/pytest tests/unit/mempool/test_pool_trim.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/mempool/pool.py tests/unit/mempool/test_pool_trim.py
git commit -m "feat(mempool): trim_to + release_threshold + PoolTrim event (Phase 16 M3)"
```

---

## Task 12: Best-fit deeper tests + tiebreak by oldest free

**Files:**
- Test: `tests/unit/mempool/test_pool_best_fit.py` (new)

- [ ] **Step 1: Write tests**

Create `tests/unit/mempool/test_pool_best_fit.py`:
```python
def test_best_fit_among_multiple_candidates_picks_smallest():
    """3 free blocks ≥ requested → smallest is picked."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a64 = pool.malloc_async(s, 64)
    a128 = pool.malloc_async(s, 128)
    a256 = pool.malloc_async(s, 256)
    pool.free_async(s, a256)
    pool.free_async(s, a128)
    pool.free_async(s, a64)
    a_new = pool.malloc_async(s, 50)    # smallest fit is a64
    assert a_new._slab_index == a64._slab_index


def test_best_fit_tiebreak_oldest_free_when_sizes_equal():
    """Two equal-size free blocks → oldest free wins."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a1 = pool.malloc_async(s, 64)
    a2 = pool.malloc_async(s, 64)
    pool.free_async(s, a1)               # freed first (oldest)
    pool.free_async(s, a2)               # freed second
    a_new = pool.malloc_async(s, 64)
    # a1 was freed first → its block should be reused
    assert a_new._slab_index == a1._slab_index


def test_best_fit_block_keeps_full_size_when_oversized():
    """Requesting 50 from a 64-byte block returns a 64-byte allocation (no splitting)."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 64)
    pool.free_async(s, a)
    a_new = pool.malloc_async(s, 50)
    assert a_new.n_bytes == 64    # block size, not request size
```

- [ ] **Step 2: Run to confirm pass**

Run: `.venv/bin/pytest tests/unit/mempool/test_pool_best_fit.py -v`
Expected: 3 passed (logic from Task 2 already supports best-fit).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/mempool/test_pool_best_fit.py
git commit -m "test(mempool): best-fit selection + oldest-free tiebreak coverage (Phase 16 M3)"
```

---

## Task 13: `examples/mempool_fragmentation/` + parity test

**Files:**
- Create: `examples/mempool_fragmentation/{__init__.py, kernel.ptx, reference.py, run.py, README.md}`
- Create: `tests/parity/test_mempool_fragmentation.py`

- [ ] **Step 1: Write failing parity test**

Create `tests/parity/test_mempool_fragmentation.py`:
```python
import pathlib


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_fragmentation"


def test_mempool_fragmentation_best_fit_reuses_correct_blocks():
    """After freeing all, re-alloc 1024 + 2048 reuses exact size matches."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream

    pool = MemoryPool()
    s = Stream()
    a1024 = pool.malloc_async(s, 1024)
    a2048 = pool.malloc_async(s, 2048)
    a4096 = pool.malloc_async(s, 4096)
    a1024b = pool.malloc_async(s, 1024)
    a2048b = pool.malloc_async(s, 2048)
    assert len(pool.slabs) == 5

    for a in (a1024, a2048, a4096, a1024b, a2048b):
        pool.free_async(s, a)

    # Re-alloc 1024 → reuses one of the 1024 blocks (oldest freed)
    new1024 = pool.malloc_async(s, 1024)
    assert new1024._slab_index == a1024._slab_index    # oldest 1024 wins

    # Re-alloc 2048 → reuses oldest 2048
    new2048 = pool.malloc_async(s, 2048)
    assert new2048._slab_index == a2048._slab_index

    assert len(pool.slabs) == 5    # no new growth


def test_mempool_fragmentation_runs():
    import subprocess, sys
    res = subprocess.run([sys.executable, str(_DIR / "run.py")],
                         capture_output=True, timeout=30)
    assert res.returncode == 0, res.stderr.decode()[-500:]
```

- [ ] **Step 2: Run to confirm fail**

Expected: example missing.

- [ ] **Step 3: Create example files**

`examples/mempool_fragmentation/__init__.py` — empty.

`examples/mempool_fragmentation/kernel.ptx`:
```
.visible .entry inc(.param .u64 OUT)
{
    .reg .u64 %rd<5>;
    .reg .u32 %r<5>;

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

`examples/mempool_fragmentation/reference.py`:
```python
def reference():
    return {"slabs_after_realloc": 5}
```

`examples/mempool_fragmentation/run.py`:
```python
from gpusim.mempool.pool import MemoryPool
from gpusim.api import Stream


def main():
    pool = MemoryPool()
    s = Stream()
    sizes = [1024, 2048, 4096, 1024, 2048]
    allocs = [pool.malloc_async(s, n) for n in sizes]
    print(f"after 5 alloc: {len(pool.slabs)} slabs (expected 5)")

    for a in allocs:
        pool.free_async(s, a)

    a1 = pool.malloc_async(s, 1024)
    a2 = pool.malloc_async(s, 2048)
    print(f"after re-alloc 1024+2048: {len(pool.slabs)} slabs (expected 5, no growth)")
    print(f"reused 1024 from slab {a1._slab_index}, 2048 from slab {a2._slab_index}")

    # Show trim behavior
    for a in (a1, a2):
        pool.free_async(s, a)
    released = pool.trim_to(release_threshold_bytes=0)
    print(f"trim_to(0) released {released} bytes (all 5 slabs)")


if __name__ == "__main__":
    main()
```

`examples/mempool_fragmentation/README.md`:
```markdown
# mempool_fragmentation — Phase 16

Allocate 5 blocks of mixed sizes [1024, 2048, 4096, 1024, 2048], free all,
then re-alloc 1024 and 2048 — best-fit picks the right blocks (no growth).

Then `trim_to(0)` releases all 5 fully-free slabs back to the OS.

## Run
```bash
python run.py
```
```

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/pytest tests/parity/test_mempool_fragmentation.py -v`
Expected: 2 passed.

Sanity:
```bash
.venv/bin/python examples/mempool_fragmentation/run.py
```
Expected: prints "5 slabs", "no growth", "released ... bytes".

- [ ] **Step 5: Commit**

```bash
git add examples/mempool_fragmentation/ tests/parity/test_mempool_fragmentation.py
git commit -m "feat(examples): mempool_fragmentation — best-fit + trim demo (Phase 16 M3)"
```

---

## Task 14: Tag M3

- [ ] **Step 1: Run full non-slow suite**

Run: `.venv/bin/pytest -m "not slow" -q`
Expected: ~803-808 passed.

- [ ] **Step 2: Tag**

```bash
git tag M3-phase16-complete
```

---

# M4 — 4 Metrics + Train-Step Example

## Task 15: 4 Phase 16 metrics

**Files:**
- Modify: `gpusim/analysis/metrics.py` — append 4 metrics
- Test: `tests/unit/analysis/test_phase16_metrics.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/analysis/test_phase16_metrics.py`:
```python
def test_pool_high_water_mark_zero_for_empty_recorder():
    from gpusim.analysis.metrics import pool_high_water_mark
    from gpusim.trace.recorder import Recorder
    assert pool_high_water_mark(Recorder(), pool_id=1) == 0


def test_pool_high_water_mark_tracks_peak():
    from gpusim.analysis.metrics import pool_high_water_mark
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.pool_allocate(pool_id=1, stream_id=0, n_bytes=100, ptr_id=0,
                       reused=False, cycle=0)
    rec.pool_allocate(pool_id=1, stream_id=0, n_bytes=200, ptr_id=1,
                       reused=False, cycle=10)   # peak = 300
    rec.pool_free(pool_id=1, stream_id=0, ptr_id=0, n_bytes=100, cycle=20)
    rec.pool_allocate(pool_id=1, stream_id=0, n_bytes=50, ptr_id=2,
                       reused=False, cycle=30)   # in_flight = 250 (still < 300)
    assert pool_high_water_mark(rec, pool_id=1) == 300


def test_pool_high_water_mark_per_pool():
    from gpusim.analysis.metrics import pool_high_water_mark
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.pool_allocate(pool_id=1, stream_id=0, n_bytes=100, ptr_id=0,
                       reused=False, cycle=0)
    rec.pool_allocate(pool_id=2, stream_id=0, n_bytes=500, ptr_id=1,
                       reused=False, cycle=0)
    assert pool_high_water_mark(rec, pool_id=1) == 100
    assert pool_high_water_mark(rec, pool_id=2) == 500


def test_pool_reuse_rate_zero_when_no_allocs():
    from gpusim.analysis.metrics import pool_reuse_rate
    from gpusim.trace.recorder import Recorder
    assert pool_reuse_rate(Recorder(), pool_id=1) == 0.0


def test_pool_reuse_rate_correctly_reports_fraction():
    from gpusim.analysis.metrics import pool_reuse_rate
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    for i in range(4):
        rec.pool_allocate(pool_id=1, stream_id=0, n_bytes=64, ptr_id=i,
                           reused=(i > 0), cycle=i*10)
    assert pool_reuse_rate(rec, pool_id=1) == 0.75


def test_pool_alloc_count():
    from gpusim.analysis.metrics import pool_alloc_count
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    for i in range(5):
        rec.pool_allocate(pool_id=1, stream_id=0, n_bytes=64, ptr_id=i,
                           reused=False, cycle=0)
    assert pool_alloc_count(rec, pool_id=1) == 5


def test_pool_release_total_bytes_sums_trim_events():
    from gpusim.analysis.metrics import pool_release_total_bytes
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.pool_trim(pool_id=1, n_bytes_released=1024, cycle=0)
    rec.pool_trim(pool_id=1, n_bytes_released=2048, cycle=10)
    rec.pool_trim(pool_id=2, n_bytes_released=500, cycle=20)
    assert pool_release_total_bytes(rec, pool_id=1) == 3072
    assert pool_release_total_bytes(rec, pool_id=2) == 500


def test_pool_release_total_bytes_zero_when_no_trim():
    from gpusim.analysis.metrics import pool_release_total_bytes
    from gpusim.trace.recorder import Recorder
    assert pool_release_total_bytes(Recorder(), pool_id=1) == 0
```

- [ ] **Step 2: Run to confirm fail**

Run: `.venv/bin/pytest tests/unit/analysis/test_phase16_metrics.py -v`
Expected: FAIL — metrics not defined.

- [ ] **Step 3: Append 4 metrics to `gpusim/analysis/metrics.py`**

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

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/pytest tests/unit/analysis/test_phase16_metrics.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add gpusim/analysis/metrics.py tests/unit/analysis/test_phase16_metrics.py
git commit -m "feat(analysis): 4 Phase 16 metrics — pool_high_water_mark, pool_reuse_rate, pool_alloc_count, pool_release_total_bytes"
```

---

## Task 16: Allocation view kernel-compatibility test

**Files:**
- Test: `tests/unit/mempool/test_allocation_view.py` (new)

- [ ] **Step 1: Write tests**

Create `tests/unit/mempool/test_allocation_view.py`:
```python
def test_allocation_buf_is_writable_typed_view():
    """Pool buffer is a numpy view that supports element write + readback."""
    import numpy as np
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 16, dtype=np.uint32)
    assert a.buf.shape == (4,)            # 16 bytes / 4 = 4 elements
    a.buf[:] = [1, 2, 3, 4]
    assert list(a.buf) == [1, 2, 3, 4]


def test_allocation_buf_writes_persist_through_free_and_realloc():
    """Pool does not zero on free; readers see prior writer's data until they overwrite."""
    import numpy as np
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a1 = pool.malloc_async(s, 16, dtype=np.uint8)
    a1.buf[:] = 0x55
    pool.free_async(s, a1)
    a2 = pool.malloc_async(s, 16, dtype=np.uint8)
    assert (a2.buf == 0x55).all()


def test_allocation_buf_works_as_kernel_param():
    """Allocation.buf is shape-compatible with kernel param expectations
    (numpy ndarray with .shape, .dtype, .data, indexing). We don't actually launch
    a kernel here — gpusim's launcher reads ndarrays via numpy attributes only."""
    import numpy as np
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 32, dtype=np.uint32)
    # Ensure ndarray protocol
    assert hasattr(a.buf, "shape")
    assert hasattr(a.buf, "dtype")
    assert hasattr(a.buf, "data")
    assert isinstance(a.buf, np.ndarray)
    # Indexable
    a.buf[0] = 42
    assert int(a.buf[0]) == 42
```

- [ ] **Step 2: Run**

Run: `.venv/bin/pytest tests/unit/mempool/test_allocation_view.py -v`
Expected: 3 passed (no implementation needed; covered by existing pool logic).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/mempool/test_allocation_view.py
git commit -m "test(mempool): Allocation.buf is a writable typed numpy view (Phase 16 M4)"
```

---

## Task 17: `examples/mempool_train_step/` + parity test

**Files:**
- Create: `examples/mempool_train_step/{__init__.py, kernel.ptx, reference.py, run.py, README.md}`
- Create: `tests/parity/test_mempool_train_step.py`

- [ ] **Step 1: Write failing parity test**

Create `tests/parity/test_mempool_train_step.py`:
```python
import pathlib


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_train_step"


def test_mempool_train_step_high_water_stabilizes():
    """Train-step pattern: each iter alloc + free; high_water == iter1 working set."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream

    pool = MemoryPool()
    s = Stream()

    for _ in range(10):
        act = s.malloc_async(pool, 8 * 1024)
        grad = s.malloc_async(pool, 8 * 1024)
        # ... in real training, kernel would touch these ...
        s.free_async(pool, act)
        s.free_async(pool, grad)
        pool.synchronize_stream(s)

    # high_water_mark should equal iter 1's peak (16 KB, two 8 KB blocks)
    assert pool.high_water_mark == 16 * 1024
    assert len(pool.slabs) == 2          # only 2 slabs, reused 9 more iters


def test_mempool_train_step_reuse_rate_high():
    """After 10 iters of alloc/free, reuse rate should be 18/20 = 0.9 (first 2 fresh, next 18 reused)."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.trace.recorder import Recorder
    from gpusim.api import Stream
    from gpusim.analysis.metrics import pool_reuse_rate, pool_alloc_count

    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()

    for _ in range(10):
        act = s.malloc_async(pool, 8 * 1024)
        grad = s.malloc_async(pool, 8 * 1024)
        s.free_async(pool, act)
        s.free_async(pool, grad)

    assert pool_alloc_count(rec, pool._pool_id) == 20
    assert pool_reuse_rate(rec, pool._pool_id) == 0.9    # 18 reused / 20 total


def test_mempool_train_step_runs():
    import subprocess, sys
    res = subprocess.run([sys.executable, str(_DIR / "run.py")],
                         capture_output=True, timeout=30)
    assert res.returncode == 0, res.stderr.decode()[-500:]
```

- [ ] **Step 2: Run to confirm fail**

Expected: example missing.

- [ ] **Step 3: Create example files**

`examples/mempool_train_step/__init__.py` — empty.

`examples/mempool_train_step/kernel.ptx`:
```
.visible .entry train_step(.param .u64 ACT, .param .u64 GRAD)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<6>;

    ld.param.u64 %rd0, [ACT];
    ld.param.u64 %rd1, [GRAD];

    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd2, %r1;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;

    mov.u32 %r2, 1;
    st.global.u32 [%rd3], %r2;
    st.global.u32 [%rd4], %r2;

    ret;
}
```

`examples/mempool_train_step/reference.py`:
```python
def reference():
    return {"high_water_mark_bytes": 16 * 1024, "iters": 10}
```

`examples/mempool_train_step/run.py`:
```python
import numpy as np
from gpusim.mempool.pool import MemoryPool
from gpusim.api import Stream
from gpusim.trace.recorder import Recorder
from gpusim.analysis.metrics import (
    pool_high_water_mark, pool_reuse_rate, pool_alloc_count,
)


def main():
    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()

    iters = 10
    for i in range(iters):
        act = s.malloc_async(pool, 8 * 1024, dtype=np.float32)
        grad = s.malloc_async(pool, 8 * 1024, dtype=np.float32)
        # Touch buffers (would be a real kernel in production)
        act.buf[:] = float(i)
        grad.buf[:] = float(-i)
        s.free_async(pool, act)
        s.free_async(pool, grad)
        pool.synchronize_stream(s)

    pid = pool._pool_id
    print(f"after {iters} train steps:")
    print(f"  high_water_mark = {pool_high_water_mark(rec, pid)} bytes "
            f"(expected {16 * 1024})")
    print(f"  reuse_rate      = {pool_reuse_rate(rec, pid):.2f} "
            f"(expected {18/20:.2f})")
    print(f"  alloc_count     = {pool_alloc_count(rec, pid)}")
    print(f"  slabs           = {len(pool.slabs)} "
            f"(expected 2 — first iter creates both, rest reuse)")


if __name__ == "__main__":
    main()
```

`examples/mempool_train_step/README.md`:
```markdown
# mempool_train_step — Phase 16

Toy training-step pattern: each iteration allocates an activation buffer (8 KB)
and a gradient buffer (8 KB), uses them, then frees both. After warmup, every
iteration reuses the same two blocks — `high_water_mark` plateaus at 16 KB,
reuse rate approaches 1.0.

## Run
```bash
python run.py
```
```

- [ ] **Step 4: Run to confirm pass**

Run: `.venv/bin/pytest tests/parity/test_mempool_train_step.py -v`
Expected: 3 passed.

Sanity:
```bash
.venv/bin/python examples/mempool_train_step/run.py
```
Expected: prints `high_water_mark = 16384`, `reuse_rate = 0.90`, `slabs = 2`.

- [ ] **Step 5: Commit**

```bash
git add examples/mempool_train_step/ tests/parity/test_mempool_train_step.py
git commit -m "feat(examples): mempool_train_step — alloc/free pattern, reuse converges (Phase 16 M4)"
```

---

## Task 18: Tag M4

- [ ] **Step 1: Run full non-slow suite**

Run: `.venv/bin/pytest -m "not slow" -q`
Expected: ~815-820 passed.

- [ ] **Step 2: Tag**

```bash
git tag M4-phase16-complete
```

---

# M5 — Tutorials + Microbench + Regression Rename + README + Ship

## Task 19: 4 tutorial chapters

**Files:**
- Create: `docs/tutorial/62-mempool-basic.md`
- Create: `docs/tutorial/63-mempool-fragmentation.md`
- Create: `docs/tutorial/64-mempool-multi-stream.md`
- Create: `docs/tutorial/65-mempool-train-step.md`

Use Phase 15 chapters (58-61) as the structural template: English body, Chinese subheadings `看模拟器` / `改一改` / `真机对照`, ~500-700 words each.

- [ ] **Step 1: Reference template**

Run: `cat docs/tutorial/58-stream-capture-basic.md`

- [ ] **Step 2: Create chapter 62**

Create `docs/tutorial/62-mempool-basic.md`:
```markdown
# 62 · Memory Pool Basic — Stream-Ordered alloc/free with Reuse

Phase 16 introduces `gpusim.mempool.MemoryPool` — a stream-ordered allocator
modeled after CUDA's `cudaMallocAsync` / `cudaFreeAsync` family. Each
allocation returns an `Allocation` whose `.buf` attribute is a typed numpy
view, so kernels accept it just like any other buffer.

This chapter walks the simplest pattern: one stream allocates and frees a
1 KB block four times. The first allocation grows the pool; the next three
reuse the freed block.

## What the example does

```python
from gpusim.mempool.pool import MemoryPool
from gpusim.api import Stream

pool = MemoryPool()
s = Stream()
for i in range(4):
    a = pool.malloc_async(s, 1024)
    a.buf[:] = i + 1                # write into the buffer
    pool.free_async(s, a)
print(pool.high_water_mark, len(pool.slabs))   # 1024 1
```

## 看模拟器

`malloc_async` first checks the per-stream free list (`free_blocks_by_stream[s.stream_id]`).
On the first iteration, the list is empty so `_grow` allocates a fresh
`bytearray(1024)` slab and returns a numpy view via `np.frombuffer`.

`free_async` pushes a `_FreeBlock(n_bytes=1024, slab_index=0, byte_offset=0,
freed_at_count=N)` onto the per-stream free list and decrements
`in_flight_bytes`.

On iteration 2, `malloc_async` finds the free block, pops it, and reuses the
same slab — no growth. `pool.slabs` stays length 1, `high_water_mark` stays
at 1024.

The recorder (when attached as `pool._recorder = rec`) emits a `PoolAllocate`
event each time, with `reused=False` for the first call and `reused=True`
afterward. The metric `pool_reuse_rate(recorder, pool_id)` reports the
fraction of `PoolAllocate.reused == True` events.

## 改一改

- Allocate two blocks of 1024 each before freeing. Now `pool.slabs == 2`,
  `high_water_mark == 2048`, and the next iter reuses the most recently freed
  block (oldest-free tiebreak).
- Pass `dtype=np.float32`: the `buf` view is shape `(256,)`, dtype `float32`,
  and reads/writes go through the same underlying `bytearray`.

## 真机对照

Real CUDA: `cudaMallocAsync(&ptr, n, stream)` and `cudaFreeAsync(ptr, stream)`.
Free on stream A makes the block reusable by subsequent allocations on stream A
immediately. PyTorch's caching allocator wraps these and adds heuristics for
block splitting and rounding (we keep best-fit unsplit). Phase 17 may add
splitting and the `cudaMemPoolReuseAllowOpportunistic` semantics.
```

- [ ] **Step 3: Create chapter 63**

Create `docs/tutorial/63-mempool-fragmentation.md`:
```markdown
# 63 · Memory Pool Fragmentation — Best-Fit + trim_to

When allocations of mixed sizes are freed and re-requested, the pool's
**best-fit** policy picks the smallest free block that fits. Among equal-size
blocks, the **oldest free** wins (FIFO tiebreak).

This chapter walks an example that:
1. Allocates 5 mixed-size blocks: `[1024, 2048, 4096, 1024, 2048]`.
2. Frees all 5.
3. Re-allocates 1024 + 2048 — best-fit picks the right blocks; pool does
   **not** grow.
4. Frees the new allocations and calls `pool.trim_to(0)` — all 5 fully-free
   slabs are released back to the OS.

## What the example does

```python
pool = MemoryPool(); s = Stream()
allocs = [pool.malloc_async(s, n) for n in (1024, 2048, 4096, 1024, 2048)]
for a in allocs: pool.free_async(s, a)
a1 = pool.malloc_async(s, 1024)        # reuses oldest 1024 (slot 0)
a2 = pool.malloc_async(s, 2048)        # reuses oldest 2048 (slot 1)
print(len(pool.slabs))                  # 5 — no new growth

for a in (a1, a2): pool.free_async(s, a)
released = pool.trim_to(release_threshold_bytes=0)
print(released)                         # full size of all 5 slabs
```

## 看模拟器

`_pop_best_fit` walks the candidate list, sorts by `(n_bytes, freed_at_count)`,
and pops the smallest-fit-then-oldest-free block. Returning the original block
size (no splitting) means a 50-byte request from a 64-byte free block yields
a 64-byte allocation — splitting is YAGNI for now.

`trim_to(release_threshold_bytes)` aggregates per-slab free bytes; only slabs
with `per_slab_free == slab_size` are eligible. It releases each eligible slab
individually as long as `current_total - this_slab >= release_threshold_bytes`.
After release, slabs in `pool.slabs` become `None` (preserving indices) and
free blocks pointing to them are pruned from `free_blocks_by_stream`.

## 改一改

- Set `release_threshold_bytes` higher than total slab bytes — `trim_to`
  releases nothing; `pool.slabs` stays untouched.
- Hold one allocation back unfreed before calling `trim_to`. That slab is not
  fully free, so it survives.
- Free in a different order to see oldest-free tiebreak in action.

## 真机对照

CUDA: `cudaMemPoolTrimTo(pool, minBytesToKeep)` releases unused memory above
the threshold. The matching attribute is
`cudaMemPoolAttrReleaseThreshold`. PyTorch exposes this as
`torch.cuda.empty_cache()`; JAX exposes it through XLA's allocator. The
release-threshold semantics in Phase 16 mirror the CUDA contract — keep the
pool above the floor, drop slabs above it.
```

- [ ] **Step 4: Create chapter 64**

Create `docs/tutorial/64-mempool-multi-stream.md`:
```markdown
# 64 · Memory Pool Multi-Stream — Cross-Stream Reuse via synchronize_stream

The stream-ordered contract: a block freed on stream A is **only** reusable by
another stream after a synchronization point. CUDA expresses this implicitly
via stream events; Phase 16 makes the synchronization explicit with
`pool.synchronize_stream(stream)`. This keeps the simulator honest about
when cross-stream reuse is and isn't safe.

## What the example does

```python
pool = MemoryPool()
sA, sB = Stream(), Stream()

a = sA.malloc_async(pool, 256)
sA.free_async(pool, a)
sB.malloc_async(pool, 256)            # cannot reach sA's free list — pool grows

# Re-run with explicit synchronize_stream:
pool2 = MemoryPool(); sC, sD = Stream(), Stream()
a2 = sC.malloc_async(pool2, 256)
sC.free_async(pool2, a2)
pool2.synchronize_stream(sC)          # promote sC's free list to cross-stream pool
sD.malloc_async(pool2, 256)           # reuses the promoted block; no growth
```

## 看模拟器

`free_async` pushes the block onto `free_blocks_by_stream[sid]` (per-stream
list). `malloc_async` checks the per-stream list first, then the cross-stream
pool keyed under `-1`.

`synchronize_stream(stream)` pops the entire per-stream list for `sid` and
extends the cross-stream pool with it. Blocks remain pool-owned; ownership of
the slab does not change.

The contract: cross-stream reuse is **only** possible after `synchronize_stream`.
Without it, the cross-stream pool is empty for that block, so the allocator
falls back to `_grow`. This keeps the simulator from silently allowing reuse
that real CUDA would forbid.

## 改一改

- Don't call `synchronize_stream` and observe `len(pool.slabs)` grow on the
  second stream's malloc.
- Synchronize before freeing — `synchronize_stream` is a no-op when nothing is
  on the per-stream list.
- Free on stream A, malloc the same size on stream A *first*, then on stream B
  — same-stream reuse takes priority and B falls through to grow.

## 真机对照

CUDA's stream-ordered allocator detects cross-stream reuse opportunities
through event sync — `cudaEventRecord(event, A)` followed by
`cudaStreamWaitEvent(B, event)` makes B's allocations after the wait eligible
to reuse blocks freed on A before the record. The `cudaMemPoolReuseAllowInternalDependencies`
attribute controls this. Phase 16 keeps the synchronization point explicit
(`synchronize_stream`) — Phase 17 may auto-promote on `record/wait` pairs.
```

- [ ] **Step 5: Create chapter 65**

Create `docs/tutorial/65-mempool-train-step.md`:
```markdown
# 65 · Memory Pool Train-Step — Reuse Convergence

A typical training loop allocates buffers (activations, gradients, temporaries)
each iteration and frees them at the end. Without a pool, every iteration would
hit the kernel allocator. With a pool, the first iteration warms up the working
set; every iteration after that reuses the same blocks. The high-water mark
plateaus, and the reuse rate climbs toward 1.

## What the example does

```python
pool = MemoryPool()
pool._recorder = rec
s = Stream()
for _ in range(10):
    act = s.malloc_async(pool, 8 * 1024, dtype=np.float32)
    grad = s.malloc_async(pool, 8 * 1024, dtype=np.float32)
    # ... real kernel would compute on act/grad here ...
    s.free_async(pool, act)
    s.free_async(pool, grad)
    pool.synchronize_stream(s)
# high_water_mark plateaus at 16 KB; reuse_rate = 18/20 = 0.9
```

## 看模拟器

Iteration 1: both `malloc_async` calls miss the free list and `_grow` adds a
slab each. `high_water_mark` rises from 0 → 8 KB → 16 KB.

Iteration 2-10: each `malloc_async` finds the freed block from the previous
iteration on the per-stream list (best-fit, oldest-free wins). Each malloc
emits a `PoolAllocate` event with `reused=True`.

After 10 iterations: 20 total `malloc_async` calls. 2 fresh
(`reused=False`), 18 reused → `pool_reuse_rate = 0.9`. `pool_high_water_mark`
stays at 16 KB. `len(pool.slabs) == 2`.

## 改一改

- Increase the iter count to 100 — `reuse_rate` approaches 1.0
  (`(100 * 2 - 2) / (100 * 2) = 0.99`).
- Add a third allocation per iteration with a varying size. The pool grows
  larger but still plateaus.
- Skip `synchronize_stream` and call free on a different stream from
  malloc — see the pool grow because cross-stream reuse is unavailable.

## 真机对照

PyTorch's caching allocator achieves the same effect transparently —
`tensor.zero_()` in a loop never round-trips to the CUDA allocator after the
first iter. JAX's XLA allocator does the same with even larger working sets.
The metric to watch is `torch.cuda.memory_stats()["allocated_bytes.peak"]`,
which is the equivalent of Phase 16's `pool_high_water_mark`.
```

- [ ] **Step 6: Commit**

```bash
git add docs/tutorial/62-mempool-basic.md docs/tutorial/63-mempool-fragmentation.md docs/tutorial/64-mempool-multi-stream.md docs/tutorial/65-mempool-train-step.md
git commit -m "docs(tutorial): chapters 62-65 — Phase 16 memory pools"
```

---

## Task 20: Microbench facts + runtime

**Files:**
- Create: `tests/microbench/test_phase16_facts.py`
- Create: `tests/microbench/test_phase16_runtime.py`

- [ ] **Step 1: Create facts microbench**

Create `tests/microbench/test_phase16_facts.py`:
```python
"""Phase 16 microbench — memory pool facts."""


def test_first_alloc_is_fresh_second_after_free_is_reused():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.trace.recorder import Recorder
    from gpusim.api import Stream
    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()
    a1 = pool.malloc_async(s, 64)
    pool.free_async(s, a1)
    pool.malloc_async(s, 64)
    assert rec.pool_allocate_events[0].reused is False
    assert rec.pool_allocate_events[1].reused is True


def test_cross_stream_alloc_without_sync_grows():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    sA = Stream(); sB = Stream()
    a = pool.malloc_async(sA, 64)
    pool.free_async(sA, a)
    pool.malloc_async(sB, 64)
    assert len(pool.slabs) == 2


def test_trim_respects_release_threshold():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a1 = pool.malloc_async(s, 1000)
    a2 = pool.malloc_async(s, 2000)
    pool.free_async(s, a1)
    pool.free_async(s, a2)
    # threshold 2500: neither slab can be released without dropping below.
    released = pool.trim_to(release_threshold_bytes=2500)
    assert released == 0


def test_high_water_mark_monotone_non_decreasing():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    history = []
    a1 = pool.malloc_async(s, 100); history.append(pool.high_water_mark)
    a2 = pool.malloc_async(s, 200); history.append(pool.high_water_mark)
    pool.free_async(s, a1);          history.append(pool.high_water_mark)
    pool.free_async(s, a2);          history.append(pool.high_water_mark)
    assert history == [100, 300, 300, 300]


def test_pool_id_increments():
    from gpusim.mempool.pool import MemoryPool
    p1 = MemoryPool()
    p2 = MemoryPool()
    p3 = MemoryPool()
    assert p2._pool_id == p1._pool_id + 1
    assert p3._pool_id == p2._pool_id + 1


def test_synchronize_stream_promotes_blocks():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 64)
    pool.free_async(s, a)
    assert len(pool.free_blocks_by_stream[s.stream_id]) == 1
    pool.synchronize_stream(s)
    assert len(pool.free_blocks_by_stream[s.stream_id]) == 0
    assert len(pool.free_blocks_by_stream[-1]) == 1
```

- [ ] **Step 2: Run facts**

Run: `.venv/bin/pytest tests/microbench/test_phase16_facts.py -v`
Expected: 6 passed.

- [ ] **Step 3: Create runtime microbench**

Create `tests/microbench/test_phase16_runtime.py`:
```python
import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_mempool_basic_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_basic"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_mempool_multi_stream_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_multi_stream"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_mempool_fragmentation_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_fragmentation"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_mempool_train_step_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_train_step"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30
```

- [ ] **Step 4: Run runtime (slow)**

Run: `.venv/bin/pytest tests/microbench/test_phase16_runtime.py -v -m slow`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/microbench/test_phase16_facts.py tests/microbench/test_phase16_runtime.py
git commit -m "test(microbench): Phase 16 facts (6) + runtime (4 slow)"
```

---

## Task 21: Regression rename phase1_14 → phase1_15 + add 4 Phase 15 examples

**Files:**
- Rename: `tests/parity/test_phase1_14_examples_unchanged.py` → `tests/parity/test_phase1_15_examples_unchanged.py`
- Modify the renamed file: docstring, list name `PHASE_1_14_EXAMPLES` → `PHASE_1_15_EXAMPLES`, function names `test_phase_1_14_*` → `test_phase_1_15_*`, append the 4 Phase 15 examples.

- [ ] **Step 1: Rename file**

```bash
git mv tests/parity/test_phase1_14_examples_unchanged.py tests/parity/test_phase1_15_examples_unchanged.py
```

- [ ] **Step 2: Update docstring + list name**

Edit `tests/parity/test_phase1_15_examples_unchanged.py`:

Line 1:
```python
"""Smoke-test: each Phase 1-15 example runs without crashing on Phase 15 Device path."""
```

Rename constant globally: `PHASE_1_14_EXAMPLES` → `PHASE_1_15_EXAMPLES` (use `replace_all`).

Rename functions: `test_phase_1_14_example_smoke` → `test_phase_1_15_example_smoke`, and `test_phase_1_14_example_smoke_slow` → `test_phase_1_15_example_smoke_slow`.

Update parametrize decorators to reference the renamed constant.

- [ ] **Step 3: Append 4 Phase 15 examples to the list**

Locate the `# Phase 14` block (4 entries: `persistent_kernel_server`, `dynamic_parallelism_recursive`, `persistent_work_queue`, `persistent_pipeline`). After this block, append:
```python
    # Phase 15
    "stream_capture_basic",
    "stream_capture_multi_stream",
    "graph_conditional_branch",
    "graph_while_loop",
```

- [ ] **Step 4: Run renamed regression**

Run: `.venv/bin/pytest tests/parity/test_phase1_15_examples_unchanged.py -v -m "not slow"`
Expected: ~58-59 passed (Phase 14 baseline 55 + 4 Phase 15 examples; minus any without `run.py`).

- [ ] **Step 5: Commit**

```bash
git add tests/parity/test_phase1_15_examples_unchanged.py
git commit -m "test(regression): rename phase1_14 → phase1_15 + add 4 Phase 15 examples"
```

---

## Task 22: README v16

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Locate the phase status table**

Run: `grep -n "Phase status" README.md` and read the surrounding lines to confirm column count.

- [ ] **Step 2: Add Phase 16 row**

Edit `README.md` — add a row to the phase status table after Phase 15:
```markdown
| 16 | Memory Pools + Stream-Ordered Async Allocator | ✅ |
```
(Match the column count and separator style used in the existing table.)

- [ ] **Step 3: Add Phase 16 detailed section**

After the existing Phase 15 detailed section (search for `### Phase 15`), append:
```markdown
### Phase 16 ✅ — Memory Pools + Stream-Ordered Async Allocator

`gpusim.mempool.MemoryPool` adds a stream-ordered allocator modeled after CUDA's
`cudaMallocAsync` family. **`pool.malloc_async(stream, n_bytes, dtype=...)`**
returns an `Allocation` whose `.buf` is a typed numpy view that any kernel
accepts as a buffer. **`pool.free_async(stream, allocation)`** returns the block
to the per-stream free list — the next `malloc_async` on the same stream
reuses it immediately (best-fit with oldest-free tiebreak). Cross-stream reuse
requires explicit **`pool.synchronize_stream(stream)`** to promote per-stream
free blocks into the cross-stream pool. **`pool.trim_to(release_threshold_bytes)`**
releases fully-free slabs above the threshold back to the OS. **Convenience
methods on Stream**: `s.malloc_async(pool, n)` and `s.free_async(pool, a)`.
**4 trace events**: `PoolAllocate`, `PoolFree`, `PoolGrow`, `PoolTrim`.
**4 metrics**: `pool_high_water_mark`, `pool_reuse_rate`, `pool_alloc_count`,
`pool_release_total_bytes`. **4 examples**: `mempool_basic`,
`mempool_multi_stream`, `mempool_fragmentation`, `mempool_train_step`.
**4 tutorial chapters** (62-65). **100% backward compatible** with Phase 1-15.
```

- [ ] **Step 4: Verify**

Run: `grep -A 2 "Phase 16 ✅" README.md`
Expected: prints the new section header and first lines.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs(readme): v16 — Phase 16 capabilities (memory pools + stream-ordered async allocator)"
```

---

## Task 23: Final acceptance + tag `phase16-complete`

- [ ] **Step 1: Run full non-slow suite**

Run: `.venv/bin/pytest -m "not slow" -q`
Expected: ~825-830 passed (Phase 15 baseline 770 + ~55-60 Phase 16 unit/parity/microbench/regression additions).

If count differs significantly, investigate before tagging.

- [ ] **Step 2: Run slow suite**

Run: `.venv/bin/pytest -m slow -q`
Expected: 4 Phase 16 runtime tests pass (plus pre-existing slow tests).

- [ ] **Step 3: Verify all 4 milestone tags**

Run: `git tag -l 'M*-phase16-complete' | sort`
Expected:
```
M1-phase16-complete
M2-phase16-complete
M3-phase16-complete
M4-phase16-complete
```

- [ ] **Step 4: Tag `phase16-complete`**

```bash
git tag phase16-complete
```

- [ ] **Step 5: Verify final tag list**

Run: `git tag -l 'M*-phase16-complete' 'phase16-complete' | sort -V`
Expected:
```
M1-phase16-complete
M2-phase16-complete
M3-phase16-complete
M4-phase16-complete
phase16-complete
```

---

## Acceptance criteria

Phase 16 ships when:

- [ ] All 5 milestone tags present (`M1-phase16-complete` ... `M4-phase16-complete`, `phase16-complete`)
- [ ] All 4 examples run cleanly via `python run.py`
- [ ] All 4 parity tests pass
- [ ] Microbench facts: reuse vs fresh-alloc flag, cross-stream grows without sync, trim respects release_threshold, high_water_mark monotone, pool ID increments, synchronize_stream promotes blocks
- [ ] Phase 1-15 regression test (renamed) passes with 4 Phase 15 examples added
- [ ] Test count: 770 → ~825-830 (+55-60)
- [ ] README v16 documents Phase 16 capabilities
