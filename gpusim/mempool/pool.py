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

    def synchronize_stream(self, stream) -> None:
        """Promote all per-stream free blocks for this stream into the cross-stream pool.
        Phase 16: explicit promotion lets later mallocs on other streams reuse these blocks."""
        sid = stream.stream_id
        promoted = self.free_blocks_by_stream.pop(sid, [])
        if promoted:
            self.free_blocks_by_stream[-1].extend(promoted)

    def trim_to(self, release_threshold_bytes: int) -> int:
        """Release fully-free slabs whose total free bytes exceed the threshold.
        Returns total bytes released."""
        per_slab_free = defaultdict(int)
        for stream_key, blocks in self.free_blocks_by_stream.items():
            for b in blocks:
                per_slab_free[b.slab_index] += b.n_bytes

        released = 0
        slabs_to_release = []
        for slab_idx, free_bytes in per_slab_free.items():
            if free_bytes != self.slab_n_bytes[slab_idx]:
                continue
            current_total = sum(self.slab_n_bytes)
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
