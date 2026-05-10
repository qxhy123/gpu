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
