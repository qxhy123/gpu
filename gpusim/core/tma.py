"""TMA-lite: TensorDescriptor pool + cp.async.bulk.tensor.2d functional copy.
Simplified Hopper TMA — no swizzle, no multicast, no async pipelining at this layer
(timing handled separately by SM main loop)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TmaDescriptor:
    """Resolved runtime descriptor (different from frontend.ir.TensorDescriptor —
    that one carries register names; this one has resolved values)."""
    gmem_base: int
    dim_x: int       # number of columns (innermost dim)
    dim_y: int       # number of rows
    stride_y: int    # row stride in elements (NOT bytes)
    elem_bytes: int


class TensorDescriptorPool:
    """Per-SM pool of TMA descriptors. `gpusim.tma_desc` allocates entries here."""

    def __init__(self) -> None:
        self._entries: list[TmaDescriptor] = []

    def allocate(self, *, gmem_base: int, dim_x: int, dim_y: int,
                  stride_y: int, elem_bytes: int) -> int:
        """Allocate a new entry; return handle (index)."""
        self._entries.append(TmaDescriptor(
            gmem_base=gmem_base, dim_x=dim_x, dim_y=dim_y,
            stride_y=stride_y, elem_bytes=elem_bytes,
        ))
        return len(self._entries) - 1

    def lookup(self, handle: int) -> TmaDescriptor:
        return self._entries[handle]


def do_bulk_copy_2d(*, gmem, smem, cta_id: int, smem_dst: int,
                     desc: TmaDescriptor) -> int:
    """Copy a dim_y × dim_x tile (row-major in gmem) into smem starting at smem_dst.
    Returns total bytes copied."""
    bytes_per_row = desc.dim_x * desc.elem_bytes
    src_stride_bytes = desc.stride_y * desc.elem_bytes
    smem_buf = smem._cta[cta_id]
    for row in range(desc.dim_y):
        gmem_addr = desc.gmem_base + row * src_stride_bytes
        chunk = gmem.load_bytes(gmem_addr, bytes_per_row)
        dst_off = smem_dst + row * bytes_per_row
        smem_buf[dst_off:dst_off + bytes_per_row] = np.frombuffer(chunk, dtype=np.uint8)
    return desc.dim_y * bytes_per_row
