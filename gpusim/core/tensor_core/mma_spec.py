from __future__ import annotations
from dataclasses import dataclass
import re
from gpusim.frontend.ir import PtxType


@dataclass(frozen=True)
class MmaSpec:
    """Decoded mma/wgmma opcode."""
    is_async: bool
    m: int; n: int; k: int
    layout_a: str
    layout_b: str
    dtype_d: PtxType
    dtype_a: PtxType
    dtype_b: PtxType
    dtype_c: PtxType


_SHAPE_RE = re.compile(r"m(\d+)n(\d+)k(\d+)")


def _ptx_type(name: str) -> PtxType | None:
    try:
        return PtxType(name)
    except ValueError:
        return None


def parse_mma_op(op: str) -> MmaSpec | None:
    """Decode a mma/wgmma compute opcode string. Returns None for non-mma ops
    (including wgmma control ops: fence/commit_group/wait_group)."""
    parts = op.split(".")
    if not parts:
        return None
    is_sync = parts[0] == "mma" and parts[1:3] == ["sync", "aligned"]
    is_async_compute = (
        parts[0] == "wgmma"
        and len(parts) > 1
        and parts[1] == "mma_async"
    )
    if not (is_sync or is_async_compute):
        return None

    shape_idx = next((i for i, p in enumerate(parts) if _SHAPE_RE.fullmatch(p)), -1)
    if shape_idx < 0:
        return None
    m = _SHAPE_RE.fullmatch(parts[shape_idx])
    M, N, K = int(m.group(1)), int(m.group(2)), int(m.group(3))

    rest = parts[shape_idx + 1:]
    if is_sync:
        if len(rest) < 6:
            return None
        layout_a, layout_b = rest[0], rest[1]
        dtype_d = _ptx_type(rest[2])
        dtype_a = _ptx_type(rest[3])
        dtype_b = _ptx_type(rest[4])
        dtype_c = _ptx_type(rest[5])
        if None in (dtype_d, dtype_a, dtype_b, dtype_c):
            return None
        return MmaSpec(False, M, N, K, layout_a, layout_b,
                        dtype_d, dtype_a, dtype_b, dtype_c)

    # wgmma.mma_async.sync.aligned.m64n128k16.<dtype_d>.<dtype_a>.<dtype_b>
    # layout fixed (a row, b col) per Hopper convention; dtype_c defaults to dtype_d
    if len(rest) < 3:
        return None
    dtype_d = _ptx_type(rest[0])
    dtype_a = _ptx_type(rest[1])
    dtype_b = _ptx_type(rest[2])
    if None in (dtype_d, dtype_a, dtype_b):
        return None
    return MmaSpec(True, M, N, K, "row", "col",
                    dtype_d, dtype_a, dtype_b, dtype_d)
