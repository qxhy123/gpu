from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class InflightBulkStore:
    issued_at: int
    completion_at: int
    bytes_total: int
    commit_group_id: int = -1


@dataclass
class BulkStoreQueue:
    capacity: int = 16
    in_flight: list[InflightBulkStore] = field(default_factory=list)
    committed_groups: list[int] = field(default_factory=list)
    next_group_id: int = 0

    def try_push(self, f: InflightBulkStore) -> bool:
        if len(self.in_flight) >= self.capacity:
            return False
        self.in_flight.append(f)
        return True

    def commit_group(self) -> int:
        gid = self.next_group_id
        self.next_group_id += 1
        for f in self.in_flight:
            if f.commit_group_id < 0:
                f.commit_group_id = gid
        self.committed_groups.append(gid)
        return gid

    def must_wait(self, target_n: int) -> bool:
        return len(self.committed_groups) > target_n

    def drain_completed_groups(self, now: int) -> list[int]:
        drained: list[int] = []
        while self.committed_groups:
            gid = self.committed_groups[0]
            in_group = [f for f in self.in_flight if f.commit_group_id == gid]
            if not all(f.completion_at <= now for f in in_group):
                break
            drained.append(gid)
            self.in_flight = [f for f in self.in_flight if f.commit_group_id != gid]
            self.committed_groups.pop(0)
        return drained


def do_bulk_store_2d(*, gmem, smem, cta_id: int, smem_src: int,
                       desc) -> int:
    """Copy a dim_y × dim_x tile from smem[smem_src:] to gmem (row-major)
    using desc.stride_y rows. Returns total bytes stored."""
    bytes_per_row = desc.dim_x * desc.elem_bytes
    dst_stride_bytes = desc.stride_y * desc.elem_bytes
    smem_buf = smem._cta[cta_id]
    for row in range(desc.dim_y):
        gmem_addr = desc.gmem_base + row * dst_stride_bytes
        src_off = smem_src + row * bytes_per_row
        chunk = bytes(smem_buf[src_off:src_off + bytes_per_row])
        gmem.store_bytes(gmem_addr, chunk)
    return desc.dim_y * bytes_per_row
