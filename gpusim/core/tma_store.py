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
