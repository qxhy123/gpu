"""Phase 14: WorkQueue for persistent kernels."""
from collections import deque
from dataclasses import dataclass, field


@dataclass
class WorkQueue:
    items: deque = field(default_factory=deque)
    stopped: bool = False

    def push(self, item) -> None:
        if self.stopped:
            raise RuntimeError("queue stopped; cannot push")
        self.items.append(item)

    def pop(self):
        if not self.items:
            return None
        return self.items.popleft()

    def stop(self) -> None:
        self.stopped = True

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def is_stopped(self) -> bool:
        return self.stopped
