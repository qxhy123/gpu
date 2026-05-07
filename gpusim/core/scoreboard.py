from __future__ import annotations


class Scoreboard:
    """Tracks the cycle at which each in-flight write becomes visible."""

    def __init__(self):
        self._pending: dict[str, int] = {}

    def mark_write(self, reg: str, available_at_cycle: int) -> None:
        cur = self._pending.get(reg, -1)
        if available_at_cycle > cur:
            self._pending[reg] = available_at_cycle

    def has_pending(self, reg: str, now: int) -> bool:
        c = self._pending.get(reg, -1)
        return c > now

    def ready_at(self, reg: str, now: int) -> int:
        c = self._pending.get(reg, -1)
        return max(now, c) if c > now else now

    def advance(self, now: int) -> None:
        self._pending = {r: c for r, c in self._pending.items() if c > now}
