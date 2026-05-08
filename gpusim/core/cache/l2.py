from __future__ import annotations
from typing import Protocol
from gpusim.config.schema import CacheConfig


class HBMProtocol(Protocol):
    def request(self, line_addr: int, now: int) -> int: ...
    def write_request(self, line_addr: int, now: int) -> int: ...


class L2Cache:
    """Mock L2 for M1: returns fixed latency for all requests.
    M2 replaces this with a tag-precise + write-back implementation."""

    def __init__(self, cfg: CacheConfig, hbm: HBMProtocol):
        self.cfg = cfg
        self.hbm = hbm

    def fetch(self, *, line_addr: int, now: int) -> int:
        """L1 calls this on miss. Mock: return now + l2_hit_latency."""
        return now + self.cfg.l2_hit_latency

    def write_through(self, line_addr: int, now: int) -> None:
        """Receive a write-through from L1. Mock: ignore."""
        pass
