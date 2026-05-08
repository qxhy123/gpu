from __future__ import annotations
from gpusim.config.schema import HBMConfig


class HBM:
    """Mock HBM for M1/M2: returns fixed latency for all requests.
    M3 replaces this with a channel + bank + row buffer implementation."""

    def __init__(self, cfg: HBMConfig):
        self.cfg = cfg

    def request(self, line_addr: int, now: int) -> int:
        return now + self.cfg.row_miss_latency * 4   # rough placeholder

    def write_request(self, line_addr: int, now: int) -> int:
        return now + self.cfg.row_miss_latency * 4
