from __future__ import annotations
from gpusim.config.schema import HBMConfig


def decompose_addr(addr: int, cfg: HBMConfig) -> tuple[int, int, int, int]:
    """Returns (channel, bank, col_in_row, row).
    Layout: [6:0]=offset [9:7]=ch [14:10]=col [18:15]=bank [30:19]=row
    """
    c   = (addr >> 7)  & 0x7
    col = (addr >> 10) & 0x1F
    b   = (addr >> 15) & 0xF
    row = (addr >> 19) & 0xFFF
    return (c, b, col, row)


class HBM:
    """Phase 2 HBM model: channel-level serialization + per-bank row buffer."""

    def __init__(self, cfg: HBMConfig, recorder=None):
        self.cfg = cfg
        self._channel_busy_until = [0] * cfg.channels
        self._bank_open_row: list[list[int | None]] = [
            [None] * cfg.banks_per_channel for _ in range(cfg.channels)
        ]
        self._recorder = recorder

    def request(self, line_addr: int, now: int) -> int:
        return self._service(line_addr, kind="READ", now=now)

    def write_request(self, line_addr: int, now: int) -> int:
        return self._service(line_addr, kind="WRITE_BACK", now=now)

    def _service(self, line_addr: int, kind: str, now: int) -> int:
        # Convert line_addr to byte address: cache passes line_addr = phys_addr >> 7
        byte_addr = line_addr * 128
        c, b, col, row = decompose_addr(byte_addr, self.cfg)

        start = max(now, self._channel_busy_until[c])
        if self._bank_open_row[c][b] == row:
            latency = self.cfg.row_hit_latency
            row_kind = "ROW_HIT"
        else:
            latency = self.cfg.row_miss_latency
            self._bank_open_row[c][b] = row
            row_kind = "ROW_MISS"

        end = start + latency
        self._channel_busy_until[c] = end

        if self._recorder is not None:
            self._recorder.hbm_access(
                cycle=now,
                served_at=end,
                addr=line_addr,
                channel=c,
                bank=b,
                row=row,
                kind=kind,
                row_kind=row_kind,
                queue_wait=start - now,
            )
        return end
