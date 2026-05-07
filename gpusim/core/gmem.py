from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CoalesceInfo:
    n_transactions: int
    n_active: int
    efficiency: float


def coalescing_info(addresses: list[int], active_mask: int = (1 << 32) - 1,
                    sector_bytes: int = 128) -> CoalesceInfo:
    sectors: set[int] = set()
    n_active = 0
    for lane, addr in enumerate(addresses):
        if not (active_mask >> lane) & 1:
            continue
        n_active += 1
        sectors.add(addr // sector_bytes)
    n_tx = max(1, len(sectors)) if n_active > 0 else 0
    eff = (n_active / (n_tx * 32)) if n_tx > 0 else 0.0
    return CoalesceInfo(n_transactions=n_tx, n_active=n_active, efficiency=eff)
