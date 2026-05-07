from __future__ import annotations
from collections import defaultdict


def bank_conflict_degree(addresses: list[int], active_mask: int = (1 << 32) - 1,
                         banks: int = 32, word_bytes: int = 4) -> int:
    """Per-bank max count of distinct addresses; 1 = no conflict."""
    by_bank: dict[int, set[int]] = defaultdict(set)
    for lane, addr in enumerate(addresses):
        if not (active_mask >> lane) & 1:
            continue
        bank = (addr // word_bytes) % banks
        by_bank[bank].add(addr)
    if not by_bank:
        return 1
    return max(len(addrs) for addrs in by_bank.values())
