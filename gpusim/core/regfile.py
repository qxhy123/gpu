from __future__ import annotations
import re

_DIGITS = re.compile(r"(\d+)$")


def bank_of(reg: str, banks: int = 4) -> int:
    m = _DIGITS.search(reg)
    return int(m.group(1)) % banks if m else 0


def operand_extra_cycles(src_regs: list[str], banks: int = 4) -> int:
    """Count duplicate-bank reads beyond the first; each costs +1 cycle."""
    seen: dict[int, int] = {}
    for r in src_regs:
        b = bank_of(r, banks)
        seen[b] = seen.get(b, 0) + 1
    extra = sum(c - 1 for c in seen.values() if c > 1)
    return extra
