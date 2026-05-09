from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ClusterBarrierPool:
    """Per-cluster barrier state (Device-owned).

    Tracks which cluster ranks have arrived. When all expected ranks have
    arrived, the barrier flips phase (0 ↔ 1) and clears arrived_mask, allowing
    a new round.
    """
    expected: int
    arrived_mask: int = 0
    phase: int = 0

    def arrive(self, cluster_rank: int) -> bool:
        """Mark a rank as arrived. Returns True if this completes the barrier."""
        self.arrived_mask |= (1 << cluster_rank)
        if bin(self.arrived_mask).count("1") >= self.expected:
            self.arrived_mask = 0
            self.phase ^= 1
            return True
        return False

    def is_released(self, captured_phase: int) -> bool:
        """Return True if barrier has flipped past captured_phase."""
        return self.phase != captured_phase
