# tests/unit/core/test_warp_scheduler.py
from gpusim.core.warp import Warp, StallReason
from gpusim.core.scheduler import LRRScheduler, GTOScheduler

class _FakeKernel:
    def __init__(self, n): self.instrs = [None] * n; self.labels = {}; self.ipdom = {}

def test_lrr_round_robins_among_ready_warps():
    warps = [Warp(warp_id=i, kernel=_FakeKernel(10)) for i in range(4)]
    sched = LRRScheduler(warp_count=4)
    picks = []
    for _ in range(8):
        chosen = sched.pick(now=0, candidates=lambda i: True)
        picks.append(chosen)
    assert picks == [0, 1, 2, 3, 0, 1, 2, 3]

def test_lrr_skips_not_ready():
    sched = LRRScheduler(warp_count=4)
    ready = {0: True, 1: False, 2: True, 3: False}
    picks = [sched.pick(now=0, candidates=lambda i: ready[i]) for _ in range(4)]
    assert picks == [0, 2, 0, 2]

def test_gto_sticks_to_one_warp_until_stall():
    sched = GTOScheduler(warp_count=4)
    ready = {0: True, 1: True, 2: True, 3: True}
    a = sched.pick(now=0, candidates=lambda i: ready[i])
    b = sched.pick(now=1, candidates=lambda i: ready[i])
    assert a == b

def test_gto_switches_to_oldest_ready_on_stall():
    sched = GTOScheduler(warp_count=4)
    sched.pick(now=0, candidates=lambda i: True)
    nxt = sched.pick(now=1, candidates=lambda i: i != 0)
    assert nxt == 1
