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

def test_mshr_full_is_a_stall_reason():
    from gpusim.core.warp import StallReason
    assert StallReason.MSHR_FULL.value == "MSHR_FULL"


def test_warp_has_warp_group_id_field():
    from gpusim.core.warp import Warp
    w = Warp(warp_id=5, kernel=None)
    # default warp_group_id = warp_id // 4
    assert w.warp_group_id == 1
    assert w.wgmma_pending_pc == -1


def test_stall_reason_has_wgmma_tokens():
    from gpusim.core.warp import StallReason
    assert StallReason.WGMMA_QUEUE_FULL.value == "WGMMA_QUEUE_FULL"
    assert StallReason.WGMMA_WAIT.value == "WGMMA_WAIT"


def test_phase4_stall_tokens_and_pending_pc():
    from gpusim.core.warp import StallReason, Warp
    assert StallReason.L2_MSHR_FULL.value == "L2_MSHR_FULL"
    assert StallReason.BULK_STORE_QUEUE_FULL.value == "BULK_STORE_QUEUE_FULL"
    assert StallReason.BULK_STORE_WAIT.value == "BULK_STORE_WAIT"
    w = Warp(warp_id=0, kernel=None)
    assert w.bulk_store_pending_pc == -1


def test_phase5_warp_cluster_fields_default():
    from gpusim.core.warp import Warp
    w = Warp(warp_id=0, kernel=None)
    assert w.cluster_id == -1
    assert w.cluster_rank == -1
    assert w.cluster_barrier_arrived is False
    assert w.cluster_barrier_wait_pc == -1
    assert w.cluster_barrier_phase_at_wait == -1


def test_phase5_cluster_barrier_wait_stall_token():
    from gpusim.core.warp import StallReason
    assert StallReason.CLUSTER_BARRIER_WAIT.value == "CLUSTER_BARRIER_WAIT"
