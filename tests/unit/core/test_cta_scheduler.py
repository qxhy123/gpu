class FakeSM:
    def __init__(self, sm_id: int, capacity: int = 32, n_warps: int = 0):
        self.sm_id = sm_id
        self._capacity = capacity
        self._n_warps = n_warps
    def can_admit_cta(self, occ) -> bool:
        return self._capacity > 0
    def active_warp_count(self) -> int:
        return self._n_warps


class FakeOcc:
    active_ctas = 32


def test_rr_cycles_through_sms():
    from gpusim.core.scheduler import RRCtaScheduler
    sched = RRCtaScheduler()
    sms = [FakeSM(i) for i in range(4)]
    picks = []
    for _ in range(8):
        sm = sched.pick(sms, FakeOcc())
        picks.append(sm.sm_id)
    assert picks == [0, 1, 2, 3, 0, 1, 2, 3]


def test_rr_skips_full_sms():
    from gpusim.core.scheduler import RRCtaScheduler
    sched = RRCtaScheduler()
    sms = [FakeSM(0), FakeSM(1, capacity=0),
           FakeSM(2), FakeSM(3, capacity=0)]
    picks = [sched.pick(sms, FakeOcc()).sm_id for _ in range(4)]
    assert picks == [0, 2, 0, 2]


def test_rr_returns_none_when_all_full():
    from gpusim.core.scheduler import RRCtaScheduler
    sched = RRCtaScheduler()
    sms = [FakeSM(i, capacity=0) for i in range(4)]
    assert sched.pick(sms, FakeOcc()) is None


def test_greedy_picks_least_loaded():
    from gpusim.core.scheduler import GreedyCtaScheduler
    sched = GreedyCtaScheduler()
    sms = [FakeSM(0, n_warps=8), FakeSM(1, n_warps=2),
           FakeSM(2, n_warps=16), FakeSM(3, n_warps=4)]
    assert sched.pick(sms, FakeOcc()).sm_id == 1


def test_greedy_returns_none_when_all_full():
    from gpusim.core.scheduler import GreedyCtaScheduler
    sched = GreedyCtaScheduler()
    sms = [FakeSM(i, capacity=0) for i in range(4)]
    assert sched.pick(sms, FakeOcc()) is None


def test_factory_dispatches_by_string():
    from gpusim.core.scheduler import make_cta_scheduler, RRCtaScheduler, GreedyCtaScheduler
    assert isinstance(make_cta_scheduler("rr"), RRCtaScheduler)
    assert isinstance(make_cta_scheduler("greedy"), GreedyCtaScheduler)
    import pytest
    with pytest.raises(ValueError):
        make_cta_scheduler("bogus")


def test_rr_peek_returns_k_sms():
    from gpusim.core.scheduler import RRCtaScheduler
    sched = RRCtaScheduler()
    sms = [FakeSM(i) for i in range(4)]
    result = sched.peek(sms, FakeOcc(), k=2)
    assert result is not None
    assert [sm.sm_id for sm in result] == [0, 1]


def test_rr_peek_doesnt_advance_until_commit():
    from gpusim.core.scheduler import RRCtaScheduler
    sched = RRCtaScheduler()
    sms = [FakeSM(i) for i in range(4)]
    sched.peek(sms, FakeOcc(), k=2)
    sched.peek(sms, FakeOcc(), k=2)
    result = sched.peek(sms, FakeOcc(), k=2)
    assert [sm.sm_id for sm in result] == [0, 1]


def test_rr_peek_commit_advances():
    from gpusim.core.scheduler import RRCtaScheduler
    sched = RRCtaScheduler()
    sms = [FakeSM(i) for i in range(4)]
    sched.peek(sms, FakeOcc(), k=2); sched.commit(k=2)
    result = sched.peek(sms, FakeOcc(), k=2)
    assert [sm.sm_id for sm in result] == [2, 3]


def test_rr_peek_returns_none_when_insufficient():
    from gpusim.core.scheduler import RRCtaScheduler
    sched = RRCtaScheduler()
    sms = [FakeSM(0), FakeSM(1, capacity=0), FakeSM(2, capacity=0), FakeSM(3, capacity=0)]
    result = sched.peek(sms, FakeOcc(), k=2)
    assert result is None


def test_greedy_peek_returns_k_least_loaded():
    from gpusim.core.scheduler import GreedyCtaScheduler
    sched = GreedyCtaScheduler()
    sms = [FakeSM(0, n_warps=8), FakeSM(1, n_warps=2),
           FakeSM(2, n_warps=16), FakeSM(3, n_warps=4)]
    result = sched.peek(sms, FakeOcc(), k=2)
    ids = sorted(sm.sm_id for sm in result)
    assert ids == [1, 3]


def test_greedy_commit_is_noop():
    from gpusim.core.scheduler import GreedyCtaScheduler
    sched = GreedyCtaScheduler()
    sms = [FakeSM(i) for i in range(4)]
    sched.peek(sms, FakeOcc(), k=2)
    sched.commit(k=2)
    result = sched.peek(sms, FakeOcc(), k=2)
    assert len(result) == 2


def test_pick_k1_equivalent_to_old_pick():
    """For k=1 (Phase 4 default), peek+commit should behave like pick."""
    from gpusim.core.scheduler import RRCtaScheduler
    sched = RRCtaScheduler()
    sms = [FakeSM(i) for i in range(4)]
    picks = []
    for _ in range(8):
        result = sched.peek(sms, FakeOcc(), k=1)
        sched.commit(k=1)
        picks.append(result[0].sm_id)
    assert picks == [0, 1, 2, 3, 0, 1, 2, 3]
