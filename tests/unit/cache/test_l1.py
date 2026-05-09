from gpusim.core.cache.l1 import L1Cache, AccessResult, Hit, MissNewMSHR, MissMergeMSHR, Reject
from gpusim.config.schema import CacheConfig


class MockL2:
    """Mock L2 returning fixed completion cycle for any request."""
    def __init__(self, latency: int = 200):
        self.latency = latency
        self.requests: list[tuple[int, int]] = []  # (line_addr, request_at)
        self.write_throughs: list[tuple[int, int]] = []  # (line_addr, now)

    def fetch(self, line_addr: int, now: int, sm_id: int = -1,
              requesting_stream_id: int = -1) -> int:
        self.requests.append((line_addr, now))
        return now + self.latency

    def write_through(self, line_addr: int, now: int,
                      requesting_stream_id: int = -1) -> None:
        self.write_throughs.append((line_addr, now))


def make_l1(cfg=None) -> tuple[L1Cache, MockL2]:
    cfg = cfg or CacheConfig()
    l2 = MockL2()
    l1 = L1Cache(cfg=cfg, l2=l2)
    return l1, l2


def test_first_load_misses_and_allocates_mshr():
    l1, l2 = make_l1()
    res = l1.access(line_addr=0x100, warp_id=0, dst_regs=("r1",), mode="load", now=0)
    assert isinstance(res, MissNewMSHR)
    assert res.ready_at > 0
    assert len(l2.requests) == 1


def test_repeated_load_to_same_line_merges_mshr():
    l1, l2 = make_l1()
    r1 = l1.access(line_addr=0x100, warp_id=0, dst_regs=("r1",), mode="load", now=0)
    assert isinstance(r1, MissNewMSHR)
    r2 = l1.access(line_addr=0x100, warp_id=1, dst_regs=("r2",), mode="load", now=5)
    assert isinstance(r2, MissMergeMSHR)
    assert r2.ready_at == r1.ready_at        # same expected completion
    assert len(l2.requests) == 1              # only one downstream fetch


def test_load_after_install_hits():
    l1, l2 = make_l1()
    res = l1.access(line_addr=0x100, warp_id=0, dst_regs=("r1",), mode="load", now=0)
    expected = res.ready_at
    l1.install_completed_lines(now=expected)
    res2 = l1.access(line_addr=0x100, warp_id=0, dst_regs=("r2",), mode="load",
                     now=expected + 10)
    assert isinstance(res2, Hit)
    cfg = CacheConfig()
    assert res2.ready_at == expected + 10 + cfg.l1_hit_latency


def test_mshr_full_returns_reject():
    cfg = CacheConfig(mshr_slots=2)
    l2 = MockL2()
    l1 = L1Cache(cfg=cfg, l2=l2)
    l1.access(line_addr=0x100, warp_id=0, dst_regs=("r1",), mode="load", now=0)
    l1.access(line_addr=0x200, warp_id=0, dst_regs=("r1",), mode="load", now=0)
    res = l1.access(line_addr=0x300, warp_id=0, dst_regs=("r1",), mode="load", now=0)
    assert isinstance(res, Reject)


def test_store_miss_bypasses_l1_no_mshr():
    """Phase 2 spec §3.4: store-miss bypasses L1 (no-write-allocate)."""
    l1, l2 = make_l1()
    res = l1.access(line_addr=0x100, warp_id=0, dst_regs=(), mode="store", now=0)
    assert isinstance(res, Hit)
    # store didn't allocate MSHR or trigger L2 fetch
    assert len(l2.requests) == 0
    # still no L1 line for this address
    res2 = l1.access(line_addr=0x100, warp_id=0, dst_regs=("r1",), mode="load", now=10)
    assert isinstance(res2, MissNewMSHR)


def test_eviction_silent_no_writeback():
    """L1 is write-through → no dirty bit → eviction is silent."""
    l1, l2 = make_l1()
    # fill one set with 4 ways then evict
    line0 = 0x000  # set_idx = 0x000 & 0xFF = 0
    line1 = 0x100  # set_idx = 0x100 & 0xFF = 0
    line2 = 0x200  # set_idx = 0
    line3 = 0x300  # set_idx = 0
    line4 = 0x400  # set_idx = 0 — evicts line0
    for la in (line0, line1, line2, line3):
        r = l1.access(line_addr=la, warp_id=0, dst_regs=("r1",), mode="load", now=0)
        l1.install_completed_lines(now=r.ready_at)
    r = l1.access(line_addr=line4, warp_id=0, dst_regs=("r1",), mode="load", now=1000)
    l1.install_completed_lines(now=r.ready_at)
    # access line0 should miss again (was evicted)
    r2 = l1.access(line_addr=line0, warp_id=0, dst_regs=("r1",), mode="load", now=2000)
    assert isinstance(r2, MissNewMSHR)


def test_store_miss_propagates_write_through_to_l2():
    """Per spec §3.4: store-miss bypasses L1 (no-write-allocate) but
    still flows write-through to L2."""
    from unittest.mock import MagicMock, call
    cfg = CacheConfig()
    l2 = MagicMock()
    l2.fetch = MagicMock(return_value=200)
    l2.write_through = MagicMock()
    l1 = L1Cache(cfg, l2)
    res = l1.access(line_addr=0x100, warp_id=0, dst_regs=(),
                    mode="store", now=10)
    assert isinstance(res, Hit)
    l2.write_through.assert_called_once()
    call_kwargs = l2.write_through.call_args[1]
    assert call_kwargs.get("line_addr") == 0x100
    assert call_kwargs.get("now") == 10


def test_store_hit_propagates_write_through_to_l2():
    """Store hit on L1 line: mark touch, but still write-through to L2."""
    from unittest.mock import MagicMock
    cfg = CacheConfig()
    l2 = MagicMock()
    l2.fetch = MagicMock(return_value=200)
    l2.write_through = MagicMock()
    l1 = L1Cache(cfg, l2)
    # bring line in
    res = l1.access(line_addr=0x100, warp_id=0, dst_regs=("r1",),
                    mode="load", now=0)
    l1.install_completed_lines(now=res.ready_at)
    # store hit
    l2.write_through.reset_mock()
    res2 = l1.access(line_addr=0x100, warp_id=0, dst_regs=(),
                     mode="store", now=res.ready_at + 10)
    assert isinstance(res2, Hit)
    l2.write_through.assert_called_once()
    call_kwargs = l2.write_through.call_args[1]
    assert call_kwargs.get("line_addr") == 0x100
    assert call_kwargs.get("now") == res.ready_at + 10
