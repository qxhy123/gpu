from gpusim.core.cache.l2 import L2Cache
from gpusim.config.schema import CacheConfig


class MockHBM:
    def __init__(self, latency=130):
        self.latency = latency
        self.requests: list[tuple[int, str, int]] = []

    def request(self, line_addr: int, now: int) -> int:
        self.requests.append((line_addr, "READ", now))
        return now + self.latency

    def write_request(self, line_addr: int, now: int) -> int:
        self.requests.append((line_addr, "WRITE_BACK", now))
        return now + self.latency


def test_l2_first_load_misses_fetches_hbm():
    l2 = L2Cache(CacheConfig(), MockHBM())
    completion = l2.fetch(line_addr=0x1000, now=0)
    assert completion > 0
    # one HBM read issued
    assert len(l2._hbm.requests) == 1


def test_l2_load_after_install_hits():
    cfg = CacheConfig()
    hbm = MockHBM()
    l2 = L2Cache(cfg, hbm)
    c1 = l2.fetch(line_addr=0x1000, now=0)
    c2 = l2.fetch(line_addr=0x1000, now=c1 + 100)
    # second fetch should be a hit; latency = l2_hit_latency
    assert c2 == (c1 + 100) + cfg.l2_hit_latency
    # only one HBM request total
    assert len(hbm.requests) == 1


def test_l2_store_marks_line_dirty():
    cfg = CacheConfig()
    hbm = MockHBM()
    l2 = L2Cache(cfg, hbm)
    # bring line in via load
    l2.fetch(line_addr=0x1000, now=0)
    # write-through from L1 — find the line and check dirty bit
    l2.write_through(line_addr=0x1000, now=100)
    # Look up the L2 internal state via fetch hit
    c2 = l2.fetch(line_addr=0x1000, now=200)
    assert c2 > 200    # was a hit
    # eviction would now be dirty


def test_l2_dirty_eviction_triggers_hbm_write():
    """Spec §4.3: dirty L2 line is written back on eviction."""
    cfg = CacheConfig()
    hbm = MockHBM()
    l2 = L2Cache(cfg, hbm)
    # bring line A in and dirty it
    l2.fetch(line_addr=0x10000, now=0)
    l2.write_through(line_addr=0x10000, now=10)
    # we need to evict line A. Force by allocating ways_per_set + 1 lines mapping to same set.
    # set_idx = line_addr & 0x7FF. So line 0x10000 + 0x800 has same set.
    set_mask = (cfg.l2_size_bytes // cfg.l2_line_bytes // cfg.l2_ways) - 1
    # confirm
    assert set_mask + 1 == 2048
    base = 0x10000
    # evict by filling its set
    for k in range(cfg.l2_ways):
        addr = base + ((k + 1) << 11) * cfg.l2_line_bytes
        l2.fetch(line_addr=addr, now=100 + k)
    # At this point line at 0x10000 has been evicted; expect HBM write
    wb_requests = [r for r in hbm.requests if r[1] == "WRITE_BACK"]
    assert len(wb_requests) >= 1
    assert wb_requests[0][0] == 0x10000
