def test_l2_mshr_alloc_new_entry():
    from gpusim.core.cache.l2_mshr import L2Mshr
    mshr = L2Mshr(n_slots=4)
    allocated, entry = mshr.lookup_or_alloc(line_addr=42, sm_id=0, now=10)
    assert allocated is True
    assert entry.line_addr == 42
    assert entry.completion_at == -1


def test_l2_mshr_merge_same_line_from_different_sm():
    from gpusim.core.cache.l2_mshr import L2Mshr
    mshr = L2Mshr(n_slots=4)
    a, e1 = mshr.lookup_or_alloc(line_addr=42, sm_id=0, now=10)
    b, e2 = mshr.lookup_or_alloc(line_addr=42, sm_id=3, now=12)
    assert a is True and b is False
    assert e1 is e2


def test_l2_mshr_full_returns_none():
    from gpusim.core.cache.l2_mshr import L2Mshr
    mshr = L2Mshr(n_slots=2)
    mshr.lookup_or_alloc(line_addr=0, sm_id=0, now=0)
    mshr.lookup_or_alloc(line_addr=1, sm_id=0, now=0)
    a, e = mshr.lookup_or_alloc(line_addr=2, sm_id=0, now=0)
    assert a is False and e is None


def test_l2_mshr_release_frees_slot():
    from gpusim.core.cache.l2_mshr import L2Mshr
    mshr = L2Mshr(n_slots=1)
    mshr.lookup_or_alloc(line_addr=0, sm_id=0, now=0)
    a, e = mshr.lookup_or_alloc(line_addr=1, sm_id=0, now=0)
    assert e is None
    mshr.release(0)
    a, e = mshr.lookup_or_alloc(line_addr=1, sm_id=0, now=0)
    assert a is True


def test_l2_mshr_active_count():
    from gpusim.core.cache.l2_mshr import L2Mshr
    mshr = L2Mshr(n_slots=4)
    assert mshr.active_count() == 0
    mshr.lookup_or_alloc(line_addr=0, sm_id=0, now=0)
    mshr.lookup_or_alloc(line_addr=1, sm_id=0, now=0)
    assert mshr.active_count() == 2


def test_l2_fetch_with_mshr_full_returns_negative_one():
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.config.schema import CacheConfig
    class _NoOpHbm:
        def request(self, line_addr, now): return now + 100
        def write_request(self, line_addr, now): return now + 100
    cfg = CacheConfig(l2_mshr_slots=2)
    l2 = L2Cache(cfg, _NoOpHbm())
    r1 = l2.fetch(line_addr=0x1000, sm_id=0, now=0)
    r2 = l2.fetch(line_addr=0x2000, sm_id=1, now=0)
    assert r1 > 0 and r2 > 0
    r3 = l2.fetch(line_addr=0x3000, sm_id=2, now=0)
    assert r3 == -1


def test_l2_fetch_records_origin_sm_on_install():
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.config.schema import CacheConfig
    class _NoOpHbm:
        def request(self, line_addr, now): return now + 100
        def write_request(self, line_addr, now): return now + 100
    cfg = CacheConfig(l2_mshr_slots=4)
    l2 = L2Cache(cfg, _NoOpHbm())
    l2.fetch(line_addr=0x1000, sm_id=3, now=0)
    l2.tick(now=10000)
    set_idx = 0x1000 & l2._set_mask
    tag = 0x1000 >> l2._set_bits
    line = l2._sets[set_idx].find(tag)
    assert line is not None
    assert line.origin_sm == 3


def test_l2_cross_sm_hit_records_metadata_in_recorder():
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.config.schema import CacheConfig
    class _NoOpHbm:
        def request(self, line_addr, now): return now + 100
        def write_request(self, line_addr, now): return now + 100
    class _Rec:
        def __init__(self): self.events = []
        def l2_access(self, **kw): self.events.append(kw)
        def l2_mshr(self, **kw): pass
    cfg = CacheConfig(l2_mshr_slots=4)
    rec = _Rec()
    l2 = L2Cache(cfg, _NoOpHbm(), recorder=rec)
    l2.fetch(line_addr=0x1000, sm_id=0, now=0)
    l2.tick(now=10000)
    rec.events.clear()
    l2.fetch(line_addr=0x1000, sm_id=5, now=20000)
    hits = [e for e in rec.events if e.get("kind") == "HIT"]
    assert hits and hits[0].get("origin_sm") == 0
    assert hits[0].get("hit_sm") == 5


def test_l1_propagates_l2_mshr_full_as_reject():
    from gpusim.core.cache.l1 import L1Cache, Reject
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.config.schema import CacheConfig
    class _NoOpHbm:
        def request(self, line_addr, now): return now + 100
        def write_request(self, line_addr, now): return now + 100
    cfg = CacheConfig(l2_mshr_slots=1)
    l2 = L2Cache(cfg, _NoOpHbm())
    l1 = L1Cache(cfg, l2)
    r1 = l1.access(line_addr=0x1000, warp_id=0, dst_regs=(),
                    mode="load", now=0)
    r2 = l1.access(line_addr=0x2000, warp_id=0, dst_regs=(),
                    mode="load", now=0)
    assert isinstance(r2, Reject)
    assert getattr(r2, "reason", "MSHR_FULL") in ("MSHR_FULL", "L2_MSHR_FULL")
