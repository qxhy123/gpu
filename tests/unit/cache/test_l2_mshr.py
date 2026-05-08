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
