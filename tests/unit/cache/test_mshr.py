from gpusim.core.cache.mshr import MSHRPool, MSHREntry, Waiter


def test_pool_starts_empty():
    p = MSHRPool(slots=4)
    assert p.is_full() is False
    assert p.find_for_line(0x100) is None

def test_allocate_returns_entry():
    p = MSHRPool(slots=4)
    e = p.allocate(line_addr=0x100, issued_at=10, expected=410,
                   warp_id=0, dst_regs=("r1",))
    assert e is not None
    assert e.line_addr == 0x100
    assert e.expected_complete == 410
    assert len(e.waiters) == 1
    assert e.waiters[0].warp_id == 0
    assert e.waiters[0].dst_regs == ("r1",)

def test_allocate_when_full_returns_none():
    p = MSHRPool(slots=2)
    p.allocate(line_addr=0x100, issued_at=0, expected=400, warp_id=0, dst_regs=("r1",))
    p.allocate(line_addr=0x200, issued_at=0, expected=400, warp_id=0, dst_regs=("r1",))
    assert p.is_full()
    e = p.allocate(line_addr=0x300, issued_at=0, expected=400, warp_id=0, dst_regs=("r1",))
    assert e is None

def test_find_returns_existing_entry_for_same_line():
    p = MSHRPool(slots=4)
    p.allocate(line_addr=0x100, issued_at=10, expected=410, warp_id=0, dst_regs=("r1",))
    e = p.find_for_line(0x100)
    assert e is not None
    assert e.line_addr == 0x100

def test_add_waiter_merges_into_existing_entry():
    p = MSHRPool(slots=4)
    e = p.allocate(line_addr=0x100, issued_at=10, expected=410,
                   warp_id=0, dst_regs=("r1",))
    e.add_waiter(warp_id=1, dst_regs=("r2", "r3"))
    assert len(e.waiters) == 2
    assert e.waiters[1].warp_id == 1
    assert e.waiters[1].dst_regs == ("r2", "r3")

def test_release_frees_slot():
    p = MSHRPool(slots=2)
    e1 = p.allocate(line_addr=0x100, issued_at=0, expected=400, warp_id=0, dst_regs=("r1",))
    p.allocate(line_addr=0x200, issued_at=0, expected=400, warp_id=0, dst_regs=("r1",))
    assert p.is_full()
    p.release(e1)
    assert not p.is_full()
    e3 = p.allocate(line_addr=0x300, issued_at=0, expected=400, warp_id=0, dst_regs=("r1",))
    assert e3 is not None

def test_active_entries_iterates():
    p = MSHRPool(slots=4)
    p.allocate(line_addr=0x100, issued_at=0, expected=400, warp_id=0, dst_regs=("r1",))
    p.allocate(line_addr=0x200, issued_at=5, expected=405, warp_id=0, dst_regs=("r1",))
    addrs = sorted(e.line_addr for e in p.active_entries())
    assert addrs == [0x100, 0x200]
