import pytest
from gpusim.core.cache.line import CacheLine, CacheSet

def test_cacheline_fields():
    line = CacheLine(tag=0xDEAD, valid=True, dirty=False, lru_pos=0)
    assert line.tag == 0xDEAD
    assert line.valid is True
    assert line.dirty is False
    assert line.lru_pos == 0

def test_cacheset_starts_empty():
    s = CacheSet(ways=4)
    assert s.find(0xCAFE) is None
    assert all(not w.valid for w in s.ways)

def test_cacheset_install_makes_mru():
    s = CacheSet(ways=4)
    s.install(tag=0xAAAA, dirty=False)
    line = s.find(0xAAAA)
    assert line is not None
    assert line.tag == 0xAAAA
    assert line.lru_pos == 0
    assert line.valid is True

def test_cacheset_lru_update_on_hit():
    s = CacheSet(ways=4)
    s.install(tag=0xA, dirty=False)
    s.install(tag=0xB, dirty=False)
    s.install(tag=0xC, dirty=False)
    s.install(tag=0xD, dirty=False)
    # last installed = MRU
    assert s.find(0xD).lru_pos == 0
    assert s.find(0xA).lru_pos == 3
    # touch oldest → it becomes MRU
    s.touch(s.find(0xA))
    assert s.find(0xA).lru_pos == 0
    assert s.find(0xD).lru_pos == 1

def test_cacheset_eviction_picks_lru():
    s = CacheSet(ways=4)
    s.install(tag=0xA, dirty=False)
    s.install(tag=0xB, dirty=False)
    s.install(tag=0xC, dirty=True)
    s.install(tag=0xD, dirty=False)
    # tag A is LRU (lru_pos==3); installing E evicts it
    victim = s.install(tag=0xE, dirty=False)
    assert victim is not None
    assert victim.tag == 0xA
    assert s.find(0xA) is None
    assert s.find(0xE).lru_pos == 0

def test_cacheset_dirty_eviction_returns_dirty_victim():
    s = CacheSet(ways=2)
    s.install(tag=0xA, dirty=True)
    s.install(tag=0xB, dirty=False)
    victim = s.install(tag=0xC, dirty=False)
    assert victim is not None
    assert victim.dirty is True
    assert victim.tag == 0xA
