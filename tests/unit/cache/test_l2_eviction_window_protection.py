"""Tests for CacheSet.install with window protection (Phase 9 T7).

CacheSet.install should accept `requesting_stream_id` and
`line_in_window_check` kwargs so callers can prevent eviction of cache lines
that belong to a different stream's protected window.
"""
from gpusim.core.cache.line import CacheSet, CacheLine


def _make_protected_set(n_ways: int = 4) -> CacheSet:
    """Helper: build a fully-populated CacheSet where all lines are owned by
    stream 5 and marked in_window=True."""
    cs = CacheSet(ways=n_ways)
    for i, line in enumerate(cs._lines):
        line.tag = i
        line.valid = True
        line.dirty = False
        line.owner_stream_id = 5
        line.in_window = True
        line.lru_pos = n_ways - 1 - i  # line[0] = LRU
    return cs


def _is_in_window(line: CacheLine, set_idx: int) -> bool:
    return line.in_window


# ---------------------------------------------------------------------------
# T7-A: install skips lines protected by another stream's window
# ---------------------------------------------------------------------------

def test_cache_set_install_skips_window_protected_lines_from_other_stream():
    """CacheSet.install skips lines protected by another stream's window."""
    cs = CacheSet(ways=4)

    # Fill all 4 ways
    cs.install(tag=10, dirty=False, origin_sm=0)
    cs.install(tag=20, dirty=False, origin_sm=0)
    cs.install(tag=30, dirty=False, origin_sm=0)
    cs.install(tag=40, dirty=False, origin_sm=0)
    # After 4 installs: tag=10 is LRU (lru_pos=3)

    # Mark the LRU valid line (tag=10) as protected by stream 5
    lru_line = max(cs._lines, key=lambda l: l.lru_pos)
    assert lru_line.tag == 10, f"Expected tag=10 as LRU, got {lru_line.tag}"
    lru_line.owner_stream_id = 5
    lru_line.in_window = True

    # Stream 7 tries to install — should NOT evict the protected line (tag=10)
    # Instead it should evict the next-LRU unprotected line (tag=20, lru_pos=2)
    new_line = cs.install(tag=99, dirty=False, origin_sm=7,
                          requesting_stream_id=7,
                          line_in_window_check=_is_in_window)

    # The protected line (owner_stream_id=5, tag=10) must still be present
    protected = next((l for l in cs._lines if l.owner_stream_id == 5), None)
    assert protected is not None, "Protected line was incorrectly evicted"
    assert protected.in_window is True
    assert protected.tag == 10

    # Tag 99 must now be installed
    assert cs.find(99) is not None, "New tag was not installed"


def test_cache_set_install_skips_only_other_stream_lines():
    """A protected line owned by the *requesting* stream is still evictable."""
    cs = CacheSet(ways=2)
    cs.install(tag=1, dirty=False, origin_sm=7)
    cs.install(tag=2, dirty=False, origin_sm=7)

    # Mark both lines as in_window for stream 7
    for line in cs._lines:
        line.owner_stream_id = 7
        line.in_window = True

    # Stream 7 installs — it may evict its own protected lines
    result = cs.install(tag=99, dirty=False, origin_sm=7,
                        requesting_stream_id=7,
                        line_in_window_check=_is_in_window)
    # Should succeed (result is the evicted line, not None)
    assert result is not None
    assert cs.find(99) is not None


# ---------------------------------------------------------------------------
# T7-B: returns None when all ways are protected by other streams
# ---------------------------------------------------------------------------

def test_cache_set_install_returns_none_when_all_protected():
    """install returns None when every way is protected by another stream."""
    cs = CacheSet(ways=2)
    cs.install(tag=1, dirty=False, origin_sm=5)
    cs.install(tag=2, dirty=False, origin_sm=5)

    for line in cs._lines:
        line.owner_stream_id = 5
        line.in_window = True

    result = cs.install(tag=999, dirty=False, origin_sm=7,
                        requesting_stream_id=7,
                        line_in_window_check=_is_in_window)
    assert result is None, "Expected None when all lines are window-protected"
    # Original lines must remain
    assert cs.find(1) is not None
    assert cs.find(2) is not None


# ---------------------------------------------------------------------------
# T7-C: backward-compat — no kwargs → original LRU eviction
# ---------------------------------------------------------------------------

def test_cache_set_install_backward_compat_no_window():
    """Calling install without window kwargs preserves original behaviour."""
    cs = CacheSet(ways=4)
    cs.install(tag=0xA, dirty=False)
    cs.install(tag=0xB, dirty=False)
    cs.install(tag=0xC, dirty=True)
    cs.install(tag=0xD, dirty=False)
    # 0xA is LRU; installing 0xE should evict it
    victim = cs.install(tag=0xE, dirty=False)
    assert victim is not None
    assert victim.tag == 0xA
    assert cs.find(0xE) is not None


# ---------------------------------------------------------------------------
# T7-D: free way found → no eviction needed, window check irrelevant
# ---------------------------------------------------------------------------

def test_cache_set_install_free_way_ignores_window():
    """If a free way exists, window protection is not consulted."""
    cs = CacheSet(ways=4)
    cs.install(tag=1, dirty=False, origin_sm=5)

    # Mark the installed line as protected by a different stream
    for line in cs._lines:
        if line.valid:
            line.owner_stream_id = 5
            line.in_window = True

    # Plenty of free ways → should succeed without touching the protected line
    result = cs.install(tag=2, dirty=False, origin_sm=7,
                        requesting_stream_id=7,
                        line_in_window_check=_is_in_window)
    assert result is None  # free-way path returns None (no eviction)
    assert cs.find(2) is not None
    assert cs.find(1) is not None  # protected line still present


# ---------------------------------------------------------------------------
# T7-E: line_in_window_check=None → no filtering (all ways candidates)
# ---------------------------------------------------------------------------

def test_cache_set_install_none_check_evicts_lru():
    """With line_in_window_check=None even 'in_window' lines are evictable."""
    cs = _make_protected_set(n_ways=2)
    # lru_pos order: line[0] has lru_pos=1 (LRU), line[1] has lru_pos=0 (MRU)
    lru = max(cs._lines, key=lambda l: l.lru_pos)
    lru_tag = lru.tag

    victim = cs.install(tag=999, dirty=False, origin_sm=0,
                        requesting_stream_id=0,
                        line_in_window_check=None)
    assert victim is not None
    assert victim.tag == lru_tag


# ---------------------------------------------------------------------------
# T9: SubCore gmem path records hit/in_window in GmemEvent (end-to-end)
# ---------------------------------------------------------------------------

def test_gmem_event_carries_hit_and_in_window():
    """End-to-end: gmem load via SubCore records hit and in_window in GmemEvent."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    src = """
.entry test(.param .u64 IN, .param .u64 OUT) {
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    ld.param.u64 %rd0, [IN];
    ld.param.u64 %rd1, [OUT];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd2, %r1;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.u32 %r2, [%rd3];
    add.u64 %rd4, %rd1, %rd2;
    st.global.u32 [%rd4], %r2;
    ret;
}
"""
    import numpy as np
    n = 32
    IN = np.arange(n, dtype=np.uint32)
    OUT = np.zeros(n, dtype=np.uint32)
    cfg = load_default()
    s = Stream()
    s.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
             params={"IN": IN, "OUT": OUT}, kernel_name="copy", config=cfg)
    multi_res = gpusim.synchronize(streams=[s], config=cfg)

    rec = multi_res._recorder
    if rec is not None:
        gmem_events = rec.gmem_accesses()
        # GmemEvent fields hit/in_window must exist and be booleans
        assert gmem_events, "Expected at least one GmemEvent from ld.global/st.global"
        ev = gmem_events[0]
        assert hasattr(ev, "hit"), "GmemEvent missing 'hit' field"
        assert hasattr(ev, "in_window"), "GmemEvent missing 'in_window' field"
        assert isinstance(ev.hit, bool), f"hit should be bool, got {type(ev.hit)}"
        assert isinstance(ev.in_window, bool), f"in_window should be bool, got {type(ev.in_window)}"
        # First ld.global access should be a miss (cold cache)
        assert ev.hit is False, (
            f"First ld.global on cold cache should record hit=False, got {ev.hit}"
        )
        # No stream window registered → in_window should be False
        assert ev.in_window is False, (
            f"No stream window registered → in_window should be False, got {ev.in_window}"
        )
