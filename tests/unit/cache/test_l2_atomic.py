"""Tests for L2Cache.atomic_op integration with L2AtomicQueue (Phase 6 T12)."""
from __future__ import annotations


def test_l2_atomic_op_serializes_same_line():
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.config.schema import CacheConfig

    class _NoOpHbm:
        def request(self, line_addr, now): return now + 100
        def write_request(self, line_addr, now): return now + 100

    cfg = CacheConfig()
    l2 = L2Cache(cfg, _NoOpHbm())
    c1 = l2.atomic_op(line_addr=0x1000, sm_id=0, op="add", op_kind="atom", now=0)
    c2 = l2.atomic_op(line_addr=0x1000, sm_id=1, op="add", op_kind="atom", now=0)
    assert c2 > c1
    assert (c2 - c1) >= cfg.atomic_op_latency


def test_l2_atomic_op_different_lines_parallel():
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.config.schema import CacheConfig

    class _NoOpHbm:
        def request(self, line_addr, now): return now + 100
        def write_request(self, line_addr, now): return now + 100

    cfg = CacheConfig()
    l2 = L2Cache(cfg, _NoOpHbm())
    c1 = l2.atomic_op(line_addr=0x1000, sm_id=0, op="add", op_kind="atom", now=0)
    c2 = l2.atomic_op(line_addr=0x2000, sm_id=1, op="add", op_kind="atom", now=0)
    assert c1 == c2
