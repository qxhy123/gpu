from gpusim.core.functional_units import FUSet, FUKind
from gpusim.config.loader import load_default

def test_fu_classify_op():
    s = FUSet(load_default().fu)
    assert s.classify("add.s32") is FUKind.INT
    assert s.classify("add.f32") is FUKind.FP32
    assert s.classify("mad.f32") is FUKind.FP32
    assert s.classify("ld.global.f32") is FUKind.LSU
    assert s.classify("st.shared.f32") is FUKind.LSU
    assert s.classify("bra") is FUKind.BRU
    assert s.classify("@%p1 bra L1") is FUKind.BRU  # already-stripped op
    assert s.classify("bar.sync") is FUKind.SYNC
    assert s.classify("setp.lt.s32") is FUKind.INT

def test_fu_busy_state():
    s = FUSet(load_default().fu)
    assert s.is_free(FUKind.FP32, now=0)
    s.reserve(FUKind.FP32, now=0, occupancy_cycles=1)
    assert s.is_free(FUKind.FP32, now=0) is False
    assert s.is_free(FUKind.FP32, now=1)

def test_latency_lookup():
    s = FUSet(load_default().fu)
    assert s.result_latency("add.f32") == 4
    assert s.result_latency("mad.f32") == 4
    assert s.result_latency("ld.global.f32") == 400
    assert s.result_latency("ld.shared.f32") == 20
    assert s.result_latency("st.global.f32") == 0
    assert s.result_latency("bra") == 1
