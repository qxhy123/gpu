from gpusim.core.functional_units import FUSet, FUKind
from gpusim.config.loader import load_default

def test_fu_classify_op():
    s = FUSet(load_default().sm.fu)
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
    s = FUSet(load_default().sm.fu)
    assert s.is_free(FUKind.FP32, now=0)
    s.reserve(FUKind.FP32, now=0, occupancy_cycles=1)
    assert s.is_free(FUKind.FP32, now=0) is False
    assert s.is_free(FUKind.FP32, now=1)

def test_latency_lookup():
    s = FUSet(load_default().sm.fu)
    assert s.result_latency("add.f32") == 4
    assert s.result_latency("mad.f32") == 4
    assert s.result_latency("ld.global.f32") == 400
    assert s.result_latency("ld.shared.f32") == 20
    assert s.result_latency("st.global.f32") == 0
    assert s.result_latency("bra") == 1


def test_fukind_has_tc():
    from gpusim.core.functional_units import FUKind
    assert FUKind.TC.value == "tc"


def test_classify_mma_to_tc():
    from gpusim.core.functional_units import FUSet, FUKind
    from gpusim.config.schema import FUConfig
    fus = FUSet(FUConfig())
    assert fus.classify("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32") is FUKind.TC
    assert fus.classify("wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16") is FUKind.TC


def test_classify_cp_async_bulk_to_lsu():
    from gpusim.core.functional_units import FUSet, FUKind
    from gpusim.config.schema import FUConfig
    fus = FUSet(FUConfig())
    assert fus.classify("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes") is FUKind.LSU


def test_classify_mbarrier_to_sync():
    from gpusim.core.functional_units import FUSet, FUKind
    from gpusim.config.schema import FUConfig
    fus = FUSet(FUConfig())
    assert fus.classify("mbarrier.arrive.shared::cta") is FUKind.SYNC


def test_tensor_core_config_default():
    from gpusim.config.schema import SMConfig
    cfg = SMConfig()
    assert cfg.tensor_core.tc_mma_latency == 8
    assert cfg.tensor_core.tc_mma_occupancy == 1
    assert cfg.tensor_core.tc_wgmma_latency == 32
    assert cfg.tensor_core.tc_wgmma_occupancy == 4
    assert cfg.tensor_core.wgmma_queue_capacity == 16
