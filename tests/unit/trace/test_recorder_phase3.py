"""Tests for the 4 new trace event recorder methods added in Task 26."""
from gpusim.trace.recorder import Recorder


def test_recorder_records_mma_event():
    r = Recorder()
    r.mma(cycle=10, warp_id=0, pc=5, precision="f16",
          shape_m=16, shape_n=8, shape_k=16, accum_dtype="f32",
          flops_count=4096)
    assert len(r.mma_events) == 1
    e = r.mma_events[0]
    assert e.cycle == 10 and e.precision == "f16"
    assert e.flops_count == 4096


def test_recorder_records_wgmma_event():
    r = Recorder()
    r.wgmma(kind="ISSUE", cycle=20, warp_group_id=0, pc=10,
            precision="f16", shape_m=64, shape_n=128, shape_k=16,
            accum_dtype="f32", commit_group_id=-1, wait_n=-1, completion_at=52)
    assert len(r.wgmma_events) == 1
    e = r.wgmma_events[0]
    assert e.kind == "ISSUE" and e.completion_at == 52


def test_recorder_records_tma_event():
    r = Recorder()
    r.tma(cycle=30, completion_at=80, smem_dst=0, gmem_base=0x1000,
          dim_x=128, dim_y=64, bytes_total=16384, n_cache_lines=128,
          mbarrier_addr=0x800)
    assert len(r.tma_events) == 1


def test_recorder_records_mbarrier_event():
    r = Recorder()
    r.mbarrier(kind="INIT", cycle=0, cta_id=0, smem_addr=0x800,
               expected=4, arrived=0, phase=0, pred_result=False)
    r.mbarrier(kind="FLIP", cycle=85, cta_id=0, smem_addr=0x800,
               expected=4, arrived=4, phase=1, pred_result=False)
    assert len(r.mbarrier_events) == 2
