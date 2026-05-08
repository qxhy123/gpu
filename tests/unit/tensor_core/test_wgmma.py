def test_wgmma_queue_basic_lifecycle():
    from gpusim.core.tensor_core.wgmma import WgmmaQueue, InflightWgmma
    q = WgmmaQueue(capacity=2)
    f1 = InflightWgmma(issued_at=0, completion_at=32, dst_regs=(("d0",),))
    assert q.try_push(f1) is True
    f2 = InflightWgmma(issued_at=4, completion_at=36, dst_regs=(("d4",),))
    assert q.try_push(f2) is True
    f3 = InflightWgmma(issued_at=8, completion_at=40, dst_regs=(("d8",),))
    assert q.try_push(f3) is False  # full

    gid = q.commit_group()
    assert gid == 0
    assert q.committed_groups == [0]
    # all in_flight commit to same group
    assert all(f.commit_group_id == 0 for f in q.in_flight)


def test_wgmma_queue_drain_on_wait():
    from gpusim.core.tensor_core.wgmma import WgmmaQueue, InflightWgmma
    q = WgmmaQueue(capacity=4)
    f1 = InflightWgmma(issued_at=0, completion_at=32, dst_regs=(("d0",), ("d4",)))
    f2 = InflightWgmma(issued_at=4, completion_at=40, dst_regs=(("d8",),))
    q.try_push(f1); q.try_push(f2)
    q.commit_group()                       # group 0 covers both
    # at cycle 32, only f1 done — group not drainable yet
    drained_at_32 = q.drain_completed_groups(now=32)
    assert drained_at_32 == []             # f2 not done
    # at cycle 40, both done — group drains
    drained_at_40 = q.drain_completed_groups(now=40)
    assert len(drained_at_40) == 1
    assert drained_at_40[0] == 0           # group_id
    assert q.committed_groups == []
    assert q.in_flight == []


def test_wgmma_queue_wait_group_n_blocks_until_count():
    from gpusim.core.tensor_core.wgmma import WgmmaQueue
    q = WgmmaQueue(capacity=4)
    q.committed_groups = [0, 1, 2]
    assert q.must_wait(target_n=3) is False
    assert q.must_wait(target_n=2) is True
    assert q.must_wait(target_n=1) is True


def test_execute_wgmma_for_group_fp16_matches_numpy():
    """4 warps × 32 lanes cooperate on m64n128k16 FP16 wgmma."""
    import numpy as np
    from gpusim.core.exec import WarpFnState
    from gpusim.core.tensor_core.wgmma import execute_wgmma_for_group
    from gpusim.core.tensor_core.mma_spec import parse_mma_op
    from gpusim.frontend.ir import Reg, RegGroup, PtxType

    spec = parse_mma_op("wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16")
    rng = np.random.RandomState(0)
    A = rng.randn(64, 16).astype(np.float16)
    B = rng.randn(16, 128).astype(np.float16)
    warps = [WarpFnState(warp_size=32, tids=tuple(range(32))) for _ in range(4)]

    dst_regs_per_warp = tuple(
        RegGroup(regs=tuple(Reg(name=f"d{w}_{j}", type=PtxType.f32) for j in range(64)))
        for w in range(4)
    )
    c_regs_per_warp = tuple(
        RegGroup(regs=tuple(Reg(name=f"c{w}_{j}", type=PtxType.f32) for j in range(64)))
        for w in range(4)
    )
    for warp_w, w in enumerate(warps):
        for lane in range(32):
            for j in range(64):
                w.threads[lane].set_f32(f"c{warp_w}_{j}", 0.0)

    execute_wgmma_for_group(
        spec=spec, warps=warps,
        a_smem_array=A, b_smem_array=B,
        dst_per_warp=dst_regs_per_warp, c_per_warp=c_regs_per_warp,
    )

    # Reconstruct D from 4 warps × 32 lanes × 64 regs per spec §4.2:
    # warp w, lane i, reg %dj -> D[w*16 + i/2][(i%2)*64 + j]
    D = np.zeros((64, 128), dtype=np.float32)
    for warp_w in range(4):
        for lane in range(32):
            row = warp_w * 16 + lane // 2
            col_base = (lane % 2) * 64
            for j in range(64):
                D[row, col_base + j] = warps[warp_w].threads[lane].get_f32(f"d{warp_w}_{j}")
    expected = (A.astype(np.float32) @ B.astype(np.float32))
    assert np.allclose(D, expected, atol=1e-2), f"max diff = {np.max(np.abs(D - expected))}"
