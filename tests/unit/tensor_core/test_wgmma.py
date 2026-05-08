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
