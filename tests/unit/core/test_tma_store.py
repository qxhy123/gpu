def test_inflight_bulk_store_dataclass():
    from gpusim.core.tma_store import InflightBulkStore
    f = InflightBulkStore(issued_at=0, completion_at=20, bytes_total=1024)
    assert f.commit_group_id == -1


def test_bulk_store_queue_lifecycle():
    from gpusim.core.tma_store import BulkStoreQueue, InflightBulkStore
    q = BulkStoreQueue(capacity=2)
    f1 = InflightBulkStore(issued_at=0, completion_at=10, bytes_total=128)
    f2 = InflightBulkStore(issued_at=2, completion_at=14, bytes_total=128)
    f3 = InflightBulkStore(issued_at=4, completion_at=18, bytes_total=128)
    assert q.try_push(f1) is True
    assert q.try_push(f2) is True
    assert q.try_push(f3) is False
    gid = q.commit_group()
    assert gid == 0
    assert all(f.commit_group_id == 0 for f in q.in_flight)
    drained = q.drain_completed_groups(now=10)
    assert drained == []
    drained = q.drain_completed_groups(now=14)
    assert drained == [0]
    assert q.in_flight == []


def test_bulk_store_queue_must_wait():
    from gpusim.core.tma_store import BulkStoreQueue
    q = BulkStoreQueue(capacity=4)
    q.committed_groups = [0, 1, 2]
    assert q.must_wait(target_n=3) is False
    assert q.must_wait(target_n=2) is True
    assert q.must_wait(target_n=0) is True


def test_do_bulk_store_2d_copies_correct_bytes():
    import numpy as np
    from gpusim.core.exec import GlobalMemory, SharedMemory
    from gpusim.core.tma import TmaDescriptor
    from gpusim.core.tma_store import do_bulk_store_2d

    s = SharedMemory(size_bytes=8192)
    s.allocate_cta(0, 8192)
    src_arr = np.arange(64 * 32, dtype=np.float16).reshape(64, 32)
    smem_src_off = 0
    s._cta[0][smem_src_off:smem_src_off + src_arr.nbytes] = (
        np.frombuffer(src_arr.tobytes(), dtype=np.uint8))

    g = GlobalMemory()
    dest = np.zeros(64 * 32, dtype=np.float16)
    g.bind("OUT", dest)
    desc = TmaDescriptor(gmem_base=g.address_of("OUT"), dim_x=32, dim_y=64,
                          stride_y=32, elem_bytes=2)
    do_bulk_store_2d(gmem=g, smem=s, cta_id=0, smem_src=smem_src_off, desc=desc)
    assert (dest.reshape(64, 32) == src_arr).all()
