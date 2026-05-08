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
