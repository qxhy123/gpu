def test_cluster_barrier_pool_arrive_partial_no_flip():
    from gpusim.core.cluster import ClusterBarrierPool
    pool = ClusterBarrierPool(expected=4)
    assert pool.arrive(0) is False
    assert pool.arrive(1) is False
    assert pool.phase == 0


def test_cluster_barrier_pool_arrive_complete_flips():
    from gpusim.core.cluster import ClusterBarrierPool
    pool = ClusterBarrierPool(expected=4)
    for r in range(4):
        completed = pool.arrive(r)
    assert completed is True
    assert pool.phase == 1
    assert pool.arrived_mask == 0


def test_cluster_barrier_pool_is_released():
    from gpusim.core.cluster import ClusterBarrierPool
    pool = ClusterBarrierPool(expected=2)
    assert pool.is_released(captured_phase=0) is False
    pool.arrive(0); pool.arrive(1)
    assert pool.is_released(captured_phase=0) is True


def test_cluster_barrier_pool_idempotent_rank_arrive():
    """Same rank arriving twice doesn't double-count."""
    from gpusim.core.cluster import ClusterBarrierPool
    pool = ClusterBarrierPool(expected=4)
    pool.arrive(0)
    pool.arrive(0)
    pool.arrive(1)
    pool.arrive(2)
    pool.arrive(3)
    assert pool.phase == 1
