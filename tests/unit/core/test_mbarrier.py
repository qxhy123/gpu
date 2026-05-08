def test_mbarrier_init_and_arrive():
    from gpusim.core.mbarrier import MbarrierPool
    pool = MbarrierPool()
    pool.init(smem_addr=0, expected=4)
    pool.arrive(smem_addr=0)
    pool.arrive(smem_addr=0)
    assert pool.try_wait(smem_addr=0, expected_phase=0) is False  # not yet flipped


def test_mbarrier_flip_when_count_reached():
    from gpusim.core.mbarrier import MbarrierPool
    pool = MbarrierPool()
    pool.init(smem_addr=0, expected=4)
    for _ in range(4):
        pool.arrive(smem_addr=0)
    pool.tick(now=10)   # tick processes pending and may flip
    bar = pool._barriers[0]
    assert bar.phase == 1
    assert pool.try_wait(smem_addr=0, expected_phase=0) is True


def test_mbarrier_arrive_tx_drains_at_completion():
    from gpusim.core.mbarrier import MbarrierPool
    pool = MbarrierPool()
    pool.init(smem_addr=0, expected=2)
    pool.arrive_tx(smem_addr=0, tx_bytes=1024, completion_at=20)
    # before tick at cycle 20, no arrive yet
    pool.tick(now=10)
    assert pool._barriers[0].arrived_count == 0
    pool.tick(now=20)
    assert pool._barriers[0].arrived_count == 1
    # second arrive (regular)
    pool.arrive(smem_addr=0)
    pool.tick(now=21)
    assert pool._barriers[0].phase == 1


def test_mbarrier_try_wait_phase_logic():
    from gpusim.core.mbarrier import MbarrierPool
    pool = MbarrierPool()
    pool.init(smem_addr=0, expected=1)
    # phase 0 not yet flipped
    assert pool.try_wait(smem_addr=0, expected_phase=0) is False
    pool.arrive(smem_addr=0); pool.tick(now=1)
    # phase flipped to 1
    assert pool.try_wait(smem_addr=0, expected_phase=0) is True
    # waits with phase=1 (next phase) return False until next arrive
    assert pool.try_wait(smem_addr=0, expected_phase=1) is False
