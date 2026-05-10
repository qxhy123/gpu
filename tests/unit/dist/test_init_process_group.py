def test_init_process_group_sets_state():
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)
    assert dist.get_rank() == 0
    assert dist.get_world_size() == 4
    dist.destroy_process_group()
    assert dist.get_world_size() == 0


def test_init_process_group_with_rank_2():
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=2)
    assert dist.get_rank() == 2
    dist.destroy_process_group()
