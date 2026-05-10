def test_dist_all_reduce_numpy():
    import numpy as np
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)
    t = np.full(16, 1.0, dtype=np.float32)
    dist.all_reduce(t, op="sum")
    np.testing.assert_array_equal(t, np.full(16, 4.0, dtype=np.float32))
    dist.destroy_process_group()


def test_dist_all_gather_numpy():
    import numpy as np
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)
    t = np.full(8, 1.0, dtype=np.float32)
    tensor_list = [np.zeros(8, dtype=np.float32) for _ in range(4)]
    dist.all_gather(tensor_list, t)
    # All gathered tensors should be 1.0 (uniform)
    for tl in tensor_list:
        np.testing.assert_array_equal(tl, np.full(8, 1.0, dtype=np.float32))
    dist.destroy_process_group()


def test_dist_broadcast_numpy():
    import numpy as np
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)
    t = np.arange(8, dtype=np.float32)
    dist.broadcast(t, src=0)
    # broadcast — buffer unchanged at root
    np.testing.assert_array_equal(t, np.arange(8, dtype=np.float32))
    dist.destroy_process_group()


def test_dist_reduce_scatter_numpy():
    import numpy as np
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)
    output = np.zeros(8, dtype=np.float32)
    input_list = [np.full(8, 1.0, dtype=np.float32) for _ in range(4)]
    dist.reduce_scatter(output, input_list, op="sum")
    np.testing.assert_array_equal(output, np.full(8, 4.0, dtype=np.float32))
    dist.destroy_process_group()


def test_dist_send_recv_numpy():
    import numpy as np
    import gpusim.dist as dist
    dist.init_process_group(world_size=2, rank=0)
    t = np.arange(16, dtype=np.float32)
    dist.send(t, dst=1)
    dist.recv(t, src=1)   # dummy paired recv
    dist.destroy_process_group()


def test_dist_barrier_noop():
    import gpusim.dist as dist
    dist.init_process_group(world_size=4, rank=0)
    dist.barrier()   # should not raise
    dist.destroy_process_group()
